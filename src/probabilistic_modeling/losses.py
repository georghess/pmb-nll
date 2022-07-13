from collections import defaultdict
from math import comb
from math import factorial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from core.fastmurty.mhtdaClink import (allocateWorkvarsforDA,
                                       deallocateWorkvarsforDA, mhtda, sparse)
from core.fastmurty.mhtdaClink import sparsifyByRow as sparsify
from scipy.optimize import linear_sum_assignment
from torch.distributions.multivariate_normal import MultivariateNormal

from probabilistic_modeling.modeling_utils import (
    clamp_log_variance, covariance_output_to_cholesky)


def reshape_box_preds(preds, num_classes):
    """
    Tiny helper function to reshape box predictions from [numpreds,classes*boxdim] to [numpreds,classes,boxdim]
    """
    num_preds, *_ = preds.shape
    if num_preds == 0:
        return preds

    if len(preds.shape) == 2:
        preds = preds.unsqueeze(1)
        if preds.shape[-1] > num_classes:  # if box predicted per class
            preds = preds.reshape(num_preds, num_classes, -1)
        else:
            preds = preds.repeat(1, num_classes, 1)

    return preds


def run_murtys(cost_matrix: torch.tensor, nsolutions: int):
    """
    Run fastmurtys given cost_matrix and number of assignments to search for.
    Returns associations and costs.
    Based on example_simplest.py in fastmurty.
    """
    # make all costs negative for algo to work properly
    cost_matrix_max = cost_matrix.max()
    if cost_matrix_max >= 0:
        cost_matrix = cost_matrix - (cost_matrix_max + 1)
    cost_matrix = cost_matrix.detach().numpy()
    nrows, ncolumns = cost_matrix.shape
    # sparse cost matrices only include a certain number of elements
    # the rest are implicitly infinity
    # in this case, the sparse matrix includes all elements
    # The sparse and dense versions are compiled differently (see the Makefile).
    # The variable "sparse" in mhtdaClink needs to match the version compiled
    cost_matrix_to_use = sparsify(cost_matrix, ncolumns) if sparse else cost_matrix

    # mhtda is set up to potentially take multiple input hypotheses for both rows and columns
    # input hypotheses specify a subset of rows or columns.
    # In this case, we just want to use the whole matrix.
    row_priors = np.ones((1, nrows), dtype=np.bool8)
    col_priors = np.ones((1, ncolumns), dtype=np.bool8)
    # Each hypothesis has a relative weight too.
    # These values don't matter if there is only one hypothesis...
    row_prior_weights = np.zeros(1)
    col_prior_weights = np.zeros(1)

    # The mhtda function modifies preallocated outputs rather than
    # allocating new ones. This is slightly more efficient for repeated use
    # within a tracker.
    # The cost of each returned association:
    out_costs = np.zeros(nsolutions)
    # The row-column pairs in each association:
    # Generally there will be less than nrows+ncolumns pairs in an association.
    # The unused pairs are currently set to (-2, -2)
    out_associations = np.zeros((nsolutions, nrows + ncolumns, 2), dtype=np.int32)
    # variables needed within the algorithm (a C function sets this up):
    workvars = allocateWorkvarsforDA(nrows, ncolumns, nsolutions)

    # run!
    mhtda(
        cost_matrix_to_use,
        row_priors,
        row_prior_weights,
        col_priors,
        col_prior_weights,
        out_associations,
        out_costs,
        workvars,
    )

    deallocateWorkvarsforDA(workvars)

    return out_associations, out_costs


def compute_negative_log_likelihood(
    box_scores: torch.tensor,
    box_regs: torch.tensor,
    box_covars: torch.tensor,
    gt_box: torch.tensor,
    gt_class: torch.tensor,
    image_size: List[int],
    reg_distribution: torch.distributions.distribution.Distribution,
    associations: np.ndarray,
    device: torch.device,
    intensity_func=lambda x: 0.00000001,
    scores_have_bg_cls=False,
    target_delta=None,
    pred_delta=None,
    pred_delta_chol=None,
):
    """Compute NLL for given associations.

    Args:
        box_scores (torch.tensor): [description]
        box_regs (torch.tensor): [description]
        box_covars (torch.tensor): [description]
        gt_box (torch.tensor): [description]
        gt_class (torch.tensor): [description]
        image_size (List[int]): [description]
        reg_distribution (torch.distributions.distribution.Distribution): [description]
        associations (np.ndarray[np.int32]): [description]
        device (torch.device): [description]
        intensity_func ([type], optional): [description]. Defaults to lambdax:0.00000001.

    Returns:
        [type]: [description]
    """
    if type(image_size) is not torch.tensor:
        image_size = torch.tensor(image_size)
    img_size = image_size.unsqueeze(0).to(device)
    existance_prob = 1 - box_scores[:, -1]

    num_preds, num_classes = box_scores.shape
    if scores_have_bg_cls:
        num_classes -= 1  # do not count background class
    num_gt, _ = gt_box.shape

    out_dict = defaultdict(list)
    out_dict.update(
        {
            "matched_bernoulli": [],
            "unmatched_bernoulli": [],
            "matched_ppp": [],
            "matched_bernoulli_reg": [],
            "matched_bernoulli_cls": [],
            "num_matched_bernoulli": [],
            "num_unmatched_bernoulli": [],
            "num_matched_ppp": [],
            "ppp_integral": None,
        }
    )

    nll = torch.zeros(len(associations), dtype=torch.float64, device=device)
    for a, association in enumerate(associations):
        log_matched_bernoulli = torch.tensor(0, dtype=torch.float64, device=device)
        log_unmatched_bernoulli = torch.tensor(0, dtype=torch.float64, device=device)
        log_poisson = torch.tensor(0, dtype=torch.float64, device=device)
        log_matched_regression = torch.tensor(0, dtype=torch.float64, device=device)
        log_matched_classification = torch.tensor(0, dtype=torch.float64, device=device)
        num_matched_bernoulli = 0
        num_unmatched_bernoulli = 0
        num_matched_ppp = 0
        log_matched_bernoulli_regs = []
        log_matched_bernoulli_cls = []
        log_unmatched_bernoullis = []
        log_matched_ppps = []

        for pair in association:
            pred = pair[0]
            gt = pair[1]
            if (
                0 <= pred < num_preds
            ) and gt >= 0:  # if bernoulli was assigned to a GT element
                num_matched_bernoulli += 1
                assigned_gt = gt
                k = pred
                gt_c = gt_class[assigned_gt]

                if scores_have_bg_cls:
                    r = existance_prob[k]
                else:
                    r = box_scores[k, gt_c]

                covar = box_covars[k, gt_c]
                
                if target_delta is None:
                    covar = box_covars[k, gt_c]
                    dist = reg_distribution(box_regs[k, gt_c, :], covar)
                    regression = dist.log_prob(gt_box[assigned_gt, :]).sum()
                    classification = torch.log(box_scores[k, gt_c])
                else:
                    covar = pred_delta_chol[k, gt_c]
                    dist = reg_distribution(pred_delta[k, gt_c, :], covar)
                    regression = dist.log_prob(target_delta[k, assigned_gt, :]).sum()
                    classification = torch.log(box_scores[k, gt_c])
                log_f = regression + classification
                # Save stats
                log_matched_bernoulli_regs.append(-regression.squeeze().item())
                log_matched_bernoulli_cls.append(-classification.squeeze().item())

                # Update total bernoulli component
                log_matched_bernoulli = log_matched_bernoulli + log_f.squeeze()
                log_matched_regression = log_matched_regression + regression.squeeze()
                log_matched_classification = (
                    log_matched_classification + classification.squeeze()
                )

            elif (
                0 <= pred < num_preds
            ) and gt == -1:  # if bernoulli was not assigned to a GT element
                num_unmatched_bernoulli += 1
                k = pred
                if scores_have_bg_cls:
                    log_f = torch.log(1 - existance_prob[k])
                else:
                    log_f = torch.log(1 - box_scores[k].max())
                log_unmatched_bernoulli = log_unmatched_bernoulli + log_f.squeeze()

                # Save stats
                log_unmatched_bernoullis.append(-log_f.squeeze().item())
            elif (pred >= num_preds) and (
                gt >= 0
            ):  # if poisson was assigned to a GT element
                num_matched_ppp += 1
                assigned_gt = gt
                gt_c = gt_class[assigned_gt].unsqueeze(0)
                gt_vec = torch.cat([gt_box[assigned_gt, :], gt_c])
                log_f = intensity_func(gt_vec.unsqueeze(0), img_size).squeeze()
                log_poisson = log_poisson + log_f

                # Save stats
                log_matched_ppps.append(-log_f.item())

        association_sum = log_matched_bernoulli + log_unmatched_bernoulli + log_poisson
        out_dict["matched_bernoulli"].append(-log_matched_bernoulli.item())
        out_dict["matched_bernoulli_reg"].append(-log_matched_regression.item())
        out_dict["matched_bernoulli_cls"].append(-log_matched_classification.item())
        out_dict["num_matched_bernoulli"].append(num_matched_bernoulli)
        out_dict["unmatched_bernoulli"].append(-log_unmatched_bernoulli.item())
        out_dict["num_unmatched_bernoulli"].append(num_unmatched_bernoulli)
        out_dict["matched_ppp"].append(-log_poisson.item())
        out_dict["num_matched_ppp"].append(num_matched_ppp)
        out_dict["matched_bernoulli_regs"].append(log_matched_bernoulli_regs)
        out_dict["matched_bernoulli_clss"].append(log_matched_bernoulli_cls)
        out_dict["unmatched_bernoullis"].append(log_unmatched_bernoullis)
        out_dict["matched_ppps"].append(log_matched_ppps)

        nll[a] = association_sum

    nll = torch.logsumexp(nll, -1)

    n_class = torch.tensor(num_classes).unsqueeze(0).to(device)
    ppp_regularizer = intensity_func(None, img_size, n_class, integrate=True).squeeze()
    nll = ppp_regularizer - nll
    out_dict["ppp_integral"] = ppp_regularizer.item()
    out_dict["total"] = [
        out_dict["matched_bernoulli"][i]
        + out_dict["unmatched_bernoulli"][i]
        + out_dict["matched_ppp"][i]
        + out_dict["ppp_integral"]
        for i in range(len(associations))
    ]

    return nll, out_dict


def negative_log_likelihood_matching(
    box_scores: torch.tensor,
    box_regs: torch.tensor,
    box_covars: torch.tensor,
    gt_box: torch.tensor,
    gt_class: torch.tensor,
    image_size: List[int],
    reg_distribution: torch.distributions.distribution.Distribution,
    device: torch.device,
    intensity_func=lambda x: 0.00000001,
    max_n_solutions: int = 5,
    scores_have_bg_cls=False,
    target_delta=None,
    distance_type="log_prob",
    covar_scaling = 1,
    use_target_delta_matching=True,
    pred_delta=None,
    pred_delta_chol=None,
):

    img_size = torch.tensor(image_size).unsqueeze(0).to(device)
    num_preds, num_classes = box_scores.shape
    if scores_have_bg_cls:
        num_classes -= 1  # do not count background class
    num_gt = gt_box.shape[0]
    existance_prob = 1 - box_scores[:, -1]
    # Init potential covar scaling for matching
    covar_scaling = torch.eye(box_covars.shape[-1]).to(box_covars.device)*covar_scaling
    # save indices of inf cost
    infinite_costs = []
    with torch.no_grad():
        if not(num_gt > 0 and num_preds > 0):
            associations = -np.ones((1, num_preds + num_gt, 2))
            if num_gt > 0:
                associations[0, -num_gt:, 1] = np.arange(num_gt)
            associations[0, :, 0] = np.arange(num_preds + num_gt)
            associations = associations.astype(np.int32)
            return associations

        # Assemble and fill cost matrix
        cost_matrix = torch.zeros((num_preds + num_gt, num_gt), dtype=torch.float64)

        if scores_have_bg_cls:
            r = existance_prob.unsqueeze(-1).repeat(1, num_gt)
        else:
            r = box_scores[:, gt_class]  # assume existance prob == class prob

        covar = box_covars[:, gt_class] if pred_delta_chol is None or not use_target_delta_matching else pred_delta_chol[:, gt_class]
        reg_means = box_regs if pred_delta is None or not use_target_delta_matching else pred_delta
        # Repeat gt to be [num_preds,num_gt,dim] if needed
        if len(gt_box.shape) < len(reg_means[:, gt_class].shape):
            gt_box = gt_box.unsqueeze(0).repeat(num_preds, 1, 1)

        if distance_type == "log_prob":
            # Covar is actually cholesky decomposed, hence only one multiplication with scaling
            scaled_covar = covar_scaling@covar
            dist = reg_distribution(reg_means[:, gt_class], scaled_covar)

            if target_delta is None or not use_target_delta_matching:
                log_p = dist.log_prob(gt_box)
            else:
                log_p = dist.log_prob(target_delta)

        elif distance_type == "euclidian_squared":
            # We use minus since its sign is reversed later (and cost should be minimized)
            if target_delta is None or not use_target_delta_matching:
                log_p = -(reg_means[:, gt_class] - gt_box).pow(2).sum(-1)
            else:
                log_p = -(reg_means[:, gt_class] - target_delta).pow(2).sum(-1)

        elif distance_type == "euclidian":
            # We use minus since its sign is reversed later (and cost should be minimized)
            if target_delta is None or not use_target_delta_matching:
                log_p = -(reg_means[:, gt_class] - gt_box).pow(2).sum(-1).sqrt()
            else:
                log_p = (
                    -(reg_means[:, gt_class] - target_delta).pow(2).sum(-1).sqrt()
                )
        else:
            raise NotImplementedError(
                f'Distance type for PMB-NLL matching "{distance_type}" not implemented.'
            )

        log_p = log_p.sum(-1) if len(log_p.shape) > 2 else log_p
        log_p = log_p + torch.log(
            box_scores[:, gt_class]
        )  # box regression + class scores conditioned on existance

        cost = -(log_p - torch.log(1 - r))
        cost_matrix[:num_preds] = cost
        if not torch.isfinite(cost).all():
            for k, l in torch.isfinite(cost).logical_not().nonzero():
                infinite_costs.append((k, l))
                cost_matrix[k, l] = 0

        # Build GT vector with [box, class]
        if target_delta is None or not use_target_delta_matching:
            gt_vec = torch.cat([gt_box[0, :, :], gt_class.unsqueeze(-1)], -1)
        else:
            gt_vec = torch.cat([target_delta[0, :, :], gt_class.unsqueeze(-1)], -1)
        # PPP cost
        cost = -intensity_func(gt_vec, img_size, dist_type=distance_type)
        if torch.isfinite(cost).all():
            cost_matrix[num_preds:] = torch.diag(cost)
        else:
            cost_matrix[num_preds:] = torch.diag(cost)
            for l in torch.isfinite(cost).logical_not().nonzero():
                infinite_costs.append((num_preds + l, l))
                cost_matrix[num_preds + l, l] = 0

        # Fill in "inf"
        if cost_matrix.numel() > 0:
            largest_cost = cost_matrix.max()
        for k in range(num_preds, num_preds + num_gt):  # loop over predictions
            for l in range(num_gt):  # loop over ground truths
                if k != (l + num_preds):
                    cost_matrix[k, l] = largest_cost * 3
        for coord in infinite_costs:
            k, l = coord
            cost_matrix[k, l] = largest_cost * 2

        # Find nsolutions best solutions
        nsolutions = 0
        for i in range(num_gt+1):
            if i > num_preds or nsolutions > max_n_solutions:
                break
            nsolutions += (factorial(num_preds)//factorial(num_preds-i))*comb(num_gt, i)

        nsolutions = min(
            max_n_solutions, nsolutions
        )  # comb gives maximum number unique associations
        try:
            associations, _ = run_murtys(cost_matrix, nsolutions)
        except AssertionError:
            print(
                "[NLLOD] Murtys could not find solution! Using linear sum assignment."
            )
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
            associations = -np.ones((1, num_preds + num_gt, 2))
            associations[0, :, 0] = np.arange(num_preds + num_gt)
            associations[0, row_ind, 1] = col_ind
            associations = associations.astype(np.int32)
        

    return associations


def negative_log_likelihood(
    pred_box_scores: List[torch.tensor],
    pred_box_regs: List[torch.tensor],
    pred_box_covars: List[torch.tensor],
    gt_boxes: List[torch.tensor],
    gt_classes: List[torch.tensor],
    image_sizes: List[List[int]],
    reg_distribution: torch.distributions.distribution.Distribution,
    intensity_func=lambda x: 0.00000001,
    max_n_solutions: int = 5,
    training: bool = True,
    scores_have_bg_cls: bool = True,
    target_deltas: torch.tensor = None,
    matching_distance: str = "log_prob",
    covar_scaling: float = 1.0,
    use_target_delta_matching=False,
    pred_deltas=None,
    pred_delta_chols=None,
):

    """
    Calculate NLL for a PMB prediction.
    """

    assert len(pred_box_scores) == len(pred_box_regs) == len(pred_box_covars)

    device = pred_box_scores[0].device

    nll_total_losses = torch.tensor(
        0, dtype=torch.float64, device=device, requires_grad=training
    )
    bs = len(pred_box_scores)
    total_associations = []
    total_decompositions = []

    for i in range(bs):  # loop over images
        if type(intensity_func) == list:
            if type(intensity_func[i]) != dict:
                ppp = {"matching": intensity_func[i], "loss": intensity_func[i]}
            else:
                ppp = intensity_func[i]
        else:
            if type(intensity_func) != dict:
                ppp = {"matching": intensity_func, "loss": intensity_func}
            else:
                ppp = intensity_func

        # [N, num_classes] or [N, num_classes+1]
        box_scores = pred_box_scores[i]
        num_preds, num_classes = box_scores.shape
        if scores_have_bg_cls:
            num_classes -= 1  # do not count background class
        # [N, num_classes, boxdims]
        box_regs = pred_box_regs[i]
        # [N, num_classes, boxdims, boxdims]
        box_covars = pred_box_covars[i]
        # [M, boxdims]
        gt_box = gt_boxes[i]
        # [M, 1]
        gt_class = gt_classes[i]

        if target_deltas is None:
            target_delta = None
        else:
            # [N, M, boxdims]
            target_delta = target_deltas[i]

        if pred_deltas is None:
            pred_delta = None
        else:
            # [N, M, boxdims]
            pred_delta = pred_deltas[i]
        
        if pred_delta_chols is None:
            pred_delta_chol = None
        else:
            # [N, M, boxdims]
            pred_delta_chol = pred_delta_chols[i]

        image_size = image_sizes[i]

        associations = negative_log_likelihood_matching(
            box_scores,
            box_regs,
            box_covars,
            gt_box,
            gt_class,
            image_size,
            reg_distribution,
            device,
            ppp["matching"],
            max_n_solutions,
            scores_have_bg_cls,
            target_delta,
            matching_distance,
            covar_scaling,
            use_target_delta_matching,
            pred_delta,
            pred_delta_chol,
        )

        nll, decomposition = compute_negative_log_likelihood(
            box_scores=box_scores,
            box_regs=box_regs,
            box_covars=box_covars,
            gt_box=gt_box,
            gt_class=gt_class,
            image_size=image_size,
            reg_distribution=reg_distribution,
            associations=associations,
            device=device,
            intensity_func=ppp["loss"],
            scores_have_bg_cls=scores_have_bg_cls,
            target_delta=target_delta,
            pred_delta=pred_delta,
            pred_delta_chol=pred_delta_chol,
        )

        if torch.isfinite(nll):
            # Normalize by num predictions if training
            if training:
                number_preds = decomposition["num_matched_ppp"][0]+decomposition["num_matched_bernoulli"][0]+decomposition["num_unmatched_bernoulli"][0]
                regularizer = max(1, number_preds)
                nll_total_losses = nll_total_losses + nll / regularizer
            else:
                nll_total_losses = nll_total_losses + nll
        else:
            bs = max(1, bs - 1)
            print("WARNING: Infinite loss in NLL!")
            print(f"box scores: {box_scores}")
            print(f"box_regs: {box_regs}")
            print(f"box_covars: {box_covars}")
            print(f"gt_box: {gt_box}")
            print(f"gt_class: {gt_class}")
            print(f"associations: {associations}")

        total_associations.append(associations)
        total_decompositions.append(decomposition)

    return nll_total_losses / bs, total_associations, total_decompositions
