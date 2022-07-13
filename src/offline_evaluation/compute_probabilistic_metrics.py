import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.distributions as distributions
import tqdm

# Project imports
from core.evaluation_tools import evaluation_utils, scoring_rules
from core.evaluation_tools.evaluation_utils import (
    calculate_iou,
    get_test_thing_dataset_id_to_train_contiguous_id_dict,
)
from core.setup import setup_arg_parser, setup_config
from detectron2.checkpoint import DetectionCheckpointer

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch
from detectron2.modeling import build_model
from matplotlib import image
from matplotlib import pyplot as plt
from matplotlib.pyplot import hist
from prettytable import PrettyTable
from probabilistic_inference.inference_utils import get_inference_output_dir
from probabilistic_modeling.losses import (
    compute_negative_log_likelihood,
    negative_log_likelihood,
)
from probabilistic_modeling.modeling_utils import (
    PoissonPointProcessGMM,
    PoissonPointProcessIntensityFunction,
    PoissonPointProcessUniform,
    PoissonPointUnion,
)
from scipy.spatial.distance import mahalanobis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AREA_LIMITS = {"small": [0, 1024], "medium": [1024, 9216], "large": [9216, np.inf]}


def try_squeeze(to_squeeze, dim):
    return to_squeeze.squeeze(dim) if len(to_squeeze.shape) > dim else to_squeeze


def print_nll_results_by_size(
    out, gt_boxes, inference_output_dir, area_limits=AREA_LIMITS, prefix=""
):
    title_dict = {
        "matched_bernoulli_clss": "Matched Bernoulli Classification",
        "matched_bernoulli_cls": "Matched Bernoulli Classification",
        "matched_bernoulli_reg": "Matched Bernoulli Regression",
        "matched_bernoulli_regs": "Matched Bernoulli Regression",
        "matched_bernoulli": "Matched Bernoulli",
        "matched_bernoullis": "Matched Bernoulli",
        "matched_ppp": "Matched PPP",
        "matched_ppps": "Matched PPP",
    }

    def plot_histogram(
        size_decomp, decomp_key, area_limits, filepath, max_limit=40, nbins=100
    ):
        plt.clf()
        for size in size_decomp.keys():
            hist(
                np.clip(size_decomp[size][decomp_key], 0, max_limit),
                nbins,
                alpha=0.33,
                label=size,
                ec=(0, 0, 0, 0),
                lw=0.0,
            )

        plt.title(title_dict[decomp_key])
        plt.legend()
        plt.xlim(0, max_limit)
        plt.savefig(
            os.path.join(filepath, f"{prefix}{decomp_key}.svg"),
            format="svg",
            transparent=True,
        )

    size_decomp = {size: defaultdict(list) for size in area_limits.keys()}
    for img_id, out_dict in out.items():
        boxes = gt_boxes[img_id].reshape(-1, 4)
        decomp = out_dict["decomposition"]
        # Remove unmatched detections and sort in gt-order instead
        association = np.array(out_dict["associations"][0])
        if not len(association):
            continue
        association = association[association[:, 1] > -1]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        num_gts = len(areas)
        num_preds = (
            decomp["num_unmatched_bernoulli"][0] + decomp["num_matched_bernoulli"][0]
        )
        ppp_association = association[association[:, 0] >= num_preds]

        for size, limit in area_limits.items():
            mask = torch.logical_and(limit[0] < areas, limit[1] > areas)
            gt_idx = mask.nonzero()
            matched_bernoulli_regs = [
                comp
                for assoc, comp in zip(association, decomp["matched_bernoulli_regs"][0])
                if assoc[1] in gt_idx
            ]
            size_decomp[size]["matched_bernoulli_regs"] += matched_bernoulli_regs
            size_decomp[size]["matched_bernoulli_reg"] += [sum(matched_bernoulli_regs)]

            matched_bernoulli_clss = [
                comp
                for assoc, comp in zip(association, decomp["matched_bernoulli_clss"][0])
                if assoc[1] in gt_idx
            ]
            size_decomp[size]["matched_bernoulli_clss"] += matched_bernoulli_clss
            size_decomp[size]["matched_bernoulli_cls"] += [sum(matched_bernoulli_clss)]

            size_decomp[size]["matched_bernoullis"] += [
                cls_part + reg_part
                for cls_part, reg_part in zip(
                    matched_bernoulli_clss, matched_bernoulli_regs
                )
            ]
            size_decomp[size]["matched_bernoulli"] += [
                sum(matched_bernoulli_regs) + sum(matched_bernoulli_clss)
            ]

            matched_ppps = [
                comp
                for assoc, comp in zip(ppp_association, decomp["matched_ppps"][0])
                if assoc[1] in gt_idx
            ]
            size_decomp[size]["matched_ppps"] += matched_ppps
            size_decomp[size]["matched_ppp"] += [sum(matched_ppps)]

    for size, limit in area_limits.items():
        print(f"******** Size: {size} ********")
        print(
            f"Mean matched Bernoulli: {np.mean(size_decomp[size]['matched_bernoulli']):.2f}/",
            end="",
        )
        print(f"{np.mean(size_decomp[size]['matched_bernoullis']):.2f}")

        print(
            f"Mean matched Bernoulli reg: {np.mean(size_decomp[size]['matched_bernoulli_reg']):.2f}/",
            end="",
        )
        print(f"{np.mean(size_decomp[size]['matched_bernoulli_regs']):.2f}")

        print(
            f"Mean matched Bernoulli cls: {np.mean(size_decomp[size]['matched_bernoulli_cls']):.2f}/",
            end="",
        )
        print(f"{np.mean(size_decomp[size]['matched_bernoulli_clss']):.2f}")

        print(
            f"Mean matched PPP: {np.mean(size_decomp[size]['matched_ppp']):.2f}/",
            end="",
        )
        print(f"{np.mean(size_decomp[size]['matched_ppps']):.2f}")

        print(f"**************************")

    for decomp_key in size_decomp[list(area_limits.keys())[0]]:
        plot_histogram(size_decomp, decomp_key, area_limits, inference_output_dir)


def print_nll_results(out):
    nlls = torch.tensor([el["nll"] for el in out.values() if el["nll"] > 0])
    print("*" * 40)
    print("*" * 12 + "PMB NLL results" + "*" * 13)
    print("*" * 40)
    print(f"Min NLL: {nlls.min().item()}")
    print(f"Mean NLL: {nlls.mean().item()}")
    print(f"Median NLL: {nlls.median().item()}")
    print(f"Max NLL: {nlls.max().item()}")
    print(f"Binned NLL: {torch.histc(nlls, bins=20).tolist()}")
    print("*" * 40)
    matched_bernoulli = []
    matched_bernoulli_reg = []
    matched_bernoulli_cls = []
    num_matched_bernoulli = []
    unmatched_bernoulli = []
    num_unmatched_bernoulli = []
    matched_ppp = []
    num_matched_ppp = []
    ppp_integral = []
    for img_id, out_dict in out.items():
        decomp = out_dict["decomposition"]
        matched_bernoulli.append(decomp["matched_bernoulli"][0])
        matched_bernoulli_reg.append(decomp["matched_bernoulli_reg"][0])
        matched_bernoulli_cls.append(decomp["matched_bernoulli_cls"][0])
        num_matched_bernoulli.append(decomp["num_matched_bernoulli"][0])
        unmatched_bernoulli.append(decomp["unmatched_bernoulli"][0])
        num_unmatched_bernoulli.append(decomp["num_unmatched_bernoulli"][0])
        matched_ppp.append(decomp["matched_ppp"][0])
        num_matched_ppp.append(decomp["num_matched_ppp"][0])
        ppp_integral.append(decomp["ppp_integral"])
    matched_bernoulli = np.array(matched_bernoulli)
    matched_bernoulli_reg = np.array(matched_bernoulli_reg)
    matched_bernoulli_cls = np.array(matched_bernoulli_cls)
    num_matched_bernoulli = np.array(num_matched_bernoulli)
    unmatched_bernoulli = np.array(unmatched_bernoulli)
    num_unmatched_bernoulli = np.array(num_unmatched_bernoulli)
    matched_ppp = np.array(matched_ppp)
    num_matched_ppp = np.array(num_matched_ppp)
    num_matched_ppp = num_matched_ppp[matched_ppp < np.inf]
    matched_ppp = matched_ppp[matched_ppp < np.inf]
    matched_bernoulli_norm = matched_bernoulli.sum() / (num_matched_bernoulli.sum())
    matched_bernoulli_reg_norm = matched_bernoulli_reg.sum() / (
        num_matched_bernoulli.sum()
    )
    matched_bernoulli_cls_norm = matched_bernoulli_cls.sum() / (
        num_matched_bernoulli.sum()
    )
    print(f"Mean matched Bernoulli: {np.mean(matched_bernoulli):.2f}/", end="")
    print(f"{matched_bernoulli_norm:.2f}")
    print(f"Mean matched Bernoulli reg: {np.mean(matched_bernoulli_reg):.2f}/", end="")
    print(f"{matched_bernoulli_reg_norm:.2f}")
    print(f"Mean matched Bernoulli cls: {np.mean(matched_bernoulli_cls):.2f}/", end="")
    print(f"{matched_bernoulli_cls_norm:.2f}")

    unmatched_bernoulli_norm = unmatched_bernoulli.sum() / (
        num_unmatched_bernoulli.sum()
    )
    print(f"Mean unmatched Bernoulli: {np.mean(unmatched_bernoulli):.2f}/", end="")
    print(f"{unmatched_bernoulli_norm:.2f}")

    matched_ppp_norm = matched_ppp.sum() / num_matched_ppp.sum()
    print(f"Mean matched PPP: {np.mean(matched_ppp):.2f}/", end="")
    print(f"{matched_ppp_norm:.2f}")
    print(f"Mean PPP integral: {np.mean(ppp_integral):.2f}")
    print("*" * 40)


def plot_nll_results(out, inference_output_dir, prefix=""):
    matched_bernoulli = []
    matched_bernoulli_reg = []
    matched_bernoulli_cls = []
    num_matched_bernoulli = []
    unmatched_bernoulli = []
    num_unmatched_bernoulli = []
    matched_ppp = []
    num_matched_ppp = []
    ppp_integral = []
    for img_id, out_dict in out.items():
        decomp = out_dict["decomposition"]
        matched_bernoulli += [
            reg + classification
            for reg, classification in zip(
                decomp["matched_bernoulli_regs"][0],
                decomp["matched_bernoulli_clss"][0],
            )
        ]
        matched_bernoulli_reg += decomp["matched_bernoulli_regs"][0]
        matched_bernoulli_cls += decomp["matched_bernoulli_clss"][0]
        num_matched_bernoulli.append(decomp["num_matched_bernoulli"][0])

        unmatched_bernoulli += decomp["unmatched_bernoullis"][0]
        num_unmatched_bernoulli.append(decomp["num_unmatched_bernoulli"][0])

        matched_ppp += decomp["matched_ppps"][0]
        num_matched_ppp.append(decomp["num_matched_ppp"][0])
        ppp_integral.append(decomp["ppp_integral"])

    plt.figure()
    plt.hist(np.clip(matched_bernoulli, 0, 40), 100, ec=(0, 0, 0, 0), lw=0.0)
    plt.xlim(0, 40)
    plt.title("Matched Bernoulli")
    plt.savefig(
        os.path.join(inference_output_dir, f"{prefix}matched_bernoulli_histogram.svg"),
        format="svg",
        transparent=True,
    )

    plt.clf()
    plt.hist(np.clip(matched_bernoulli_reg, 0, 40), 100, ec=(0, 0, 0, 0), lw=0.0)
    plt.xlim(0, 40)
    plt.title("Matched Bernoulli regression")
    plt.savefig(
        os.path.join(
            inference_output_dir, f"{prefix}matched_bernoulli_reg_histogram.svg"
        ),
        format="svg",
        transparent=True,
    )

    plt.clf()
    plt.hist(np.clip(matched_bernoulli_cls, 0, 5), 100, ec=(0, 0, 0, 0), lw=0.0)
    plt.xlim(0, 5)
    plt.title("Matched Bernoulli Classification")
    plt.savefig(
        os.path.join(
            inference_output_dir, f"{prefix}matched_bernoulli_cls_histogram.svg"
        ),
        format="svg",
        transparent=True,
    )

    plt.clf()
    plt.hist(np.clip(unmatched_bernoulli, 0, 10), 100, ec=(0, 0, 0, 0), lw=0.0)
    plt.xlim(0, 10)
    plt.title("Unmatched Bernoulli")
    plt.savefig(
        os.path.join(
            inference_output_dir, f"{prefix}unmatched_bernoulli_histogram.svg"
        ),
        format="svg",
        transparent=True,
    )

    plt.clf()
    plt.hist(np.clip(matched_ppp, 0, 40), 100, ec=(0, 0, 0, 0), lw=0.0)
    plt.xlim(0, 40)
    plt.title("Matched PPP")
    plt.savefig(
        os.path.join(inference_output_dir, f"{prefix}matched_ppp_histogram.svg"),
        format="svg",
        transparent=True,
    )


def compute_pmb_nll(
    cfg,
    inference_output_dir,
    cat_mapping_dict,
    min_allowed_score=0.0,
    print_results=True,
    plot_results=True,
    print_by_size=True,
    load_nll_results=True,
):
    results_file = os.path.join(
        inference_output_dir, f"nll_results_minallowedscore_{min_allowed_score}.pkl"
    )
    if load_nll_results and os.path.isfile(results_file):
        with open(results_file, "rb") as f:
            out = pickle.load(f)

        if print_results:
            print_nll_results(out)

        if plot_results:
            plot_nll_results(out, inference_output_dir)

        if print_by_size:
            (
                preprocessed_predicted_instances,
                preprocessed_gt_instances,
            ) = evaluation_utils.get_per_frame_preprocessed_instances(
                cfg, inference_output_dir, min_allowed_score
            )
            gt_boxes = preprocessed_gt_instances["gt_boxes"]
            print_nll_results_by_size(out, gt_boxes, inference_output_dir)

        return out

    with torch.no_grad():
        # Load predictions and GT
        (
            preprocessed_predicted_instances,
            preprocessed_gt_instances,
        ) = evaluation_utils.get_per_frame_preprocessed_instances(
            cfg, inference_output_dir, min_allowed_score
        )
        predicted_box_means = preprocessed_predicted_instances["predicted_boxes"]
        predicted_cls_probs = preprocessed_predicted_instances["predicted_cls_probs"]
        predicted_box_covariances = preprocessed_predicted_instances[
            "predicted_covar_mats"
        ]

        if "ppp_weights" in preprocessed_predicted_instances:
            predicted_ppp = preprocessed_predicted_instances["ppp_weights"]
        elif "log_ppp_intensity" in preprocessed_predicted_instances:
            predicted_ppp = preprocessed_predicted_instances["log_ppp_intensity"]
        else:
            predicted_ppp = defaultdict(list)

        if cfg.PROBABILISTIC_INFERENCE.LOAD_PPP_FROM_MODEL:
            model = build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=True
            )
            ppp = model.get_ppp_intensity_function()
            ppp.set_normalization_of_bboxes(True)
            ppp.update_distribution()
            predicted_ppp = defaultdict(int)

        image_sizes = preprocessed_predicted_instances["image_size"]
        gt_box_means = preprocessed_gt_instances["gt_boxes"]
        gt_cat_idxs = preprocessed_gt_instances["gt_cat_idxs"]

        # Initialize results
        out = defaultdict(dict)
        print("[NLLOD] Started evaluating NLL for dataset.")
        with tqdm.tqdm(total=len(predicted_box_means)) as pbar:
            for image_id in predicted_box_means:
                ppp_mix = PoissonPointUnion()
                pbar.update(1)
                image_size = image_sizes[image_id]
                ################ GT STUFF ###########################
                gt_boxes = gt_box_means[image_id]
                if len(gt_boxes.shape) < 2:
                    gt_boxes = gt_boxes.view(-1, 4)
                gt_classes = (
                    torch.as_tensor(
                        [
                            cat_mapping_dict[cat_id.item()]
                            for cat_id in gt_cat_idxs[image_id].long().view(-1, 1)
                        ]
                    )
                    .long()
                    .to(device)
                )

                ################# PREDICTION STUFF ####################
                pred_cls_probs = predicted_cls_probs[image_id].clamp(1e-6, 1 - 1e-6)
                if cfg.MODEL.META_ARCHITECTURE == "ProbabilisticRetinaNet":
                    num_classes = pred_cls_probs.shape[-1]
                    scores_have_bg_cls = False
                else:
                    num_classes = pred_cls_probs.shape[-1] - 1
                    scores_have_bg_cls = True

                pred_box_means = (
                    predicted_box_means[image_id].unsqueeze(1).repeat(1, num_classes, 1)
                )

                pred_box_covs = predicted_box_covariances[image_id]
                pred_box_covs = pred_box_covs.unsqueeze(1).repeat(1, num_classes, 1, 1)
                pred_ppp_weights = predicted_ppp[image_id]
                if not cfg.PROBABILISTIC_INFERENCE.TREAT_AS_MB:
                    if cfg.PROBABILISTIC_INFERENCE.PPP_CONFIDENCE_THRES > 0:
                        if scores_have_bg_cls:
                            max_conf = 1 - pred_cls_probs[..., -1]
                        else:
                            max_conf = pred_cls_probs[..., :num_classes].max(dim=1)[0]
                        ppp_preds_idx = (
                            max_conf <= cfg.PROBABILISTIC_INFERENCE.PPP_CONFIDENCE_THRES
                        )
                        if not ppp_preds_idx.any():
                            ppp_preds = PoissonPointProcessIntensityFunction(
                                cfg, log_intensity=-np.inf, device=gt_boxes.device
                            )
                        else:
                            mixture_dict = {}
                            mixture_dict["weights"] = max_conf[ppp_preds_idx]
                            mixture_dict["means"] = pred_box_means[ppp_preds_idx, 0]
                            mixture_dict["covs"] = pred_box_covs[ppp_preds_idx, 0]
                            mixture_dict["cls_probs"] = pred_cls_probs[
                                ppp_preds_idx, :num_classes
                            ]
                            mixture_dict[
                                "reg_dist_type"
                            ] = (
                                cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.DISTRIBUTION_TYPE
                            )
                            if (
                                cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.DISTRIBUTION_TYPE
                                == "gaussian"
                            ):
                                mixture_dict[
                                    "reg_dist"
                                ] = distributions.multivariate_normal.MultivariateNormal
                                mixture_dict["reg_kwargs"] = {
                                    "covariance_matrix": mixture_dict["covs"]
                                }
                            elif (
                                cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.DISTRIBUTION_TYPE
                                == "laplacian"
                            ):
                                mixture_dict["reg_dist"] = distributions.laplace.Laplace
                                mixture_dict["reg_kwargs"] = {
                                    "scale": torch.sqrt(
                                        mixture_dict["covs"].diagonal(dim1=-2, dim2=-1)
                                        / 2
                                    )
                                }

                            ppp_preds = PoissonPointProcessIntensityFunction(
                                cfg, predictions=mixture_dict
                            )

                            pred_box_means = pred_box_means[ppp_preds_idx.logical_not()]
                            pred_box_covs = pred_box_covs[ppp_preds_idx.logical_not()]
                            pred_cls_probs = pred_cls_probs[ppp_preds_idx.logical_not()]

                        ppp_mix.add_ppp(ppp_preds)

                    if cfg.PROBABILISTIC_INFERENCE.LOAD_PPP_FROM_MODEL:
                        ppp = ppp
                    elif isinstance(pred_ppp_weights, dict):
                        ppp = PoissonPointProcessIntensityFunction(
                            cfg, device=gt_boxes.device
                        )
                        ppp.load_weights(pred_ppp_weights)
                    elif isinstance(pred_ppp_weights, torch.Tensor):
                        ppp = PoissonPointProcessIntensityFunction(
                            cfg, log_intensity=pred_ppp_weights, device=gt_boxes.device
                        )
                    else:
                        print(
                            "[NLLOD] PPP intensity function not found in annotations, using config"
                        )
                        pred_ppp_weights = -np.inf
                        ppp = PoissonPointProcessIntensityFunction(
                            cfg, log_intensity=pred_ppp_weights, device=gt_boxes.device
                        )
                else:
                    pred_ppp_weights = -np.inf
                    ppp = PoissonPointProcessIntensityFunction(
                        cfg, log_intensity=pred_ppp_weights
                    )
                ppp_mix.add_ppp(ppp)

                if (
                    cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.DISTRIBUTION_TYPE
                    == "gaussian"
                ):
                    reg_distribution = lambda x, y: distributions.multivariate_normal.MultivariateNormal(
                        x, y
                    )
                elif (
                    cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.DISTRIBUTION_TYPE
                    == "laplacian"
                ):
                    reg_distribution = lambda x, y: distributions.laplace.Laplace(
                        loc=x, scale=torch.sqrt(y.diagonal(dim1=-2, dim2=-1) / 2)
                    )
                else:
                    raise Exception(
                        f"Bounding box uncertainty distribution {cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.DISTRIBUTION_TYPE} is not available."
                    )

                try:
                    nll, associations, decompositions = negative_log_likelihood(
                        pred_box_scores=[pred_cls_probs],
                        pred_box_regs=[pred_box_means],
                        pred_box_covars=[pred_box_covs],
                        gt_boxes=[gt_boxes],
                        gt_classes=[gt_classes],
                        image_sizes=[image_size],
                        reg_distribution=reg_distribution,
                        intensity_func=ppp_mix,
                        max_n_solutions=cfg.MODEL.PROBABILISTIC_MODELING.NLL_MAX_NUM_SOLUTIONS,
                        training=False,
                        scores_have_bg_cls=scores_have_bg_cls,
                    )
                    out[image_id] = {
                        "nll": nll.item(),
                        "associations": associations[0].tolist(),
                        "decomposition": decompositions[0],
                    }
                except Exception as e:
                    print(
                        f"Image {image_id} raised error. Will not be used to calculate NLL."
                    )
                    print(e)
    with open(
        os.path.join(
            inference_output_dir,
            f"nll_results_minallowedscore_{min_allowed_score}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(out, f)

    if print_results:
        print_nll_results(out)

    if plot_results:
        plot_nll_results(out, inference_output_dir)

    if print_by_size:
        gt_boxes = preprocessed_gt_instances["gt_boxes"]
        print_nll_results_by_size(out, gt_boxes, inference_output_dir)

    return out


def main(
    args,
    cfg=None,
    iou_min=None,
    iou_correct=None,
    min_allowed_score=None,
    print_results=True,
    inference_output_dir="",
    image_ids=[],
):

    # Setup config
    if cfg is None:
        cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

    cfg.defrost()
    cfg.ACTUAL_TEST_DATASET = args.test_dataset

    # Setup torch device and num_threads
    torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    # Build path to gt instances and inference output
    if inference_output_dir == "":
        inference_output_dir = get_inference_output_dir(
            cfg["OUTPUT_DIR"],
            args.test_dataset,
            args.inference_config,
            args.image_corruption_level,
        )

    # Get thresholds to perform evaluation on
    if iou_min is None:
        iou_min = args.iou_min
    if iou_correct is None:
        iou_correct = args.iou_correct

    if min_allowed_score is None or min_allowed_score < 0:
        # Check if F-1 Score has been previously computed ON THE ORIGINAL
        # DATASET such as COCO even when evaluating on OpenImages.
        try:
            with open(os.path.join(inference_output_dir, "mAP_res.txt"), "r") as f:
                min_allowed_score = f.read().strip("][\n").split(", ")[-1]
                min_allowed_score = round(float(min_allowed_score), 4)
        except FileNotFoundError:
            # If not, process all detections. Not recommended as the results might be influenced by very low scoring
            # detections that would normally be removed in robotics/vision
            # applications.
            min_allowed_score = 0.0
    # Get category mapping dictionary:
    train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        cfg.DATASETS.TRAIN[0]
    ).thing_dataset_id_to_contiguous_id
    test_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        args.test_dataset
    ).thing_dataset_id_to_contiguous_id

    cat_mapping_dict = get_test_thing_dataset_id_to_train_contiguous_id_dict(
        cfg,
        args,
        train_thing_dataset_id_to_contiguous_id,
        test_thing_dataset_id_to_contiguous_id,
    )

    # Compute NLL results
    load_nll_results = len(image_ids) == 0
    nll_results = compute_pmb_nll(
        cfg, inference_output_dir, cat_mapping_dict, min_allowed_score, print_results, load_nll_results=load_nll_results
    )

    # Get matched results by either generating them or loading from file.
    with torch.no_grad():
        matched_results = evaluation_utils.get_matched_results(
            cfg,
            inference_output_dir,
            iou_min=iou_min,
            iou_correct=iou_correct,
            min_allowed_score=min_allowed_score,
        )

        # Build preliminary dicts required for computing classification scores.
        for matched_results_key in matched_results.keys():
            if "gt_cat_idxs" in matched_results[matched_results_key].keys():
                # First we convert the written things indices to contiguous
                # indices.
                gt_converted_cat_idxs = matched_results[matched_results_key][
                    "gt_cat_idxs"
                ]
                gt_converted_cat_idxs = try_squeeze(gt_converted_cat_idxs, 1)
                gt_converted_cat_idxs = torch.as_tensor(
                    [
                        cat_mapping_dict[class_idx.cpu().tolist()]
                        for class_idx in gt_converted_cat_idxs
                    ]
                ).to(device)
                matched_results[matched_results_key][
                    "gt_converted_cat_idxs"
                ] = gt_converted_cat_idxs.to(device)
                if "predicted_cls_probs" in matched_results[matched_results_key].keys():
                    predicted_cls_probs = matched_results[matched_results_key][
                        "predicted_cls_probs"
                    ]
                    # This is required for evaluation of retinanet based
                    # detections.
                    matched_results[matched_results_key][
                        "predicted_score_of_gt_category"
                    ] = torch.gather(
                        predicted_cls_probs, 1, gt_converted_cat_idxs.unsqueeze(1)
                    ).squeeze(
                        1
                    )
                matched_results[matched_results_key][
                    "gt_cat_idxs"
                ] = gt_converted_cat_idxs
            else:
                if cfg.MODEL.META_ARCHITECTURE == "ProbabilisticRetinaNet":
                    # For false positives, the correct category is background. For retinanet, since no explicit
                    # background category is available, this value is computed as 1.0 - score of the predicted
                    # category.
                    predicted_class_probs, predicted_class_idx = matched_results[
                        matched_results_key
                    ]["predicted_cls_probs"].max(1)
                    matched_results[matched_results_key][
                        "predicted_score_of_gt_category"
                    ] = (1.0 - predicted_class_probs)
                    matched_results[matched_results_key][
                        "predicted_cat_idxs"
                    ] = predicted_class_idx
                else:
                    # For RCNN/DETR based networks, a background category is
                    # explicitly available.
                    matched_results[matched_results_key][
                        "predicted_score_of_gt_category"
                    ] = matched_results[matched_results_key]["predicted_cls_probs"][
                        :, -1
                    ]
                    _, predicted_class_idx = matched_results[matched_results_key][
                        "predicted_cls_probs"
                    ][:, :-1].max(1)
                    matched_results[matched_results_key][
                        "predicted_cat_idxs"
                    ] = predicted_class_idx

        # Load the different detection partitions
        true_positives = matched_results["true_positives"]
        duplicates = matched_results["duplicates"]
        localization_errors = matched_results["localization_errors"]
        false_negatives = matched_results["false_negatives"]
        false_positives = matched_results["false_positives"]

        # Get the number of elements in each partition
        num_true_positives = true_positives["predicted_box_means"].shape[0]
        num_duplicates = duplicates["predicted_box_means"].shape[0]
        num_localization_errors = localization_errors["predicted_box_means"].shape[0]
        num_false_negatives = false_negatives["gt_box_means"].shape[0]
        num_false_positives = false_positives["predicted_box_means"].shape[0]

        per_class_output_list = []
        for class_idx in cat_mapping_dict.values():
            true_positives_valid_idxs = (
                true_positives["gt_converted_cat_idxs"] == class_idx
            )
            localization_errors_valid_idxs = (
                localization_errors["gt_converted_cat_idxs"] == class_idx
            )
            duplicates_valid_idxs = duplicates["gt_converted_cat_idxs"] == class_idx
            false_positives_valid_idxs = (
                false_positives["predicted_cat_idxs"] == class_idx
            )

            if cfg.MODEL.META_ARCHITECTURE == "ProbabilisticRetinaNet":
                # Compute classification metrics for every partition
                true_positives_cls_analysis = scoring_rules.sigmoid_compute_cls_scores(
                    true_positives, true_positives_valid_idxs
                )
                localization_errors_cls_analysis = (
                    scoring_rules.sigmoid_compute_cls_scores(
                        localization_errors, localization_errors_valid_idxs
                    )
                )
                duplicates_cls_analysis = scoring_rules.sigmoid_compute_cls_scores(
                    duplicates, duplicates_valid_idxs
                )
                false_positives_cls_analysis = scoring_rules.sigmoid_compute_cls_scores(
                    false_positives, false_positives_valid_idxs
                )

            else:
                # Compute classification metrics for every partition
                true_positives_cls_analysis = scoring_rules.softmax_compute_cls_scores(
                    true_positives, true_positives_valid_idxs
                )
                localization_errors_cls_analysis = (
                    scoring_rules.softmax_compute_cls_scores(
                        localization_errors, localization_errors_valid_idxs
                    )
                )
                duplicates_cls_analysis = scoring_rules.softmax_compute_cls_scores(
                    duplicates, duplicates_valid_idxs
                )
                false_positives_cls_analysis = scoring_rules.softmax_compute_cls_scores(
                    false_positives, false_positives_valid_idxs
                )

            # Compute regression metrics for every partition
            true_positives_reg_analysis = scoring_rules.compute_reg_scores(
                true_positives, true_positives_valid_idxs
            )
            localization_errors_reg_analysis = scoring_rules.compute_reg_scores(
                localization_errors, localization_errors_valid_idxs
            )
            duplicates_reg_analysis = scoring_rules.compute_reg_scores(
                duplicates, duplicates_valid_idxs
            )
            false_positives_reg_analysis = scoring_rules.compute_reg_scores_fn(
                false_positives, false_positives_valid_idxs
            )

            per_class_output_list.append(
                {
                    "true_positives_cls_analysis": true_positives_cls_analysis,
                    "true_positives_reg_analysis": true_positives_reg_analysis,
                    "localization_errors_cls_analysis": localization_errors_cls_analysis,
                    "localization_errors_reg_analysis": localization_errors_reg_analysis,
                    "duplicates_cls_analysis": duplicates_cls_analysis,
                    "duplicates_reg_analysis": duplicates_reg_analysis,
                    "false_positives_cls_analysis": false_positives_cls_analysis,
                    "false_positives_reg_analysis": false_positives_reg_analysis,
                }
            )

        final_accumulated_output_dict = dict()
        final_average_output_dict = dict()

        for key in per_class_output_list[0].keys():
            average_output_dict = dict()
            for inner_key in per_class_output_list[0][key].keys():
                collected_values = [
                    per_class_output[key][inner_key]
                    if per_class_output[key][inner_key] is not None
                    else np.NaN
                    for per_class_output in per_class_output_list
                ]
                collected_values = np.array(collected_values)

                if key in average_output_dict.keys():
                    # Use nan mean since some classes do not have duplicates for
                    # instance or has one duplicate for instance. torch.std returns nan in that case
                    # so we handle those here. This should not have any effect on the final results, as
                    # it only affects inter-class variance which we do not
                    # report anyways.
                    average_output_dict[key].update(
                        {
                            inner_key: np.nanmean(collected_values),
                            inner_key + "_std": np.nanstd(collected_values, ddof=1),
                        }
                    )
                    final_accumulated_output_dict[key].update(
                        {inner_key: collected_values}
                    )
                else:
                    average_output_dict.update(
                        {
                            key: {
                                inner_key: np.nanmean(collected_values),
                                inner_key + "_std": np.nanstd(collected_values, ddof=1),
                            }
                        }
                    )
                    final_accumulated_output_dict.update(
                        {key: {inner_key: collected_values}}
                    )
            final_average_output_dict.update(average_output_dict)

        final_accumulated_output_dict.update(
            {
                "num_instances": {
                    "num_true_positives": num_true_positives,
                    "num_duplicates": num_duplicates,
                    "num_localization_errors": num_localization_errors,
                    "num_false_positives": num_false_positives,
                    "num_false_negatives": num_false_negatives,
                }
            }
        )

        if print_results:
            # Summarize and print all
            table = PrettyTable()
            table.field_names = [
                "Output Type",
                "Number of Instances",
                "Cls Negative Log Likelihood",
                "Cls Brier Score",
                "Reg TP Negative Log Likelihood / FP Entropy",
                "Reg Energy Score",
            ]
            table.add_row(
                [
                    "True Positives:",
                    num_true_positives,
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["true_positives_cls_analysis"][
                            "ignorance_score_mean"
                        ],
                        final_average_output_dict["true_positives_cls_analysis"][
                            "ignorance_score_mean_std"
                        ],
                    ),
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["true_positives_cls_analysis"][
                            "brier_score_mean"
                        ],
                        final_average_output_dict["true_positives_cls_analysis"][
                            "brier_score_mean_std"
                        ],
                    ),
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["true_positives_reg_analysis"][
                            "ignorance_score_mean"
                        ],
                        final_average_output_dict["true_positives_reg_analysis"][
                            "ignorance_score_mean_std"
                        ],
                    ),
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["true_positives_reg_analysis"][
                            "energy_score_mean"
                        ],
                        final_average_output_dict["true_positives_reg_analysis"][
                            "energy_score_mean_std"
                        ],
                    ),
                ]
            )
            table.add_row(
                [
                    "Duplicates:",
                    num_duplicates,
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["duplicates_cls_analysis"][
                            "ignorance_score_mean"
                        ],
                        final_average_output_dict["duplicates_cls_analysis"][
                            "ignorance_score_mean_std"
                        ],
                    ),
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["duplicates_cls_analysis"][
                            "brier_score_mean"
                        ],
                        final_average_output_dict["duplicates_cls_analysis"][
                            "brier_score_mean_std"
                        ],
                    ),
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["duplicates_reg_analysis"][
                            "ignorance_score_mean"
                        ],
                        final_average_output_dict["duplicates_reg_analysis"][
                            "ignorance_score_mean_std"
                        ],
                    ),
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["duplicates_reg_analysis"][
                            "energy_score_mean"
                        ],
                        final_average_output_dict["duplicates_reg_analysis"][
                            "energy_score_mean_std"
                        ],
                    ),
                ]
            )
            table.add_row(
                [
                    "Localization Errors:",
                    num_localization_errors,
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["localization_errors_cls_analysis"][
                            "ignorance_score_mean"
                        ],
                        final_average_output_dict["localization_errors_cls_analysis"][
                            "ignorance_score_mean_std"
                        ],
                    ),
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["localization_errors_cls_analysis"][
                            "brier_score_mean"
                        ],
                        final_average_output_dict["localization_errors_cls_analysis"][
                            "brier_score_mean_std"
                        ],
                    ),
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["localization_errors_reg_analysis"][
                            "ignorance_score_mean"
                        ],
                        final_average_output_dict["localization_errors_reg_analysis"][
                            "ignorance_score_mean_std"
                        ],
                    ),
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["localization_errors_reg_analysis"][
                            "energy_score_mean"
                        ],
                        final_average_output_dict["localization_errors_reg_analysis"][
                            "energy_score_mean_std"
                        ],
                    ),
                ]
            )
            table.add_row(
                [
                    "False Positives:",
                    num_false_positives,
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["false_positives_cls_analysis"][
                            "ignorance_score_mean"
                        ],
                        final_average_output_dict["false_positives_cls_analysis"][
                            "ignorance_score_mean_std"
                        ],
                    ),
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["false_positives_cls_analysis"][
                            "brier_score_mean"
                        ],
                        final_average_output_dict["false_positives_cls_analysis"][
                            "brier_score_mean_std"
                        ],
                    ),
                    "{:.4f} ± {:.4f}".format(
                        final_average_output_dict["false_positives_reg_analysis"][
                            "total_entropy_mean"
                        ],
                        final_average_output_dict["false_positives_reg_analysis"][
                            "total_entropy_mean_std"
                        ],
                    ),
                    "-",
                ]
            )

            table.add_row(["False Negatives:", num_false_negatives, "-", "-", "-", "-"])
            print(table)

            text_file_name = os.path.join(
                inference_output_dir,
                "probabilistic_scoring_res_{}_{}_{}.txt".format(
                    iou_min, iou_correct, min_allowed_score
                ),
            )

            with open(text_file_name, "w") as text_file:
                print(table, file=text_file)

        dictionary_file_name = os.path.join(
            inference_output_dir,
            "probabilistic_scoring_res_{}_{}_{}.pkl".format(
                iou_min, iou_correct, min_allowed_score
            ),
        )

        with open(dictionary_file_name, "wb") as pickle_file:
            pickle.dump(final_accumulated_output_dict, pickle_file)


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()

    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
