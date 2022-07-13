import logging
import math
from typing import List, Tuple

import numpy as np
import torch
from core.visualization_tools.probabilistic_visualizer import ProbabilisticVisualizer
from detectron2.data.detection_utils import convert_image_to_rgb

# Detectron Imports
from detectron2.layers import ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import (
    RetinaNet,
    RetinaNetHead,
    permute_to_N_HWA_K,
)
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from matplotlib import cm
from probabilistic_inference import inference_utils
from torch import Tensor, distributions, nn

from probabilistic_modeling.losses import (
    negative_log_likelihood,
    negative_log_likelihood_matching,
)

# Project Imports
from probabilistic_modeling.modeling_utils import (
    PoissonPointProcessIntensityFunction,
    clamp_log_variance,
    covariance_output_to_cholesky,
    get_probabilistic_loss_weight,
    unscented_transform,
    PoissonPointUnion,
)


@META_ARCH_REGISTRY.register()
class ProbabilisticRetinaNet(RetinaNet):
    """
    Probabilistic retinanet class.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # Parse configs
        self.cls_var_loss = cfg.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NAME
        self.compute_cls_var = self.cls_var_loss != "none"
        self.cls_var_num_samples = (
            cfg.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NUM_SAMPLES
        )

        self.bbox_cov_loss = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NAME
        self.compute_bbox_cov = self.bbox_cov_loss != "none"
        self.bbox_cov_num_samples = (
            cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NUM_SAMPLES
        )
        self.bbox_cov_dist_type = (
            cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.DISTRIBUTION_TYPE
        )
        self.bbox_cov_type = (
            cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.COVARIANCE_TYPE
        )
        if self.bbox_cov_type == "diagonal":
            # Diagonal covariance matrix has N elements
            self.bbox_cov_dims = 4
        else:
            # Number of elements required to describe an NxN covariance matrix is
            # computed as:  (N * (N + 1)) / 2
            self.bbox_cov_dims = 10

        if self.bbox_cov_loss == "pmb_negative_log_likelihood":
            self.ppp_constructor = lambda x: PoissonPointProcessIntensityFunction(
                cfg, **x
            )
            self.ppp_intensity_function = PoissonPointProcessIntensityFunction(cfg, device=self.device)
            self.nll_max_num_solutions = (
                cfg.MODEL.PROBABILISTIC_MODELING.NLL_MAX_NUM_SOLUTIONS
            )
            self.matching_distance = cfg.MODEL.PROBABILISTIC_MODELING.MATCHING_DISTANCE
            self.use_prediction_mixture = cfg.MODEL.PROBABILISTIC_MODELING.PPP.USE_PREDICTION_MIXTURE

        self.dropout_rate = cfg.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE
        self.use_dropout = self.dropout_rate != 0.0

        self.current_step = 0
        self.annealing_step = (
            cfg.SOLVER.STEPS[1]
            if cfg.MODEL.PROBABILISTIC_MODELING.ANNEALING_STEP <= 0
            else cfg.MODEL.PROBABILISTIC_MODELING.ANNEALING_STEP
        )

        # Define custom probabilistic head
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.head_in_features]
        self.head = ProbabilisticRetinaNetHead(
            cfg,
            self.use_dropout,
            self.dropout_rate,
            self.compute_cls_var,
            self.compute_bbox_cov,
            self.bbox_cov_dims,
            feature_shapes,
        )

        # Send to device
        self.to(self.device)
    
    def get_ppp_intensity_function(self):
        return self.ppp_intensity_function

    def forward(
        self, batched_inputs, return_anchorwise_output=False, num_mc_dropout_runs=-1
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

            return_anchorwise_output (bool): returns raw output for probabilistic inference

            num_mc_dropout_runs (int): perform efficient monte-carlo dropout runs by running only the head and
            not full neural network.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        # Update step
        try:
            self.current_step += get_event_storage().iter
        except:
            self.current_step += 1
        # Preprocess image
        images = self.preprocess_image(batched_inputs)

        # Extract features and generate anchors
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        anchors = self.anchor_generator(features)

        # MC_Dropout inference forward
        if num_mc_dropout_runs > 1:
            anchors = anchors * num_mc_dropout_runs
            features = features * num_mc_dropout_runs
            output_dict = self.produce_raw_output(anchors, features)
            return output_dict

        # Regular inference forward
        if return_anchorwise_output:
            return self.produce_raw_output(anchors, features)

        # Training and validation forward
        (
            pred_logits,
            pred_anchor_deltas,
            pred_logits_vars,
            pred_anchor_deltas_vars,
        ) = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        if pred_logits_vars is not None:
            pred_logits_vars = [
                permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits_vars
            ]
        if pred_anchor_deltas_vars is not None:
            pred_anchor_deltas_vars = [
                permute_to_N_HWA_K(x, self.bbox_cov_dims)
                for x in pred_anchor_deltas_vars
            ]

        if self.training:
            assert (
                "instances" in batched_inputs[0]
            ), "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_classes, gt_boxes = self.label_anchors(anchors, gt_instances)

            self.anchors = torch.cat(
                [Boxes.cat(anchors).tensor for i in range(len(gt_instances))], 0
            )

            # Loss is computed based on what values are to be estimated by the neural
            # network
            losses = self.losses(
                anchors,
                gt_classes,
                gt_boxes,
                pred_logits,
                pred_anchor_deltas,
                pred_logits_vars,
                pred_anchor_deltas_vars,
                gt_instances,
                images.image_sizes,
            )

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits, pred_anchor_deltas, images.image_sizes
                    )
                    self.visualize_training(
                        batched_inputs,
                        results,
                        pred_logits,
                        pred_anchor_deltas,
                        pred_anchor_deltas_vars,
                        anchors,
                    )
            return losses
        else:
            results = self.inference(
                anchors, pred_logits, pred_anchor_deltas, images.image_sizes
            )
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image[0], height, width)
                processed_results.append({"instances": r})
            return processed_results

    def visualize_training(
        self,
        batched_inputs,
        results,
        pred_logits,
        pred_anchor_deltas,
        pred_anchor_deltas_vars,
        anchors,
    ):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        pred_instaces, kept_idx = results
        assert len(batched_inputs) == len(
            pred_instaces
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)

        # Extract NMS kept predictions
        box_scores = torch.cat([logits.squeeze() for logits in pred_logits])[
            kept_idx
        ].sigmoid()
        box_scores = torch.cat(
            (box_scores, 1 - pred_instaces[image_index].scores.unsqueeze(-1)), dim=-1
        )
        anchor_deltas = torch.cat([delta.squeeze() for delta in pred_anchor_deltas])[
            kept_idx
        ]
        anchor_delta_vars = torch.cat(
            [var.squeeze() for var in pred_anchor_deltas_vars]
        )[kept_idx]
        anchor_boxes = torch.cat([box.tensor.squeeze() for box in anchors])[kept_idx]
        cholesky_decomp = covariance_output_to_cholesky(anchor_delta_vars)

        ######## Get covariance for corner coordinates instead #########
        multivariate_normal_samples = torch.distributions.MultivariateNormal(
            anchor_deltas, scale_tril=cholesky_decomp
        )

        # Define monte-carlo samples
        distributions_samples = multivariate_normal_samples.rsample((1000,))
        distributions_samples = torch.transpose(
            torch.transpose(distributions_samples, 0, 1), 1, 2
        )
        samples_proposals = torch.repeat_interleave(
            anchor_boxes.unsqueeze(2), 1000, dim=2
        )

        # Transform samples from deltas to boxes

        box_transform = inference_utils.SampleBox2BoxTransform(
            self.box2box_transform.weights
        )
        t_dist_samples = box_transform.apply_samples_deltas(
            distributions_samples, samples_proposals
        )

        # Compute samples mean and covariance matrices.
        _, boxes_covars = inference_utils.compute_mean_covariance_torch(t_dist_samples)

        # Scale if image has been reshaped during processing
        scale_x, scale_y = (
            img.shape[1] / pred_instaces[image_index].image_size[1],
            img.shape[0] / pred_instaces[image_index].image_size[0],
        )
        scaling = torch.tensor(np.stack([scale_x, scale_y, scale_x, scale_y]) ** 2).to(
            device=boxes_covars.device
        )
        boxes_covars = (boxes_covars * scaling).float()
        processed_results = detector_postprocess(
            pred_instaces[image_index], img.shape[0], img.shape[1]
        )
        predicted_boxes = processed_results.pred_boxes.tensor

        if self.bbox_cov_dist_type == "gaussian":
            reg_distribution = (
                lambda x, y: distributions.multivariate_normal.MultivariateNormal(x, y)
            )
        elif self.bbox_cov_dist_type == "laplacian":
            reg_distribution = lambda x, y: distributions.laplace.Laplace(
                loc=x, scale=(y.diagonal(dim1=-2, dim2=-1) / np.sqrt(2))
            )
        else:
            raise Exception(
                f"Bounding box uncertainty distribution {self.bbox_cov_dist_type} is not available."
            )

        associations = negative_log_likelihood_matching(
            box_scores,
            box_regs=predicted_boxes.unsqueeze(1).repeat(1, 80, 1),
            box_covars=boxes_covars.unsqueeze(1).repeat(1, 80, 1, 1),
            gt_box=batched_inputs[image_index]["instances"].gt_boxes.tensor,
            gt_class=batched_inputs[image_index]["instances"].gt_classes,
            image_size=img.shape,
            reg_distribution=reg_distribution,
            device=boxes_covars.device,
            intensity_func=self.ppp_intensity_function,
            max_n_solutions=1,
        )

        ################# Draw results ####################
        color_map = cm.get_cmap("tab20")
        num_gt = batched_inputs[image_index]["instances"].gt_boxes.tensor.shape[0]
        gt_colors = [color_map(i) for i in range(num_gt)]
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(
            boxes=batched_inputs[image_index]["instances"].gt_boxes,
            assigned_colors=gt_colors,
        )
        anno_img = v_gt.get_image()

        num_preds = len(boxes_covars)
        pred_colors = [(0.0, 0.0, 0.0, 1.0)] * num_preds
        for i in range(num_preds):
            matched_gt = associations[0, i, 1]
            if matched_gt >= 0:
                pred_colors[i] = color_map(matched_gt)

        pred_labels = [
            f"{pred_class.item()}: {round(pred_score.item(),2)}"
            for pred_class, pred_score in zip(
                pred_instaces[image_index].pred_classes,
                pred_instaces[image_index].scores,
            )
        ]
        v_pred = ProbabilisticVisualizer(img, None)
        v_pred = v_pred.overlay_covariance_instances(
            boxes=predicted_boxes[:max_boxes].detach().cpu().numpy(),
            covariance_matrices=boxes_covars[:max_boxes].detach().cpu().numpy(),
            assigned_colors=pred_colors,
            labels=pred_labels[:max_boxes],
        )

        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = (
            f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        )
        storage.put_image(vis_name, vis_img)

    def losses(
        self,
        anchors,
        gt_classes,
        gt_boxes,
        pred_class_logits,
        pred_anchor_deltas,
        pred_class_logits_var=None,
        pred_bbox_cov=None,
        gt_instances=None,
        image_sizes: List[Tuple[int, int]] = [],
    ):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits`, `pred_anchor_deltas`, `pred_class_logits_var` and `pred_bbox_cov`, see
                :meth:`RetinaNetHead.forward`.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_classes)
        gt_labels = torch.stack(gt_classes)  # (N, R)
        # Do NMS before reshaping stuff
        if self.bbox_cov_loss == "pmb_negative_log_likelihood":
            with torch.no_grad():
                nms_results = self.inference(
                    anchors, pred_class_logits, pred_anchor_deltas, image_sizes
                )

        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
        gt_anchor_deltas = [
            self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes
        ]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos_anchors, 1)

        # classification and regression loss

        # Shapes:
        # (N x R, K) for class_logits and class_logits_var.
        # (N x R, 4), (N x R x 10) for pred_anchor_deltas and pred_class_bbox_cov respectively.

        # Transform per-feature layer lists to a single tensor
        pred_class_logits = cat(pred_class_logits, dim=1)
        pred_anchor_deltas = cat(pred_anchor_deltas, dim=1)

        if pred_class_logits_var is not None:
            pred_class_logits_var = cat(pred_class_logits_var, dim=1)

        if pred_bbox_cov is not None:
            pred_bbox_cov = cat(pred_bbox_cov, dim=1)

        gt_classes_target = torch.nn.functional.one_hot(
            gt_labels[valid_mask], num_classes=self.num_classes + 1
        )[:, :-1].to(
            pred_class_logits[0].dtype
        )  # no loss for the last (background) class

        # Classification losses
        if self.compute_cls_var:
            # Compute classification variance according to:
            # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017
            if self.cls_var_loss == "loss_attenuation":
                num_samples = self.cls_var_num_samples
                # Compute standard deviation
                pred_class_logits_var = torch.sqrt(
                    torch.exp(pred_class_logits_var[valid_mask])
                )

                pred_class_logits = pred_class_logits[valid_mask]

                # Produce normal samples using logits as the mean and the standard deviation computed above
                # Scales with GPU memory. 12 GB ---> 3 Samples per anchor for
                # COCO dataset.
                univariate_normal_dists = distributions.normal.Normal(
                    pred_class_logits, scale=pred_class_logits_var
                )

                pred_class_stochastic_logits = univariate_normal_dists.rsample(
                    (num_samples,)
                )
                pred_class_stochastic_logits = pred_class_stochastic_logits.view(
                    (
                        pred_class_stochastic_logits.shape[1] * num_samples,
                        pred_class_stochastic_logits.shape[2],
                        -1,
                    )
                )
                pred_class_stochastic_logits = pred_class_stochastic_logits.squeeze(2)

                # Produce copies of the target classes to match the number of
                # stochastic samples.
                gt_classes_target = torch.unsqueeze(gt_classes_target, 0)
                gt_classes_target = torch.repeat_interleave(
                    gt_classes_target, num_samples, dim=0
                ).view(
                    (
                        gt_classes_target.shape[1] * num_samples,
                        gt_classes_target.shape[2],
                        -1,
                    )
                )
                gt_classes_target = gt_classes_target.squeeze(2)

                # Produce copies of the target classes to form the stochastic
                # focal loss.
                loss_cls = (
                    sigmoid_focal_loss_jit(
                        pred_class_stochastic_logits,
                        gt_classes_target,
                        alpha=self.focal_loss_alpha,
                        gamma=self.focal_loss_gamma,
                        reduction="sum",
                    )
                    / (num_samples * max(1, self.loss_normalizer))
                )
            else:
                raise ValueError(
                    "Invalid classification loss name {}.".format(self.bbox_cov_loss)
                )
        else:
            # Standard loss computation in case one wants to use this code
            # without any probabilistic inference.
            loss_cls = (
                sigmoid_focal_loss_jit(
                    pred_class_logits[valid_mask],
                    gt_classes_target,
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="sum",
                )
                / max(1, self.loss_normalizer)
            )

        # Compute Regression Loss
        if self.bbox_cov_loss == "pmb_negative_log_likelihood":
            og_pred_anchor_deltas = pred_anchor_deltas

        pred_anchor_deltas = pred_anchor_deltas[pos_mask]
        gt_anchors_deltas = gt_anchor_deltas[pos_mask]
        if self.compute_bbox_cov:
            # We have to clamp the output variance else probabilistic metrics
            # go to infinity.
            if self.bbox_cov_loss == "pmb_negative_log_likelihood":
                og_pred_bbox_cov = pred_bbox_cov

            pred_bbox_cov = clamp_log_variance(pred_bbox_cov[pos_mask])
            if self.bbox_cov_loss == "negative_log_likelihood":
                if self.bbox_cov_type == "diagonal":
                    # Compute regression variance according to:
                    # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017
                    # This implementation with smooth_l1_loss outperforms using
                    # torch.distribution.multivariate_normal. Losses might have different numerical values
                    # since we do not include constants in this implementation.
                    loss_box_reg = (
                        0.5
                        * torch.exp(-pred_bbox_cov)
                        * smooth_l1_loss(
                            pred_anchor_deltas,
                            gt_anchors_deltas,
                            beta=self.smooth_l1_beta,
                        )
                    )
                    loss_covariance_regularize = 0.5 * pred_bbox_cov
                    loss_box_reg += loss_covariance_regularize

                    # Sum over all elements
                    loss_box_reg = torch.sum(loss_box_reg) / max(
                        1, self.loss_normalizer
                    )
                else:
                    # Multivariate negative log likelihood. Implemented with
                    # pytorch multivariate_normal.log_prob function. Custom implementations fail to finish training
                    # due to NAN loss.

                    # This is the Cholesky decomposition of the covariance matrix. We reconstruct it from 10 estimated
                    # parameters as a lower triangular matrix.
                    forecaster_cholesky = covariance_output_to_cholesky(pred_bbox_cov)

                    # Compute multivariate normal distribution using torch
                    # distribution functions.
                    multivariate_normal_dists = (
                        distributions.multivariate_normal.MultivariateNormal(
                            pred_anchor_deltas, scale_tril=forecaster_cholesky
                        )
                    )

                    loss_box_reg = -multivariate_normal_dists.log_prob(
                        gt_anchors_deltas
                    )
                    loss_box_reg = torch.sum(loss_box_reg) / max(
                        1, self.loss_normalizer
                    )

            elif self.bbox_cov_loss == "second_moment_matching":
                # Compute regression covariance using second moment matching.
                loss_box_reg = smooth_l1_loss(
                    pred_anchor_deltas, gt_anchors_deltas, beta=self.smooth_l1_beta
                )

                # Compute errors
                errors = pred_anchor_deltas - gt_anchors_deltas

                if self.bbox_cov_type == "diagonal":
                    # Compute second moment matching term.
                    second_moment_matching_term = smooth_l1_loss(
                        torch.exp(pred_bbox_cov), errors ** 2, beta=self.smooth_l1_beta
                    )
                    loss_box_reg += second_moment_matching_term
                    loss_box_reg = torch.sum(loss_box_reg) / max(
                        1, self.loss_normalizer
                    )
                else:
                    # Compute second moment matching term.
                    errors = torch.unsqueeze(errors, 2)
                    gt_error_covar = torch.matmul(errors, torch.transpose(errors, 2, 1))

                    # This is the cholesky decomposition of the covariance matrix. We reconstruct it from 10 estimated
                    # parameters as a lower triangular matrix.
                    forecaster_cholesky = covariance_output_to_cholesky(pred_bbox_cov)

                    predicted_covar = torch.matmul(
                        forecaster_cholesky, torch.transpose(forecaster_cholesky, 2, 1)
                    )

                    second_moment_matching_term = smooth_l1_loss(
                        predicted_covar,
                        gt_error_covar,
                        beta=self.smooth_l1_beta,
                        reduction="sum",
                    )

                    loss_box_reg = (
                        torch.sum(loss_box_reg) + second_moment_matching_term
                    ) / max(1, self.loss_normalizer)

            elif self.bbox_cov_loss == "energy_loss":
                # Compute regression variance according to energy score loss.
                forecaster_means = pred_anchor_deltas

                # Compute forecaster cholesky. Takes care of diagonal case
                # automatically.
                forecaster_cholesky = covariance_output_to_cholesky(pred_bbox_cov)

                # Define normal distribution samples. To compute energy score,
                # we need i+1 samples.

                # Define per-anchor Distributions
                multivariate_normal_dists = (
                    distributions.multivariate_normal.MultivariateNormal(
                        forecaster_means, scale_tril=forecaster_cholesky
                    )
                )

                # Define Monte-Carlo Samples
                distributions_samples = multivariate_normal_dists.rsample(
                    (self.bbox_cov_num_samples + 1,)
                )

                distributions_samples_1 = distributions_samples[
                    0 : self.bbox_cov_num_samples, :, :
                ]
                distributions_samples_2 = distributions_samples[
                    1 : self.bbox_cov_num_samples + 1, :, :
                ]

                # Compute energy score
                gt_anchors_deltas_samples = torch.repeat_interleave(
                    gt_anchors_deltas.unsqueeze(0), self.bbox_cov_num_samples, dim=0
                )

                energy_score_first_term = (
                    2.0
                    * smooth_l1_loss(
                        distributions_samples_1,
                        gt_anchors_deltas_samples,
                        beta=self.smooth_l1_beta,
                        reduction="sum",
                    )
                    / self.bbox_cov_num_samples
                )  # First term

                energy_score_second_term = (
                    -smooth_l1_loss(
                        distributions_samples_1,
                        distributions_samples_2,
                        beta=self.smooth_l1_beta,
                        reduction="sum",
                    )
                    / self.bbox_cov_num_samples
                )  # Second term

                # Final Loss
                loss_box_reg = (
                    energy_score_first_term + energy_score_second_term
                ) / max(1, self.loss_normalizer)

            elif self.bbox_cov_loss == "pmb_negative_log_likelihood":
                pred_class_scores = pred_class_logits.sigmoid()
                losses = self.nll_od_loss_with_nms(
                    nms_results,
                    gt_instances,
                    anchors,
                    pred_class_scores,
                    og_pred_anchor_deltas,
                    og_pred_bbox_cov,
                    image_sizes,
                )
                loss_box_reg = losses["loss_box_reg"]
                use_nll_loss = True

            else:
                raise ValueError(
                    "Invalid regression loss name {}.".format(self.bbox_cov_loss)
                )

            # Perform loss annealing. Essential for reliably training variance estimates using NLL in RetinaNet.
            # For energy score and second moment matching, this is optional.
            standard_regression_loss = (
                smooth_l1_loss(
                    pred_anchor_deltas,
                    gt_anchors_deltas,
                    beta=self.smooth_l1_beta,
                    reduction="sum",
                )
                / max(1, self.loss_normalizer)
            )

            probabilistic_loss_weight = get_probabilistic_loss_weight(
                self.current_step, self.annealing_step
            )
            loss_box_reg = (
                1.0 - probabilistic_loss_weight
            ) * standard_regression_loss + probabilistic_loss_weight * loss_box_reg

            if self.bbox_cov_loss == "pmb_negative_log_likelihood":
                loss_cls = (1.0 - probabilistic_loss_weight) * loss_cls

        else:
            # Standard regression loss in case no variance is needed to be
            # estimated.
            loss_box_reg = (
                smooth_l1_loss(
                    pred_anchor_deltas,
                    gt_anchors_deltas,
                    beta=self.smooth_l1_beta,
                    reduction="sum",
                )
                / max(1, self.loss_normalizer)
            )

        if use_nll_loss:
            losses["loss_cls"] = loss_cls
            losses["loss_box_reg"] = loss_box_reg
        else:
            losses = {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

        return losses

    def nll_od_loss_with_nms(
        self,
        nms_results,
        gt_instances,
        anchors,
        scores,
        deltas,
        pred_covs,
        image_shapes,
    ):
        if "log_prob" in self.matching_distance and self.matching_distance != "log_prob":
            covar_scaling = float(self.matching_distance.split("_")[-1])
            matching_distance = "log_prob"
        else:
            covar_scaling = 1
            matching_distance = self.matching_distance

        self.ppp_intensity_function.update_distribution()

        instances, kept_idx = nms_results
        bs = len(instances)

        boxes = [
            self.box2box_transform.apply_deltas(delta, anchors) for delta in deltas
        ]

        nll_pred_cov = [
            pred_cov[kept].unsqueeze(1).repeat(1, self.num_classes, 1)
            for pred_cov, kept in zip(pred_covs, kept_idx)
        ]
        nll_pred_cov = [covariance_output_to_cholesky(cov) for cov in nll_pred_cov]
        nll_scores = [score[kept] for score, kept in zip(scores, kept_idx)]
        nll_pred_deltas = [
            delta[kept].unsqueeze(1).repeat(1, self.num_classes, 1)
            for delta, kept in zip(deltas, kept_idx)
        ]

        gt_boxes = [instances.gt_boxes.tensor for instances in gt_instances]
        nll_gt_classes = [instances.gt_classes for instances in gt_instances]
        kept_proposals = [anchors[idx] for idx in kept_idx]

        trans_func = lambda x,y: self.box2box_transform.apply_deltas(x,y)
        box_means = []
        box_chols = []
        for i in range(bs):
            box_mean, box_chol = unscented_transform(nll_pred_deltas[i], nll_pred_cov[i], kept_proposals[i], trans_func)
            box_means.append(box_mean)
            box_chols.append(box_chol)

        if self.bbox_cov_dist_type == "gaussian":
            regression_dist = (
                lambda x, y: distributions.multivariate_normal.MultivariateNormal(
                    loc=x, scale_tril=y
                )
            )
        elif self.bbox_cov_dist_type == "laplacian":
            # Map cholesky decomp to laplacian scale
            regression_dist = lambda x, y: distributions.laplace.Laplace(
                loc=x, scale=y.diagonal(dim1=-2, dim2=-1) / np.sqrt(2)
            )
        else:
            raise Exception(
                f"Bounding box uncertainty distribution {self.bbox_cov_dist_type} is not available."
            )

        nll_scores = [
            torch.cat(
                (
                    nll_scores[i],
                    (
                        1
                        - nll_scores[i][
                            torch.arange(len(kept_idx[i])), instances[i].pred_classes
                        ]
                    ).unsqueeze(-1),
                ),
                dim=-1,
            )
            for i in range(bs)
        ]
        # Clamp for numerical stability
        nll_scores = [scores.clamp(1e-6, 1 - 1e-6) for scores in nll_scores]

        if self.use_prediction_mixture:
            ppps = []
            src_boxes_tot = []
            src_box_chol_tot = []
            src_boxes_deltas_tot = []
            src_boxes_deltas_chol_tot = []
            src_scores_tot = []
            gt_box_deltas = []
            for i in range(bs):
                image_shape = image_shapes[i]
                h,w = image_shape
                scaling = torch.tensor([1/w,1/h],device=box_means[i].device).repeat(2)
                pred_box_means = box_means[i]*scaling
                pred_box_chols = torch.diag_embed(scaling)@box_chols[i]
                pred_box_deltas = nll_pred_deltas[i]
                pred_box_delta_chols = nll_pred_cov[i]
                pred_cls_probs = nll_scores[i]

                #max_conf = pred_cls_probs[..., :num_classes].max(dim=1)[0]
                max_conf = 1 - pred_cls_probs[..., -1]
                ppp_preds_idx = (
                    max_conf <= self.ppp_intensity_function.ppp_confidence_thres
                )

                props = kept_proposals[i][ppp_preds_idx.logical_not()]

                # Get delta between each GT and proposal, batch-wise
                tmp = torch.stack(
                    [
                        self.box2box_transform.get_deltas(
                            props,
                            gt_boxes[i][j].unsqueeze(0).repeat(len(props), 1),
                        )
                        for j in range(len(gt_boxes[i]))
                    ]
                )

                gt_box_deltas.append(
                    tmp.permute(1, 0, 2)
                )  # [gt,pred,boxdim] -> [pred, gt, boxdim]

                gt_boxes[i] = gt_boxes[i]*scaling

                mixture_dict = {}
                mixture_dict["weights"] = max_conf[ppp_preds_idx]
                mixture_dict["means"] = pred_box_means[ppp_preds_idx, 0]
                mixture_dict["covs"] = pred_box_chols[ppp_preds_idx, 0]@pred_box_chols[ppp_preds_idx, 0].transpose(-1,-2)
                mixture_dict["cls_probs"] = pred_cls_probs[ppp_preds_idx, :self.num_classes]
                mixture_dict["reg_dist_type"] = self.bbox_cov_dist_type

                if self.bbox_cov_dist_type == "gaussian":
                    mixture_dict[
                        "reg_dist"
                    ] = distributions.multivariate_normal.MultivariateNormal
                    mixture_dict["reg_kwargs"] = {
                        "scale_tril": pred_box_chols[ppp_preds_idx, 0]
                    }
                elif self.bbox_cov_dist_type == "laplacian":
                    mixture_dict["reg_dist"] = distributions.laplace.Laplace
                    mixture_dict["reg_kwargs"] = {
                        "scale": (
                            pred_box_chols[ppp_preds_idx, 0].diagonal(dim1=-2, dim2=-1)
                            / np.sqrt(2)
                        )
                    }
                loss_ppp = PoissonPointUnion()
                loss_ppp.add_ppp(self.ppp_constructor({"predictions": mixture_dict}))
                loss_ppp.add_ppp(self.ppp_intensity_function)

                mixture_dict = {}
                mixture_dict["weights"] = max_conf[ppp_preds_idx]
                mixture_dict["means"] = pred_box_means[ppp_preds_idx, 0]
                
                scale_mat = torch.eye(pred_box_chols.shape[-1]).to(pred_box_chols.device)*covar_scaling
                scaled_chol = scale_mat@pred_box_chols[ppp_preds_idx, 0]
                mixture_dict["covs"] = (scaled_chol)@(scaled_chol.transpose(-1,-2))
                mixture_dict["cls_probs"] = pred_cls_probs[ppp_preds_idx, :self.num_classes]
                mixture_dict["reg_dist_type"] = self.bbox_cov_dist_type

                if self.bbox_cov_dist_type == "gaussian":
                    mixture_dict[
                        "reg_dist"
                    ] = distributions.multivariate_normal.MultivariateNormal
                    mixture_dict["reg_kwargs"] = {
                        "scale_tril": scaled_chol
                    }
                elif self.bbox_cov_dist_type == "laplacian":
                    mixture_dict["reg_dist"] = distributions.laplace.Laplace
                    mixture_dict["reg_kwargs"] = {
                        "scale": (
                            (scaled_chol).diagonal(dim1=-2, dim2=-1)
                            / np.sqrt(2)
                        )
                    }
                
                match_ppp = PoissonPointUnion()
                match_ppp.add_ppp(self.ppp_constructor({"predictions": mixture_dict}))
                match_ppp.add_ppp(self.ppp_intensity_function)
                ppps.append({"matching": match_ppp, "loss": loss_ppp})

                src_boxes_tot.append(pred_box_means[ppp_preds_idx.logical_not()])
                src_box_chol_tot.append(pred_box_chols[ppp_preds_idx.logical_not()])
                src_scores_tot.append(pred_cls_probs[ppp_preds_idx.logical_not()])
                src_boxes_deltas_tot.append(pred_box_deltas[ppp_preds_idx.logical_not()])
                src_boxes_deltas_chol_tot.append(pred_box_delta_chols[ppp_preds_idx.logical_not()])

            nll_pred_deltas = src_boxes_deltas_tot
            nll_pred_delta_chols = src_boxes_deltas_chol_tot
            nll_pred_boxes = src_boxes_tot
            nll_pred_cov = src_box_chol_tot
            nll_scores = src_scores_tot
            use_target_delta_matching = False
        elif self.ppp_intensity_function.ppp_intensity_type == "gaussian_mixture":
            ppps = []
            src_boxes_tot = []
            src_box_chol_tot = []
            src_boxes_deltas_tot = []
            src_boxes_deltas_chol_tot = []
            src_scores_tot = []
            gt_box_deltas = []
            for i in range(bs):
                image_shape = image_shapes[i]
                h,w = image_shape
                scaling = torch.tensor([1/w,1/h],device=box_means[i].device).repeat(2)
                pred_box_means = box_means[i]*scaling
                pred_box_chols = torch.diag_embed(scaling)@box_chols[i]
                pred_box_deltas = nll_pred_deltas[i]
                pred_box_delta_chols = nll_pred_cov[i]
                pred_cls_probs = nll_scores[i]
                props = kept_proposals[i]

                # Get delta between each GT and proposal, batch-wise
                tmp = torch.stack(
                    [
                        self.box2box_transform.get_deltas(
                            props,
                            gt_boxes[i][j].unsqueeze(0).repeat(len(props), 1),
                        )
                        for j in range(len(gt_boxes[i]))
                    ]
                )

                gt_box_deltas.append(
                    tmp.permute(1, 0, 2)
                )  # [gt,pred,boxdim] -> [pred, gt, boxdim]

                gt_boxes[i] = gt_boxes[i]*scaling

                src_boxes_tot.append(pred_box_means)
                src_box_chol_tot.append(pred_box_chols)
                src_scores_tot.append(pred_cls_probs)
                src_boxes_deltas_tot.append(pred_box_deltas)
                src_boxes_deltas_chol_tot.append(pred_box_delta_chols)

            nll_pred_deltas = src_boxes_deltas_tot
            nll_pred_delta_chols = src_boxes_deltas_chol_tot
            nll_pred_boxes = src_boxes_tot
            nll_pred_cov = src_box_chol_tot
            nll_scores = src_scores_tot
            use_target_delta_matching = False
            ppps = [{"loss": self.ppp_intensity_function, "matching": self.ppp_intensity_function}]*bs
        else:
            gt_box_deltas = []
            for i in range(len(gt_boxes)):
                # Get delta between each GT and proposal, batch-wise
                tmp = torch.stack(
                    [
                        self.box2box_transform.get_deltas(
                            kept_proposals[i],
                            gt_boxes[i][j].unsqueeze(0).repeat(len(kept_proposals[i]), 1),
                        )
                        for j in range(len(gt_boxes[i]))
                    ]
                )

                gt_box_deltas.append(
                    tmp.permute(1, 0, 2)
                )  # [gt,pred,boxdim] -> [pred, gt, boxdim]
            
            use_target_delta_matching = True
            ppps = [{"loss": self.ppp_intensity_function, "matching": self.ppp_intensity_function}]*bs
            nll_pred_delta_chols = nll_pred_cov
            nll_pred_deltas = nll_pred_deltas
            nll_pred_boxes = nll_pred_deltas
            nll_pred_cov = nll_pred_cov


        nll, associations, decompositions = negative_log_likelihood(
            nll_scores,
            nll_pred_boxes,
            nll_pred_cov,
            gt_boxes,
            nll_gt_classes,
            image_shapes,
            regression_dist,
            ppps,
            self.nll_max_num_solutions,
            target_deltas=gt_box_deltas,
            matching_distance=matching_distance,
            use_target_delta_matching=use_target_delta_matching,
            pred_deltas=nll_pred_deltas,
            pred_delta_chols=nll_pred_delta_chols,
        )

        # Save some stats
        storage = get_event_storage()
        num_classes = self.num_classes
        mean_variance = np.mean(
            [
                cov.diagonal(dim1=-2,dim2=-1)
                .pow(2)
                .mean()
                .item()
                for cov in nll_pred_cov
                if cov.shape[0] > 0
            ]
        )
        storage.put_scalar("nll/mean_covariance", mean_variance)
        ppp_intens = np.sum([ppp["loss"].integrate(
                torch.as_tensor(image_shapes).to(self.device), num_classes
            )
            .mean()
            .item()
            for ppp in ppps
            ])
        storage.put_scalar("nll/ppp_intensity", ppp_intens)

        reg_loss = np.mean(
            [
                np.clip(
                    decomp["matched_bernoulli_reg"][0]
                    / (decomp["num_matched_bernoulli"][0] + 1e-6),
                    -1e25,
                    1e25,
                )
                for decomp in decompositions
            ]
        )
        cls_loss_match = np.mean(
            [
                np.clip(
                    decomp["matched_bernoulli_cls"][0]
                    / (decomp["num_matched_bernoulli"][0] + 1e-6),
                    -1e25,
                    1e25,
                )
                for decomp in decompositions
            ]
        )
        cls_loss_no_match = np.mean(
            [
                np.clip(
                    decomp["unmatched_bernoulli"][0]
                    / (decomp["num_unmatched_bernoulli"][0] + 1e-6),
                    -1e25,
                    1e25,
                )
                for decomp in decompositions
            ]
        )

        # Collect all losses
        losses = dict()
        losses["loss_box_reg"] = nll
        # Add losses for logging, these do not propagate gradients
        losses["loss_regression"] = torch.tensor(reg_loss).to(nll.device)
        losses["loss_cls_matched"] = torch.tensor(cls_loss_match).to(nll.device)
        losses["loss_cls_unmatched"] = torch.tensor(cls_loss_no_match).to(nll.device)

        return losses

    def produce_raw_output(self, anchors, features):
        """
        Given anchors and features, produces raw pre-nms output to be used for custom fusion operations.
        """
        # Perform inference run
        (
            pred_logits,
            pred_anchor_deltas,
            pred_logits_vars,
            pred_anchor_deltas_vars,
        ) = self.head(features)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        if pred_logits_vars is not None:
            pred_logits_vars = [
                permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits_vars
            ]
        if pred_anchor_deltas_vars is not None:
            pred_anchor_deltas_vars = [
                permute_to_N_HWA_K(x, self.bbox_cov_dims)
                for x in pred_anchor_deltas_vars
            ]

        # Create raw output dictionary
        raw_output = {"anchors": anchors}

        # Shapes:
        # (N x R, K) for class_logits and class_logits_var.
        # (N x R, 4), (N x R x 10) for pred_anchor_deltas and pred_class_bbox_cov respectively.
        raw_output.update(
            {
                "box_cls": pred_logits,
                "box_delta": pred_anchor_deltas,
                "box_cls_var": pred_logits_vars,
                "box_reg_var": pred_anchor_deltas_vars,
            }
        )

        if (
            self.compute_bbox_cov
            and self.bbox_cov_loss == "pmb_negative_log_likelihood"
        ):
            ppp_output = self.ppp_intensity_function.get_weights()
            raw_output.update({"ppp": ppp_output})

        return raw_output

    def inference(
        self,
        anchors: List[Boxes],
        pred_logits: List[Tensor],
        pred_anchor_deltas: List[Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        """
        Arguments:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            pred_logits, pred_anchor_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[(h, w)]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results: List[Instances] = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(
                anchors, pred_logits_per_image, deltas_per_image, image_size
            )
            results.append(results_per_image)
        return [x[0] for x in results], [x[1] for x in results]

    def inference_single_image(
        self,
        anchors: List[Boxes],
        box_cls: List[Tensor],
        box_delta: List[Tensor],
        image_size: Tuple[int, int],
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []
        anchor_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            predicted_prob = box_cls_i.flatten().sigmoid()

            # Apply two filtering below to make NMS faster.
            # 1. Keep boxes with confidence score higher than threshold
            keep_idxs = predicted_prob > self.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(
                box_reg_i, anchors_i.tensor
            )

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)
            anchor_idxs_all.append(anchor_idxs)

        num_anchors_per_feat_lvl = [anchor.tensor.shape[0] for anchor in anchors]
        accum_anchor_nums = np.cumsum(num_anchors_per_feat_lvl).tolist()
        accum_anchor_nums = [0] + accum_anchor_nums
        anchor_idxs_all = [
            anchor_idx + prev_num_feats
            for anchor_idx, prev_num_feats in zip(anchor_idxs_all, accum_anchor_nums)
        ]
        boxes_all, scores_all, class_idxs_all, anchor_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all, anchor_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.test_nms_thresh)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result, anchor_idxs_all[keep]


class ProbabilisticRetinaNetHead(RetinaNetHead):
    """
    The head used in ProbabilisticRetinaNet for object class probability estimation, box regression, box covariance estimation.
    It has three subnets for the three tasks, with a common structure but separate parameters.
    """

    def __init__(
        self,
        cfg,
        use_dropout,
        dropout_rate,
        compute_cls_var,
        compute_bbox_cov,
        bbox_cov_dims,
        input_shape: List[ShapeSpec],
    ):
        super().__init__(cfg, input_shape)

        # Extract config information
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        self.compute_cls_var = compute_cls_var
        self.compute_bbox_cov = compute_bbox_cov
        self.bbox_cov_dims = bbox_cov_dims

        # For consistency all configs are grabbed from original RetinaNet
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU())

            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU())

            if self.use_dropout:
                cls_subnet.append(nn.Dropout(p=self.dropout_rate))
                bbox_subnet.append(nn.Dropout(p=self.dropout_rate))

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)

        self.cls_score = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )

        for modules in [
            self.cls_subnet,
            self.bbox_subnet,
            self.cls_score,
            self.bbox_pred,
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        # Create subnet for classification variance estimation.
        if self.compute_cls_var:
            self.cls_var = nn.Conv2d(
                in_channels,
                num_anchors * num_classes,
                kernel_size=3,
                stride=1,
                padding=1,
            )

            for layer in self.cls_var.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, -10.0)

        # Create subnet for bounding box covariance estimation.
        if self.compute_bbox_cov:
            self.bbox_cov = nn.Conv2d(
                in_channels,
                num_anchors * self.bbox_cov_dims,
                kernel_size=3,
                stride=1,
                padding=1,
            )

            for layer in self.bbox_cov.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.0001)
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            logits_var (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the variance of the logits modeled as a univariate
                Gaussian distribution at each spatial position for each of the A anchors and K object
                classes.

            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.

            bbox_cov (list[Tensor]): #lvl tensors, each has shape (N, Ax4 or Ax10, Hi, Wi).
                The tensor predicts elements of the box
                covariance values for every anchor. The dimensions of the box covarianc
                depends on estimating a full covariance (10) or a diagonal covariance matrix (4).
        """
        logits = []
        bbox_reg = []

        logits_var = []
        bbox_cov = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
            if self.compute_cls_var:
                logits_var.append(self.cls_var(self.cls_subnet(feature)))
            if self.compute_bbox_cov:
                bbox_cov.append(self.bbox_cov(self.bbox_subnet(feature)))

        return_vector = [logits, bbox_reg]

        if self.compute_cls_var:
            return_vector.append(logits_var)
        else:
            return_vector.append(None)

        if self.compute_bbox_cov:
            return_vector.append(bbox_cov)
        else:
            return_vector.append(None)

        return return_vector
