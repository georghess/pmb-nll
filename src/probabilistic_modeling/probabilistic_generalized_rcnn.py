import logging
from typing import Dict, List, Optional, Tuple, Union

# Detectron imports
import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import Conv2d, Linear, ShapeSpec, cat, get_norm
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from fvcore.nn import smooth_l1_loss

# Project imports
from probabilistic_inference.inference_utils import get_dir_alphas
from torch import distributions, nn
from torch.nn import functional as F

from probabilistic_modeling.losses import negative_log_likelihood, reshape_box_preds
from probabilistic_modeling.modeling_utils import (
    PoissonPointProcessIntensityFunction,
    clamp_log_variance,
    covariance_output_to_cholesky,
    get_probabilistic_loss_weight,
    unscented_transform,
    PoissonPointUnion,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@META_ARCH_REGISTRY.register()
class ProbabilisticGeneralizedRCNN(GeneralizedRCNN):
    """
    Probabilistic GeneralizedRCNN class.
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

        self.dropout_rate = cfg.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE
        self.use_dropout = self.dropout_rate != 0.0
        self.num_mc_dropout_runs = -1
        if (
            self.compute_bbox_cov
            and self.bbox_cov_loss == "pmb_negative_log_likelihood"
        ):
            ppp_constructor = lambda x: PoissonPointProcessIntensityFunction(
                cfg, **x
            )
            self.nll_max_num_solutions = (
                cfg.MODEL.PROBABILISTIC_MODELING.NLL_MAX_NUM_SOLUTIONS
            )

        self.current_step = 0

        # Define custom probabilistic head
        self.roi_heads.box_predictor = ProbabilisticFastRCNNOutputLayers(
            cfg,
            input_shape=self.roi_heads.box_head.output_shape,
            compute_cls_var=self.compute_cls_var,
            cls_var_loss=self.cls_var_loss,
            cls_var_num_samples=self.cls_var_num_samples,
            compute_bbox_cov=self.compute_bbox_cov,
            bbox_cov_loss=self.bbox_cov_loss,
            bbox_cov_type=self.bbox_cov_type,
            bbox_cov_dims=self.bbox_cov_dims,
            bbox_cov_num_samples=self.bbox_cov_num_samples,
            ppp_constructor=ppp_constructor,
            nll_max_num_solutions=self.nll_max_num_solutions,
            bbox_cov_dist_type=self.bbox_cov_dist_type,
            matching_distance=cfg.MODEL.PROBABILISTIC_MODELING.MATCHING_DISTANCE,
            use_prediction_mixture=cfg.MODEL.PROBABILISTIC_MODELING.PPP.USE_PREDICTION_MIXTURE,
        )

        # Send to device
        self.to(self.device)
    def get_ppp_intensity_function(self):
        return self.roi_heads.box_predictor.ppp_intensity_function

    def forward(
        self, batched_inputs, return_anchorwise_output=False, num_mc_dropout_runs=-1
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

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
        try:
            self.current_step += get_event_storage().iter
        except:
            self.current_step += 1

        if not self.training and num_mc_dropout_runs == -1:
            if return_anchorwise_output:
                return self.produce_raw_output(batched_inputs)
            else:
                return self.inference(batched_inputs)
        elif self.training and num_mc_dropout_runs > 1:
            self.num_mc_dropout_runs = num_mc_dropout_runs
            output_list = []
            for i in range(num_mc_dropout_runs):
                output_list.append(self.produce_raw_output(batched_inputs))
            return output_list

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances, current_step=self.current_step
        )
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                # TODO: implement to visualize probabilistic outputs
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def produce_raw_output(self, batched_inputs, detected_instances=None):
        """
        Run inference on the given inputs and return proposal-wise output for later postprocessing.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
        Returns:
            same as in :meth:`forward`.
        """
        raw_output = dict()

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            # Create raw output dictionary
            raw_output.update({"proposals": proposals[0]})

            results, _ = self.roi_heads(
                images,
                features,
                proposals,
                None,
                produce_raw_output=True,
                num_mc_dropout_runs=self.num_mc_dropout_runs,
            )
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        box_cls, box_delta, box_cls_var, box_reg_var = results

        raw_output.update(
            {
                "box_cls": box_cls,
                "box_delta": box_delta,
                "box_cls_var": box_cls_var,
                "box_reg_var": box_reg_var,
            }
        )
        if (
            self.compute_bbox_cov
            and self.bbox_cov_loss == "pmb_negative_log_likelihood"
        ):
            ppp_output = (
                self.roi_heads.box_predictor.ppp_intensity_function.get_weights()
            )
            raw_output.update({"ppp": ppp_output})

        return raw_output

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from core.visualization_tools.probabilistic_visualizer import (
            ProbabilisticVisualizer as Visualizer,
        )

        storage = get_event_storage()
        max_vis_prop = 20

        with torch.no_grad():
            self.eval()
            predictions = self.produce_raw_output(batched_inputs)
            self.train()
            predictions = (
                predictions["box_cls"],
                predictions["box_delta"],
                predictions["box_cls_var"],
                predictions["box_reg_var"],
            )
            _, _, _, pred_covs = predictions
            boxes = self.roi_heads.box_predictor.predict_boxes(predictions, proposals)
            scores = self.roi_heads.box_predictor.predict_probs(predictions, proposals)
            image_shapes = [x.image_size for x in proposals]

            # Apply NMS without score threshold
            instances, kept_idx = fast_rcnn_inference(
                boxes,
                scores,
                image_shapes,
                0.0,
                self.roi_heads.box_predictor.test_nms_thresh,
                self.roi_heads.box_predictor.test_topk_per_image,
            )

            num_prop_per_image = [len(p) for p in proposals]
            pred_covs = pred_covs.split(num_prop_per_image)

            pred_covs = [pred_cov[kept] for pred_cov, kept in zip(pred_covs, kept_idx)]
            pred_scores = [score[kept] for score, kept in zip(scores, kept_idx)]
            pred_boxes = [box[kept] for box, kept in zip(boxes, kept_idx)]

        for i, (input, prop) in enumerate(zip(batched_inputs, proposals)):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)

            v_pred = Visualizer(img, None)
            boxes = pred_boxes[i][0:box_size, :4].cpu().numpy()
            pred_cov_matrix = pred_covs[i][0:box_size, :4]
            pred_cov_matrix = clamp_log_variance(pred_cov_matrix)
            chol = covariance_output_to_cholesky(pred_cov_matrix)
            cov = (
                torch.matmul(chol, torch.transpose(chol, -1, -2)).cpu().detach().numpy()
            )

            v_pred = v_pred.overlay_covariance_instances(
                boxes=boxes, covariance_matrices=cov
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch


@ROI_HEADS_REGISTRY.register()
class ProbabilisticROIHeads(StandardROIHeads):
    """
    Probabilistic ROI heads, inherit from standard ROI heads so can be used with mask RCNN in theory.
    """

    def __init__(self, cfg, input_shape):
        super(ProbabilisticROIHeads, self).__init__(cfg, input_shape)

        self.is_mc_dropout_inference = False
        self.produce_raw_output = False
        self.current_step = 0

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        num_mc_dropout_runs=-1,
        produce_raw_output=False,
        current_step=0.0,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """

        self.is_mc_dropout_inference = num_mc_dropout_runs > 1
        self.produce_raw_output = produce_raw_output
        self.current_step = current_step

        del images
        if self.training and not self.is_mc_dropout_inference:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        # del targets

        if self.training and not self.is_mc_dropout_inference:
            losses = self._forward_box(features, proposals, targets)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, targets)
            if self.produce_raw_output:
                return pred_instances, {}
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        gt_instances: List[Instances],
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.produce_raw_output:
            return predictions

        if self.training:
            losses = self.box_predictor.losses(
                predictions, proposals, self.current_step, gt_instances
            )
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances


class ProbabilisticFastRCNNOutputLayers(nn.Module):
    """
    Four linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
      (3) box regression deltas covariance parameters (if needed)
      (4) classification logits variance (if needed)
    """

    @configurable
    def __init__(
        self,
        input_shape,
        *,
        box2box_transform,
        num_classes,
        cls_agnostic_bbox_reg=False,
        smooth_l1_beta=0.0,
        test_score_thresh=0.0,
        test_nms_thresh=0.5,
        test_topk_per_image=100,
        compute_cls_var=False,
        compute_bbox_cov=False,
        bbox_cov_dims=4,
        cls_var_loss="none",
        cls_var_num_samples=10,
        bbox_cov_loss="none",
        bbox_cov_type="diagonal",
        dropout_rate=0.0,
        annealing_step=0,
        bbox_cov_num_samples=1000,
        ppp_constructor=None,
        nll_max_num_solutions=5,
        bbox_cov_dist_type=None,
        matching_distance="log_prob",
        use_prediction_mixture=False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss.
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            compute_cls_var (bool): compute classification variance
            compute_bbox_cov (bool): compute box covariance regression parameters.
            bbox_cov_dims (int): 4 for diagonal covariance, 10 for full covariance.
            cls_var_loss (str): name of classification variance loss.
            cls_var_num_samples (int): number of samples to be used for loss computation. Usually between 10-100.
            bbox_cov_loss (str): name of box covariance loss.
            bbox_cov_type (str): 'diagonal' or 'full'. This is used to train with loss functions that accept both types.
            dropout_rate (float): 0-1, probability of drop.
            annealing_step (int): step used for KL-divergence in evidential loss to fully be functional.
            ppp_intensity_function (func): function that returns PPP intensity given sample box
            nll_max_num_solutions (int): Maximum NLL solutions to consider when computing NLL-PMB loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = (
            input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        )

        self.compute_cls_var = compute_cls_var
        self.compute_bbox_cov = compute_bbox_cov

        self.bbox_cov_dims = bbox_cov_dims
        self.bbox_cov_num_samples = bbox_cov_num_samples

        self.dropout_rate = dropout_rate
        self.use_dropout = self.dropout_rate != 0.0

        self.cls_var_loss = cls_var_loss
        self.cls_var_num_samples = cls_var_num_samples

        self.annealing_step = annealing_step

        self.bbox_cov_loss = bbox_cov_loss
        self.bbox_cov_type = bbox_cov_type
        self.bbox_cov_dist_type = bbox_cov_dist_type

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1.0 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        if self.compute_cls_var:
            self.cls_var = Linear(input_size, num_classes + 1)
            nn.init.normal_(self.cls_var.weight, std=0.0001)
            nn.init.constant_(self.cls_var.bias, 0)

        if self.compute_bbox_cov:
            self.bbox_cov = Linear(input_size, num_bbox_reg_classes * bbox_cov_dims)
            nn.init.normal_(self.bbox_cov.weight, std=0.0001)
            nn.init.constant_(self.bbox_cov.bias, 0.0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

        self.ppp_intensity_function = ppp_constructor({"device": device}) if ppp_constructor is not None else None
        self.ppp_constructor = ppp_constructor
        self.nll_max_num_solutions = nll_max_num_solutions
        self.matching_distance = matching_distance
        self.use_prediction_mixture = use_prediction_mixture

    @classmethod
    def from_config(
        cls,
        cfg,
        input_shape,
        compute_cls_var,
        cls_var_loss,
        cls_var_num_samples,
        compute_bbox_cov,
        bbox_cov_loss,
        bbox_cov_type,
        bbox_cov_dims,
        bbox_cov_num_samples,
        ppp_constructor,
        nll_max_num_solutions,
    ):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(
                weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
            ),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "compute_cls_var": compute_cls_var,
            "cls_var_loss": cls_var_loss,
            "cls_var_num_samples": cls_var_num_samples,
            "compute_bbox_cov": compute_bbox_cov,
            "bbox_cov_dims": bbox_cov_dims,
            "bbox_cov_loss": bbox_cov_loss,
            "bbox_cov_type": bbox_cov_type,
            "dropout_rate": cfg.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE,
            "annealing_step": cfg.SOLVER.STEPS[1] if cfg.MODEL.PROBABILISTIC_MODELING.ANNEALING_STEP <= 0 else cfg.MODEL.PROBABILISTIC_MODELING.ANNEALING_STEP,
            "bbox_cov_num_samples": bbox_cov_num_samples,
            "ppp_constructor": ppp_constructor,
            "nll_max_num_solutions" : nll_max_num_solutions,
            'bbox_cov_dist_type': cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.COVARIANCE_TYPE,
            "use_prediction_mixture": cfg.MODEL.PROBABILISTIC_MODELING.PPP.USE_PREDICTION_MIXTURE
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            Tensor: Nx(K+1) logits for each box
            Tensor: Nx4 or Nx(Kx4) bounding box regression deltas.
            Tensor: Nx(K+1) logits variance for each box.
            Tensor: Nx4(10) or Nx(Kx4(10)) covariance matrix parameters. 4 if diagonal, 10 if full.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)

        # Compute logits variance if needed
        if self.compute_cls_var:
            score_vars = self.cls_var(x)
        else:
            score_vars = None

        # Compute box covariance if needed
        if self.compute_bbox_cov:
            proposal_covs = self.bbox_cov(x)
        else:
            proposal_covs = None

        return scores, proposal_deltas, score_vars, proposal_covs

    def losses(self, predictions, proposals, current_step=0, gt_instances=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
            current_step: current optimizer step. Used for losses with an annealing component.
            gt_instances: list of ground truth instances

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        global device

        # Overwrite later
        use_nll_loss = False

        (
            pred_class_logits,
            pred_proposal_deltas,
            pred_class_logits_var,
            pred_proposal_covs,
        ) = predictions

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            proposals_boxes = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not proposals_boxes.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        else:
            proposals_boxes = Boxes(
                torch.zeros(0, 4, device=pred_proposal_deltas.device)
            )

        no_instances = len(proposals) == 0  # no instances found

        # Compute Classification Loss
        if no_instances:
            # TODO 0.0 * pred.sum() is enough since PT1.6
            loss_cls = 0.0 * F.cross_entropy(
                pred_class_logits,
                torch.zeros(0, dtype=torch.long, device=pred_class_logits.device),
                reduction="sum",
            )
        else:
            if self.compute_cls_var:
                # Compute classification variance according to:
                # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017
                if self.cls_var_loss == "loss_attenuation":
                    num_samples = self.cls_var_num_samples

                    # Compute standard deviation
                    pred_class_logits_var = torch.sqrt(torch.exp(pred_class_logits_var))

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
                    pred_class_logits = pred_class_stochastic_logits.squeeze(2)

                    # Produce copies of the target classes to match the number of
                    # stochastic samples.
                    gt_classes_target = torch.unsqueeze(gt_classes, 0)
                    gt_classes_target = torch.repeat_interleave(
                        gt_classes_target, num_samples, dim=0
                    ).view((gt_classes_target.shape[1] * num_samples, -1))
                    gt_classes_target = gt_classes_target.squeeze(1)

                    loss_cls = F.cross_entropy(
                        pred_class_logits, gt_classes_target, reduction="mean"
                    )
            elif self.cls_var_loss == "evidential":
                # ToDo: Currently does not provide any reasonable mAP Results
                # (15% mAP)

                # Assume dirichlet parameters are output.
                alphas = get_dir_alphas(pred_class_logits)

                # Get sum of all alphas
                dirichlet_s = alphas.sum(1).unsqueeze(1)

                # Generate one hot vectors for ground truth
                one_hot_vectors = torch.nn.functional.one_hot(
                    gt_classes, alphas.shape[1]
                )

                # Compute loss. This loss attempts to put all evidence on the
                # correct location.
                per_instance_loss = one_hot_vectors * (
                    torch.digamma(dirichlet_s) - torch.digamma(alphas)
                )

                # Compute KL divergence regularizer loss
                estimated_dirichlet = torch.distributions.dirichlet.Dirichlet(
                    (alphas - 1.0) * (1.0 - one_hot_vectors) + 1.0
                )
                uniform_dirichlet = torch.distributions.dirichlet.Dirichlet(
                    torch.ones_like(one_hot_vectors).type(torch.FloatTensor).to(device)
                )
                kl_regularization_loss = torch.distributions.kl.kl_divergence(
                    estimated_dirichlet, uniform_dirichlet
                )

                # Compute final loss
                annealing_multiplier = torch.min(
                    torch.as_tensor(current_step / self.annealing_step).to(device),
                    torch.as_tensor(1.0).to(device),
                )

                per_proposal_loss = (
                    per_instance_loss.sum(1)
                    + annealing_multiplier * kl_regularization_loss
                )

                # Compute evidence auxiliary loss
                evidence_maximization_loss = smooth_l1_loss(
                    dirichlet_s,
                    100.0 * torch.ones_like(dirichlet_s).to(device),
                    beta=self.smooth_l1_beta,
                    reduction="mean",
                )

                evidence_maximization_loss *= annealing_multiplier

                # Compute final loss
                foreground_loss = per_proposal_loss[
                    (gt_classes >= 0) & (gt_classes < pred_class_logits.shape[1] - 1)
                ]
                background_loss = per_proposal_loss[
                    gt_classes == pred_class_logits.shape[1] - 1
                ]

                loss_cls = (
                    torch.mean(foreground_loss) + torch.mean(background_loss)
                ) / 2 + 0.01 * evidence_maximization_loss
            else:
                loss_cls = F.cross_entropy(
                    pred_class_logits, gt_classes, reduction="mean"
                )

        # Compute regression loss:
        if no_instances:
            # TODO 0.0 * pred.sum() is enough since PT1.6
            loss_box_reg = 0.0 * smooth_l1_loss(
                pred_proposal_deltas,
                torch.zeros_like(pred_proposal_deltas),
                0.0,
                reduction="sum",
            )
        else:
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                proposals_boxes.tensor, gt_boxes.tensor
            )
            box_dim = gt_proposal_deltas.size(1)  # 4 or 5
            cls_agnostic_bbox_reg = pred_proposal_deltas.size(1) == box_dim
            device = pred_proposal_deltas.device

            bg_class_ind = pred_class_logits.shape[1] - 1

            # Box delta loss is only computed between the prediction for the gt class k
            # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
            # for non-gt classes and background.
            # Empty fg_inds produces a valid loss of zero as long as the size_average
            # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
            # and would produce a nan loss).
            fg_inds = torch.nonzero(
                (gt_classes >= 0) & (gt_classes < bg_class_ind), as_tuple=True
            )[0]
            if cls_agnostic_bbox_reg:
                # pred_proposal_deltas only corresponds to foreground class for
                # agnostic
                gt_class_cols = torch.arange(box_dim, device=device)
            else:
                fg_gt_classes = gt_classes[fg_inds]
                # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
                # where b is the dimension of box representation (4 or 5)
                # Note that compared to Detectron1,
                # we do not perform bounding box regression for background
                # classes.
                gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(
                    box_dim, device=device
                )
                gt_covar_class_cols = self.bbox_cov_dims * fg_gt_classes[
                    :, None
                ] + torch.arange(self.bbox_cov_dims, device=device)

            loss_reg_normalizer = gt_classes.numel()

            pred_proposal_deltas = pred_proposal_deltas[fg_inds[:, None], gt_class_cols]
            gt_proposals_delta = gt_proposal_deltas[fg_inds]

            if self.compute_bbox_cov:
                pred_proposal_covs = pred_proposal_covs[
                    fg_inds[:, None], gt_covar_class_cols
                ]
                pred_proposal_covs = clamp_log_variance(pred_proposal_covs)

                if self.bbox_cov_loss == "negative_log_likelihood":
                    if self.bbox_cov_type == "diagonal":
                        # Ger foreground proposals.
                        _proposals_boxes = proposals_boxes.tensor[fg_inds]

                        # Compute regression negative log likelihood loss according to:
                        # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017
                        loss_box_reg = (
                            0.5
                            * torch.exp(-pred_proposal_covs)
                            * smooth_l1_loss(
                                pred_proposal_deltas,
                                gt_proposals_delta,
                                beta=self.smooth_l1_beta,
                            )
                        )
                        loss_covariance_regularize = 0.5 * pred_proposal_covs
                        loss_box_reg += loss_covariance_regularize

                        loss_box_reg = torch.sum(loss_box_reg) / loss_reg_normalizer
                    else:
                        # Multivariate Gaussian Negative Log Likelihood loss using pytorch
                        # distributions.multivariate_normal.log_prob()
                        forecaster_cholesky = covariance_output_to_cholesky(
                            pred_proposal_covs
                        )

                        multivariate_normal_dists = (
                            distributions.multivariate_normal.MultivariateNormal(
                                pred_proposal_deltas, scale_tril=forecaster_cholesky
                            )
                        )

                        loss_box_reg = -multivariate_normal_dists.log_prob(
                            gt_proposals_delta
                        )
                        loss_box_reg = torch.sum(loss_box_reg) / loss_reg_normalizer

                elif self.bbox_cov_loss == "second_moment_matching":
                    # Compute regression covariance using second moment
                    # matching.
                    loss_box_reg = smooth_l1_loss(
                        pred_proposal_deltas, gt_proposals_delta, self.smooth_l1_beta
                    )
                    errors = pred_proposal_deltas - gt_proposals_delta
                    if self.bbox_cov_type == "diagonal":
                        # Handel diagonal case
                        second_moment_matching_term = smooth_l1_loss(
                            torch.exp(pred_proposal_covs),
                            errors ** 2,
                            beta=self.smooth_l1_beta,
                        )
                        loss_box_reg += second_moment_matching_term
                        loss_box_reg = torch.sum(loss_box_reg) / loss_reg_normalizer
                    else:
                        # Handel full covariance case
                        errors = torch.unsqueeze(errors, 2)
                        gt_error_covar = torch.matmul(
                            errors, torch.transpose(errors, 2, 1)
                        )

                        # This is the cholesky decomposition of the covariance matrix.
                        # We reconstruct it from 10 estimated parameters as a
                        # lower triangular matrix.
                        forecaster_cholesky = covariance_output_to_cholesky(
                            pred_proposal_covs
                        )

                        predicted_covar = torch.matmul(
                            forecaster_cholesky,
                            torch.transpose(forecaster_cholesky, 2, 1),
                        )

                        second_moment_matching_term = smooth_l1_loss(
                            predicted_covar,
                            gt_error_covar,
                            beta=self.smooth_l1_beta,
                            reduction="sum",
                        )
                        loss_box_reg = (
                            torch.sum(loss_box_reg) + second_moment_matching_term
                        ) / loss_reg_normalizer

                elif self.bbox_cov_loss == "energy_loss":
                    forecaster_cholesky = covariance_output_to_cholesky(
                        pred_proposal_covs
                    )

                    # Define per-anchor Distributions
                    multivariate_normal_dists = (
                        distributions.multivariate_normal.MultivariateNormal(
                            pred_proposal_deltas, scale_tril=forecaster_cholesky
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
                    loss_covariance_regularize = (
                        -smooth_l1_loss(
                            distributions_samples_1,
                            distributions_samples_2,
                            beta=self.smooth_l1_beta,
                            reduction="sum",
                        )
                        / self.bbox_cov_num_samples
                    )  # Second term

                    gt_proposals_delta_samples = torch.repeat_interleave(
                        gt_proposals_delta.unsqueeze(0),
                        self.bbox_cov_num_samples,
                        dim=0,
                    )

                    loss_first_moment_match = (
                        2.0
                        * smooth_l1_loss(
                            distributions_samples_1,
                            gt_proposals_delta_samples,
                            beta=self.smooth_l1_beta,
                            reduction="sum",
                        )
                        / self.bbox_cov_num_samples
                    )  # First term

                    # Final Loss
                    loss_box_reg = (
                        loss_first_moment_match + loss_covariance_regularize
                    ) / loss_reg_normalizer

                elif self.bbox_cov_loss == "pmb_negative_log_likelihood":
                    losses = self.nll_od_loss_with_nms(
                        predictions, proposals, gt_instances
                    )

                    loss_box_reg = losses["loss_box_reg"]
                    use_nll_loss = True

                else:
                    raise ValueError(
                        "Invalid regression loss name {}.".format(self.bbox_cov_loss)
                    )

                # Perform loss annealing. Not really essential in Generalized-RCNN case, but good practice for more
                # elaborate regression variance losses.
                standard_regression_loss = smooth_l1_loss(
                    pred_proposal_deltas,
                    gt_proposals_delta,
                    self.smooth_l1_beta,
                    reduction="sum",
                )
                standard_regression_loss = (
                    standard_regression_loss / loss_reg_normalizer
                )

                probabilistic_loss_weight = get_probabilistic_loss_weight(
                    current_step, self.annealing_step
                )

                loss_box_reg = (
                    (1.0 - probabilistic_loss_weight) * standard_regression_loss
                    + probabilistic_loss_weight * loss_box_reg
                )

                if use_nll_loss:
                    loss_cls = (1.0 - probabilistic_loss_weight) * loss_cls
            else:
                loss_box_reg = smooth_l1_loss(
                    pred_proposal_deltas,
                    gt_proposals_delta,
                    self.smooth_l1_beta,
                    reduction="sum",
                )
                loss_box_reg = loss_box_reg / loss_reg_normalizer

        if use_nll_loss:
            losses["loss_cls"] = loss_cls
            losses["loss_box_reg"] = loss_box_reg
        else:
            losses = {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

        return losses

    def nll_od_loss_with_nms(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        proposals: List[Instances],
        gt_instances,
    ):
        if "log_prob" in self.matching_distance and self.matching_distance != "log_prob":
            covar_scaling = float(self.matching_distance.split("_")[-1])
            matching_distance = "log_prob"
        else:
            covar_scaling = 1
            matching_distance = self.matching_distance

        self.ppp_intensity_function.update_distribution()
        _, pred_deltas, _, pred_covs = predictions
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        scores = [score.clamp(1e-6, 1 - 1e-6) for score in scores]
        _, num_classes = scores[0].shape
        num_classes -= 1  # do not count background class
        image_shapes = [x.image_size for x in proposals]
        num_prop_per_image = [len(p) for p in proposals]

        # Apply NMS without score threshold
        instances, kept_idx = fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            0.0,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

        kept_idx = [k.unique() for k in kept_idx]
        pred_covs = pred_covs.split(num_prop_per_image)
        pred_deltas = pred_deltas.split(num_prop_per_image)

        kept_proposals = [
            prop.proposal_boxes.tensor[idx] for prop, idx in zip(proposals, kept_idx)
        ]

        pred_covs = [pred_cov[kept] for pred_cov, kept in zip(pred_covs, kept_idx)]
        nll_pred_cov = [
            covariance_output_to_cholesky(clamp_log_variance(reshape_box_preds(cov, num_classes)))
            for cov in pred_covs
        ]
        nll_scores = [score[kept] for score, kept in zip(scores, kept_idx)]
        nll_pred_deltas = [
            reshape_box_preds(delta[kept], num_classes)
            for delta, kept in zip(pred_deltas, kept_idx)
        ]
        trans_func = lambda x,y: self.box2box_transform.apply_deltas(x,y)
        box_means = []
        box_chols = []
        bs = len(nll_pred_deltas)
        for i in range(bs):
            box_mean, box_chol = unscented_transform(nll_pred_deltas[i], nll_pred_cov[i], kept_proposals[i], trans_func)
            box_means.append(box_mean)
            box_chols.append(box_chol)
        
        nll_gt_classes = [instances.gt_classes for instances in gt_instances]
        gt_boxes = [instances.gt_boxes.tensor for instances in gt_instances]

        if self.bbox_cov_dist_type == "gaussian":
            regression_dist = (
                lambda x, y: distributions.multivariate_normal.MultivariateNormal(
                    loc=x, scale_tril=y
                )
            )
        elif self.bbox_cov_dist_type == "laplacian":
            regression_dist = lambda x, y: distributions.laplace.Laplace(
                loc=x, scale=y.diagonal(dim1=-2, dim2=-1) / np.sqrt(2)
            )
        else:
            raise Exception(
                f"Bounding box uncertainty distribution {self.bbox_cov_dist_type} is not available."
            )

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
                selected_chols = pred_box_chols[ppp_preds_idx, 0]
                mixture_dict["covs"] = selected_chols@(selected_chols.transpose(-1,-2))
                mixture_dict["cls_probs"] = pred_cls_probs[ppp_preds_idx, :self.num_classes]
                mixture_dict["reg_dist_type"] = self.bbox_cov_dist_type

                if self.bbox_cov_dist_type == "gaussian":
                    mixture_dict[
                        "reg_dist"
                    ] = distributions.multivariate_normal.MultivariateNormal
                    mixture_dict["reg_kwargs"] = {
                        "scale_tril": selected_chols
                    }
                elif self.bbox_cov_dist_type == "laplacian":
                    mixture_dict["reg_dist"] = distributions.laplace.Laplace
                    mixture_dict["reg_kwargs"] = {
                        "scale": (
                            selected_chols.diagonal(dim1=-2, dim2=-1)
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
            scores_have_bg_cls=True,
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
                torch.as_tensor(image_shapes).to(device), num_classes
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

    def inference(self, predictions, proposals):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device),
                gt_classes,
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas, _, _ = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        scores, _, _, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        if self.cls_var_loss == "evidential":
            alphas = get_dir_alphas(scores)
            dirichlet_s = alphas.sum(1).unsqueeze(1)
            # Compute probabilities
            probs = alphas / dirichlet_s
        else:
            probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)


# Todo: new detectron interface required copying code. Check for better
# way to inherit from FastRCNNConvFCHead.
@ROI_BOX_HEAD_REGISTRY.register()
class DropoutFastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu) and dropout.
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        conv_dims: List[int],
        fc_dims: List[int],
        conv_norm="",
        dropout_rate,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
            dropout_rate (float): p for dropout layer
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self.dropout_rate = dropout_rate
        self.use_dropout = self.dropout_rate != 0.0

        self._output_size = (
            input_shape.channels,
            input_shape.height,
            input_shape.width,
        )

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        self.fcs_dropout = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), fc_dim)
            fc_dropout = nn.Dropout(p=self.dropout_rate)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_dropout{}".format(k + 1), fc_dropout)
            self.fcs.append(fc)
            self.fcs_dropout.append(fc_dropout)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
            "dropout_rate": cfg.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE,
        }

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer, dropout in zip(self.fcs, self.fcs_dropout):
                x = F.relu(dropout(layer(x)))
        return x

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])
