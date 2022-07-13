import cv2
import os

from abc import ABC, abstractmethod

# Detectron Imports
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from core.visualization_tools.probabilistic_visualizer import ProbabilisticVisualizer

# Project Imports
from probabilistic_inference import inference_utils


class ProbabilisticPredictor(ABC):
    """
    Abstract class for probabilistic predictor.
    """

    def __init__(self, cfg):
        # Create common attributes.
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model_list = []

        # Parse config
        self.inference_mode = self.cfg.PROBABILISTIC_INFERENCE.INFERENCE_MODE
        self.mc_dropout_enabled = self.cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.ENABLE
        self.num_mc_dropout_runs = self.cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS
        self.use_mc_sampling = cfg.PROBABILISTIC_INFERENCE.USE_MC_SAMPLING

        # Set model to train for MC-Dropout runs
        if self.mc_dropout_enabled:
            self.model.train()
        else:
            self.model.eval()

        # Create ensemble if applicable.
        if self.inference_mode == 'ensembles':
            ensemble_random_seeds = self.cfg.PROBABILISTIC_INFERENCE.ENSEMBLES.RANDOM_SEED_NUMS

            for i, random_seed in enumerate(ensemble_random_seeds):
                model = build_model(self.cfg)
                model.eval()

                checkpoint_dir = os.path.join(
                    os.path.split(
                        self.cfg.OUTPUT_DIR)[0],
                    'random_seed_' +
                    str(random_seed))
                # Load last checkpoint.
                DetectionCheckpointer(
                    model,
                    save_dir=checkpoint_dir).resume_or_load(
                    cfg.MODEL.WEIGHTS,
                    resume=True)
                self.model_list.append(model)
        else:
            # Or Load single model last checkpoint.
            DetectionCheckpointer(
                self.model,
                save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS,
                resume=True)

    def __call__(self, input_im):
        # Generate detector output.
        if self.inference_mode == 'standard_nms':
            results = self.post_processing_standard_nms(input_im)
        elif self.inference_mode == 'mc_dropout_ensembles':
            results = self.post_processing_mc_dropout_ensembles(
                input_im)
        elif self.inference_mode == 'output_statistics':
            results = self.post_processing_output_statistics(
                input_im)
        elif self.inference_mode == 'ensembles':
            results = self.post_processing_ensembles(input_im, self.model_list)
        elif self.inference_mode == 'bayes_od':
            results = self.post_processing_bayes_od(input_im)
        elif self.inference_mode == 'topk_detections':
            results = self.post_processing_topk_detections(input_im)
        else:
            raise ValueError(
                'Invalid inference mode {}.'.format(
                    self.inference_mode))

        # Perform post processing on detector output.
        height = input_im[0].get("height", results.image_size[0])
        width = input_im[0].get("width", results.image_size[1])
        results = inference_utils.probabilistic_detector_postprocess(results,
                                                                     height,
                                                                     width)
        return results

    def visualize_inference(
        self,
        inputs,
        results,
        gt=None,
        min_allowed_score=-1,
        class_map=None,
        gt_class_map=None,
        num_samples=0,
    ):
        """
        A function used to visualize final network predictions.
        It shows the original image and up to 20
        predicted object bounding boxes on the original image.

        Valuable for debugging inference methods.

        Args:
            inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        max_boxes = 100

        required_width = inputs[0]["width"]
        required_height = inputs[0]["height"]

        img = inputs[0]["image"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        if self.model.input_format == "RGB":
            img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)
        img = cv2.resize(img, (required_width, required_height))

        predicted_boxes = results.pred_boxes.tensor.cpu().numpy()
        predicted_covar_mats = results.pred_boxes_covariance.cpu().numpy()
        scores = results.scores.cpu().numpy()
        #scores[0] = 0.75
        if class_map:
            labels = np.array(
                [
                    f"{class_map[cls]}: {round(score, 2)}"
                    for score, cls in zip(
                        scores.tolist(), results.pred_classes.numpy().tolist()
                    )
                ]
            )
        else:
            labels = np.array([f"{s:.2f}" for s in scores])

        if gt is not None:
            gt_boxes = gt["gt_boxes"].cpu().numpy()
            gt_labels = [class_map[gt_class_map[int(cls.squeeze())]] if class_map and gt_class_map else int(cls.squeeze()) for cls in gt["gt_cat_idxs"].cpu().numpy()]
            v_gt = ProbabilisticVisualizer(img, None)
            v_img = v_gt.overlay_instances(boxes=gt_boxes, labels=gt_labels, assigned_colors=["g"]*len(gt_labels))
            gt_img = v_img.get_image()
            gt_vis_name = f"GT. Image id {inputs[0]['image_id']}"
            cv2.imshow(gt_vis_name, gt_img)
        else:
            v_gt = None

        v_pred = ProbabilisticVisualizer(img, None) if v_gt is None else v_gt
        alpha = 0.5
        assinged_colors = ["red"]* len(predicted_boxes[scores > min_allowed_score][0:max_boxes])
        assinged_colors = None
        """v_pred.overlay_covariance_instances(
            boxes=predicted_boxes[scores < min_allowed_score][0:max_boxes],
            covariance_matrices=predicted_covar_mats[scores < min_allowed_score][
                0:max_boxes
            ],
            labels=labels[scores < min_allowed_score][0:max_boxes],
            assigned_colors=assinged_colors,
            alpha=0.05,
        )"""

        v_pred = v_pred.overlay_covariance_instances(
            boxes=predicted_boxes[scores > min_allowed_score][0:max_boxes],
            covariance_matrices=predicted_covar_mats[scores > min_allowed_score][
                0:max_boxes
            ],
            labels=labels[scores > min_allowed_score][0:max_boxes],
            assigned_colors=assinged_colors,
            alpha=0.8
        )
        
        prop_img = v_pred.get_image()
        vis_name = (
            f"{max_boxes} Highest Scoring Results. Image id {inputs[0]['image_id']}"
        )
        cv2.imshow(vis_name, prop_img)

        if num_samples > 0:
            for i in range(num_samples):
                sampled_boxes = []
                means = predicted_boxes[scores > min_allowed_score]
                covs = predicted_covar_mats[scores > min_allowed_score]
                ss = scores[scores > min_allowed_score]
                for j in range(len(means)):
                    if ss[j] < 0.1:
                        n = np.random.poisson(scores[j])
                    else:
                        n = 1 if ss[j] > np.random.rand() else 0

                    for _ in range(n):
                        sampled_box = np.random.multivariate_normal(
                            mean=means[j],
                            cov=covs[j],
                        )
                        sampled_boxes.append(sampled_box)

                sampled_boxes = np.array(sampled_boxes)
                
                v_pred_sample = ProbabilisticVisualizer(img, None)

                v_pred_sample = v_pred_sample.overlay_instances(
                    boxes=sampled_boxes,
                    assigned_colors=["red"] * len(sampled_boxes),
                    alpha=1.0,
                )

                prop_img = v_pred_sample.get_image()
                vis_name = f"sample_{i}_image_id_{inputs[0]['image_id']}.png"
                cv2.imwrite(vis_name, prop_img)

        

        cv2.waitKey()

    @abstractmethod
    def post_processing_standard_nms(self, input_im):
        pass

    @abstractmethod
    def post_processing_output_statistics(self, input_im):
        pass

    @abstractmethod
    def post_processing_mc_dropout_ensembles(self, input_im):
        pass

    @abstractmethod
    def post_processing_ensembles(self, input_im, model_list):
        pass

    @abstractmethod
    def post_processing_bayes_od(self, input_im):
        pass

    @abstractmethod
    def post_processing_topk_detections(self, input_im):
        pass