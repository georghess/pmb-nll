import os

import numpy as np

# Project imports
from core.setup import setup_arg_parser, setup_config

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch
from probabilistic_inference.inference_utils import get_inference_output_dir

# Coco evaluator tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main(args, cfg=None, inference_output_dir="", image_ids=[]):
    # Setup config
    if cfg is None:
        cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

    # Build path to inference output
    if inference_output_dir == "":
        inference_output_dir = get_inference_output_dir(
            cfg["OUTPUT_DIR"],
            args.test_dataset,
            args.inference_config,
            args.image_corruption_level,
        )

    prediction_file_name = os.path.join(
        inference_output_dir, "coco_instances_results.json"
    )

    meta_catalog = MetadataCatalog.get(args.test_dataset)

    # Evaluate detection results
    gt_coco_api = COCO(meta_catalog.json_file)
    if len(image_ids):
        gt_coco_api.anns = {
            ann_key: ann_val
            for ann_key, ann_val in gt_coco_api.anns.items()
            if ann_val["image_id"] in image_ids
        }
        gt_coco_api.catToImgs = {
            cat: [id for id in img_ids if id in image_ids]
            for cat, img_ids in gt_coco_api.catToImgs.items()
            if len([id for id in img_ids if id in image_ids])
        }
        gt_coco_api.imgToAnns = {
            id: ann for id, ann in gt_coco_api.imgToAnns.items() if id in image_ids
        }
        gt_coco_api.imgs = {
            id: info for id, info in gt_coco_api.imgs.items() if id in image_ids
        }

    res_coco_api = gt_coco_api.loadRes(prediction_file_name)
    results_api = COCOeval(gt_coco_api, res_coco_api, iouType="bbox")

    results_api.params.catIds = list(
        meta_catalog.thing_dataset_id_to_contiguous_id.keys()
    )

    # Calculate and print aggregate results
    results_api.evaluate()
    results_api.accumulate()
    results_api.summarize()

    # Compute optimal micro F1 score threshold. We compute the f1 score for
    # every class and score threshold. We then compute the score threshold that
    # maximizes the F-1 score of every class. The final score threshold is the average
    # over all classes.
    precisions = results_api.eval["precision"].mean(0)[:, :, 0, 2]
    recalls = np.expand_dims(results_api.params.recThrs, 1)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_f1_score = f1_scores.argmax(0)
    scores = results_api.eval["scores"].mean(0)[:, :, 0, 2]
    optimal_score_threshold = [
        scores[optimal_f1_score_i, i]
        for i, optimal_f1_score_i in enumerate(optimal_f1_score)
    ]
    optimal_score_threshold = np.array(optimal_score_threshold)
    optimal_score_threshold = optimal_score_threshold[optimal_score_threshold != 0]
    optimal_score_threshold = optimal_score_threshold.mean()

    print(
        "Classification Score at Optimal F-1 Score: {}".format(optimal_score_threshold)
    )

    text_file_name = os.path.join(inference_output_dir, "mAP_res.txt")

    with open(text_file_name, "w") as text_file:
        print(
            results_api.stats.tolist()
            + [
                optimal_score_threshold,
            ],
            file=text_file,
        )


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
