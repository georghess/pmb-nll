"""
Probabilistic Detectron Single Image Inference Script
Runs inference and evaluation on specified images, rather than on entire dataset.
"""
import json
import os
import sys
from shutil import copyfile, rmtree

import torch
import tqdm

import core

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), "src", "detr"))

from detectron2.data import MetadataCatalog, build_detection_test_loader
# Detectron imports
from detectron2.engine import launch

from core.evaluation_tools import evaluation_utils
# Project imports
from core.evaluation_tools.evaluation_utils import \
    get_train_contiguous_id_to_test_thing_dataset_id_dict
from core.setup import setup_arg_parser, setup_config
from offline_evaluation import (compute_average_precision,
                                compute_calibration_errors,
                                compute_ood_probabilistic_metrics,
                                compute_probabilistic_metrics)
from probabilistic_inference.inference_utils import (build_predictor,
                                                     get_inference_output_dir,
                                                     instances_to_json)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # Setup config
    cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)
    # Make sure only 1 data point is processed at a time. This simulates
    # deployment.
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 32
    cfg.SOLVER.IMS_PER_BATCH = 1

    cfg.MODEL.DEVICE = device.type

    # Set up number of cpu threads#
    torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    # Create inference output directory and copy inference config file to keep
    # track of experimental settings
    if args.inference_dir == "":
        inference_output_dir = get_inference_output_dir(
            cfg["OUTPUT_DIR"],
            args.test_dataset,
            args.inference_config,
            args.image_corruption_level,
        )
    else:
        inference_output_dir = args.inference_dir
        if not os.path.isdir(inference_output_dir):
            os.makedirs(inference_output_dir, exist_ok=True)

    os.makedirs(inference_output_dir, exist_ok=True)
    copyfile(
        args.inference_config,
        os.path.join(inference_output_dir, os.path.split(args.inference_config)[-1]),
    )

    # Get category mapping dictionary:
    train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        cfg.DATASETS.TRAIN[0]
    ).thing_dataset_id_to_contiguous_id
    test_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        args.test_dataset
    ).thing_dataset_id_to_contiguous_id

    # If both dicts are equal or if we are performing out of distribution
    # detection, just flip the test dict.
    cat_mapping_dict = get_train_contiguous_id_to_test_thing_dataset_id_dict(
        cfg,
        args,
        train_thing_dataset_id_to_contiguous_id,
        test_thing_dataset_id_to_contiguous_id,
    )

    # Build predictor
    predictor = build_predictor(cfg)
    test_data_loader = build_detection_test_loader(cfg, dataset_name=args.test_dataset)

    # Prepare GT annos
    cfg.defrost()
    cfg.ACTUAL_TEST_DATASET = args.test_dataset
    preprocessed_gt_instances = (
        evaluation_utils.get_per_frame_preprocessed_gt_instances(
            cfg, inference_output_dir
        )
    )

    final_output_list = []
    # Example for image_ids to visualize, set to empty list for all images in dataset
    image_ids = [2153,2261,6894,10764,17905,23272]

    with torch.no_grad():
        with tqdm.tqdm(total=len(test_data_loader)) as pbar:
            for idx, input_im in enumerate(test_data_loader):
                image_id = input_im[0]["image_id"]
                if len(image_ids) and image_id not in image_ids:
                    pbar.update(1)
                    continue
                
                if not args.eval_only:
                    # Apply corruption
                    outputs = predictor(input_im)
                    json_instances = instances_to_json(
                        outputs, image_id, cat_mapping_dict
                    )
                    final_output_list.extend(json_instances)
                    # Save instances for this prediction only to temporary dir
                    tmp_inference_dir = os.path.join(inference_output_dir, "tmp")
                    rmtree(tmp_inference_dir, ignore_errors=True)
                    os.makedirs(tmp_inference_dir, exist_ok=True)
                    with open(
                        os.path.join(tmp_inference_dir, "coco_instances_results.json"),
                        "w",
                    ) as fp:
                        json.dump(json_instances, fp, indent=4, separators=(",", ": "))
                    # Load in standard evaluation format
                    preprocessed_predicted_instances = (
                        evaluation_utils.eval_predictions_preprocess(json_instances)
                    )
                else:
                    tmp_inference_dir = inference_output_dir
                    outputs = (
                        evaluation_utils.get_per_frame_preprocessed_pred_instances(
                            cfg, tmp_inference_dir, image_id, 0.0
                        )
                    )

                preprocessed_gt_instance = {}
                for k, v in preprocessed_gt_instances.items():
                    for img_id, t in v.items():
                        if img_id == image_id:
                            preprocessed_gt_instance[k] = t
                if len(preprocessed_gt_instance) == 0:
                    preprocessed_gt_instance = None
                
                class_map = MetadataCatalog[cfg.ACTUAL_TEST_DATASET].get(
                        "thing_classes"
                    )
                gt_class_map = MetadataCatalog[cfg.ACTUAL_TEST_DATASET].thing_dataset_id_to_contiguous_id

                predictor.visualize_inference(
                    input_im,
                    outputs,
                    preprocessed_gt_instance,
                    min_allowed_score=0.1,
                    class_map=class_map,
                    gt_class_map=gt_class_map,
                    num_samples=0,
                )
                # Compute metrics for this prediction only
                compute_average_precision.main(args, cfg, tmp_inference_dir, [image_id])
                compute_probabilistic_metrics.main(
                    args,
                    cfg,
                    inference_output_dir=tmp_inference_dir,
                    image_ids=[image_id],
                    min_allowed_score=0.0,
                )
                pbar.update(1)

    with open(
        os.path.join(inference_output_dir, "coco_instances_results.json"), "w"
    ) as fp:
        json.dump(final_output_list, fp, indent=4, separators=(",", ": "))

    if "ood" in args.test_dataset:
        compute_ood_probabilistic_metrics.main(args, cfg)
    else:
        compute_average_precision.main(args, cfg, inference_output_dir, image_ids)
        compute_probabilistic_metrics.main(
            args, cfg, inference_output_dir=inference_output_dir, image_ids=image_ids
        )
        compute_calibration_errors.main(
            args, cfg, inference_output_dir=inference_output_dir
        )


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    # Support single gpu inference only.
    args.num_gpus = 1
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
