# Object Detection as Probabilistic Set Prediction
This is the official implementation of [Object Detection as Probabilistic Set Prediction](https://arxiv.org/abs/2203.07980).

The code builds upon the [probabilisitc detectron2](https://github.com/asharakeh/probdet) repository. 

## Disclaimer
This research code was produced by one person with a single set of eyes, it may contain bugs and errors that I did not notice by the time of release.


## Todo's
- [X] Release code
- [ ] Create nice API for evaluating arbitrary model with PMB-NLL
- [ ] Create nice API for use MB-NLL as loss function with any object detector

## Requirements
#### Software Support:
Name | Supported Versions
--- | --- |
Ubuntu |20.04
Python |3.8
CUDA |11.0+
Cudnn |8.0.1+
PyTorch |1.8+

To install requirements we provide a docker image using the provided Dockerfile.

Docker Image
```
cd Docker

# Build docker image
sh build.sh 
```

You also need to build the [fastmurty](https://github.com/motrom/fastmurty) package used for solving optimal assignment problem.
```
cd src/core/fastmurty
make
```

## Datasets

### COCO Dataset
Download the COCO Object Detection Dataset [here](https://cocodataset.org/#home). 
The COCO dataset folder should have the following structure:
<br>

     └── COCO_DATASET_ROOT
         |
         ├── annotations
         ├── train2017
         └── val2017


## Training
To train the models in the paper, use this command:

``` train
python src/train_net.py
--num-gpus xx
--dataset-dir COCO_DATASET_ROOT
--config-file COCO-Detection/architecture_name/config_name.yaml
--resume
```

For an explanation of all command line arguments, use ```python src/train_net.py -h```

## Evaluation
To run model inference after training, use this command:
```eval
python src/apply_net.py 
--dataset-dir TEST_DATASET_ROOT 
--test-dataset test_dataset_name 
--config-file path/to/config.yaml 
--inference-config /path/to/inference/config.yaml 
```

For an explanation of all command line arguments, use ```python src/apply_net.py  -h```

`--test-dataset` should be `coco_2017_custom_val`, `--dataset-dir` corresponds to the root directory of the dataset used.
Evaluation code will run inference on the test dataset and then will generate mAP and PMB-NLL. Also, the Negative Log Likelihood, Brier Score, Energy Score, and Calibration Error results following [Estimating and Evaluating Regression Predictive Uncertainty in Deep Object Detectors](https://arxiv.org/abs/2101.05036) are reported. If only evaluation of metrics is required,
add `--eval-only` to the above code snippet.

## Inference on new images
We provide a script to perform inference on new images without passing through dataset handlers.

```
python single_image_inference.py 
--image-dir /path/to/image/dir
--output-dir /path/to/output/dir
--config-file /path/to/config/file 
--inference-config /path/to/inference/config 
--model-ckpt /path/to/model.pth
```

`image-dir` is a folder containing all images to be used for inference. `output-dir` is a folder to write the output 
json file containing probabilistic detections. `model-ckpt` is the path to the model checkpoint to be used for 
inference. Look below to download model checkpoints.

## Configurations in the paper
We provide a list of config combinations that generate the architectures used in our paper, and for pretrained models we provide links to where they can be downloaded.

Method Name | Config File | Inference Config File | Model
--- | --- | --- |---
RetinaNet NLL | retinanet_R_50_FPN_3x_reg_var_nll.yaml | standard_nms.yaml | [retinanet_R_50_FPN_3x_reg_var_nll.pth](https://drive.google.com/file/d/11SghCRPC6R9joJq2aT1qYrUGVb6Xr0RM/view?usp=sharing)
RetinaNet ES | retinanet_R_50_FPN_3x_reg_var_es.yaml | standard_nms.yaml | [retinanet_R_50_FPN_3x_reg_var_es.pth](https://drive.google.com/file/d/1R0WFyeZIabtQ7V0YuUirqcqkWd5WHTB9/view?usp=sharing)
RetinaNet MB-NLL | retinanet_R_50_FPN_3x_reg_var_pmbnll_esbase.yaml | standard_nms.yaml | No pretained weights for blind review
--- | --- | --- | ---
FasterRCNN NLL | faster_rcnn_R_50_FPN_3x_reg_covar_nll.yaml | standard_nms.yaml |[faster_rcnn_R_50_FPN_3x_reg_covar_nll.pth](https://drive.google.com/file/d/1RPvvmcKfG8AZQFyyWJdDBP16il3YaTnd/view?usp=sharing)
FasterRCNN ES | faster_rcnn_R_50_FPN_3x_reg_var_es.yaml | standard_nms.yaml |[faster_rcnn_R_50_FPN_3x_reg_var_es.pth](https://drive.google.com/file/d/1Vm_eBSjl8n1T5JFLaLgXSdAg1bawJ1ky/view?usp=sharing)
FasterRCNN MB-NLL | faster_rcnn_R_50_FPN_3x_reg_var_pmbnll_esbase.yaml | standard_nms.yaml | No pretained weights for blind review
--- | --- | --- | ---
DETR NLL | detr_R_50_reg_var_nll.yaml | topk_detections.yaml | [detr_R_50_reg_var_nll.pth](https://drive.google.com/file/d/1iuk5OIF8UO2jg7PdpCZA1qlxtgQzmv54/view?usp=sharing)
DETR ES| detr_R_50_reg_var_es.yaml | topk_detections.yaml | [detr_R_50_reg_var_es.pth](https://drive.google.com/file/d/1Kgll1Ez0cLo_Wut07LJQef7eGP3xuG6_/view?usp=sharing)
DETR MB-NLL | detr_R_50_reg_var_pmbnll_esbase.yaml | topk_detections.yaml | No pretained weights for blind review

## License
This code is released under the [Apache 2.0 License](LICENSE.md).
