# Semantic Segmentation Pipeline

* [Introduction](#introduction)
* [Datasets, Models, Features](#features)
* [Models Comparison](#models-comparison)
* [Configuration](#configuration)
* [Training](#training)
* [Evaluation](#evaluation)
* [Inference](#inference)

## Introduction

There are a lot of great repositories for semantic segmentation models but most of them are quite complicated if you want to modify or only need necessary parts. 

In this repository, a complete training, evaluation and inference pipeline is written for the purpose of easy to understand and modify. 

If you want to use a custom model, custom dataset and other training configurations like optimizers, schedulers, etc., you can modify easily after taking a quick look at the codes.

## Features

Datasets
* [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) 
* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
* [CityScapes](https://www.cityscapes-dataset.com/) 
* [COCO-Stuff](https://github.com/nightrome/cocostuff) (Coming Soon)
* [MHP](https://lv-mhp.github.io/) (Coming Soon)
* [Pascal Context](https://cs.stanford.edu/~roozbeh/pascal-context/) (Coming Soon)

Models
* [SegFormer](https://arxiv.org/abs/2105.15203v2)
* [VOLO](https://arxiv.org/abs/2106.13112v1) (Coming Soon)
* [MaskFormer](https://arxiv.org/abs/2107.06278v1) (Coming Soon)
* [CSWin Transformer](https://arxiv.org/abs/2107.00652v1) (Coming Soon)
    
PyTorch Features:
* [Native TorchVision I/O](https://pytorch.org/vision/stable/io.html)
* [Native AMP](https://pytorch.org/docs/stable/notes/amp_examples.html)
* [Native DDP](https://pytorch.org/docs/stable/notes/ddp.html) (Coming Soon)


## Models Comparison

Model | CityScapes-val (mIoU) | ADE20K val (mIoU) | Params (M) | FLOPs-CityScapes (G) | FLOPs-ADE20K (G) | Weights
--- | --- | --- | --- | --- | --- | ---
SegFormer-B0 | 78.1 | 38.0 | 3.8 | 125.5 | 8.4 | [pretrained](https://drive.google.com/drive/folders/1UUMCAA51zkBBGAGh9zAX79q0YzHaa0p8?usp=sharing) / [backbone](https://drive.google.com/drive/folders/1WApNd_0T5heB5H5zhvwu6HiI1amvA8Hv?usp=sharing)
SegFormer-B1 | 80.0 | 43.1 | 13.7 | 243.7 | 15.9 | [pretrained](https://drive.google.com/drive/folders/1UUMCAA51zkBBGAGh9zAX79q0YzHaa0p8?usp=sharing) / [backbone](https://drive.google.com/drive/folders/1WApNd_0T5heB5H5zhvwu6HiI1amvA8Hv?usp=sharing)
SegFormer-B2 | 82.2 | 47.5 | 27.5 | 717.1 | 62.4 | [pretrained](https://drive.google.com/drive/folders/1UUMCAA51zkBBGAGh9zAX79q0YzHaa0p8?usp=sharing) / [backbone](https://drive.google.com/drive/folders/1WApNd_0T5heB5H5zhvwu6HiI1amvA8Hv?usp=sharing)
SegFormer-B3 | 83.3 | 50.0 | 47.3 | 962.9 | 79.0 | [pretrained](https://drive.google.com/drive/folders/1UUMCAA51zkBBGAGh9zAX79q0YzHaa0p8?usp=sharing) / [backbone](https://drive.google.com/drive/folders/1WApNd_0T5heB5H5zhvwu6HiI1amvA8Hv?usp=sharing)
SegFormer-B4 | 83.9 | 51.1 | 64.1 | 1240.6 | 95.7 | [pretrained](https://drive.google.com/drive/folders/1UUMCAA51zkBBGAGh9zAX79q0YzHaa0p8?usp=sharing) / [backbone](https://drive.google.com/drive/folders/1WApNd_0T5heB5H5zhvwu6HiI1amvA8Hv?usp=sharing)
SegFormer-B5 | 84.0 | 51.8 | 84.7 | 1460.4 | 183.3 | [pretrained](https://drive.google.com/drive/folders/1UUMCAA51zkBBGAGh9zAX79q0YzHaa0p8?usp=sharing) / [backbone](https://drive.google.com/drive/folders/1WApNd_0T5heB5H5zhvwu6HiI1amvA8Hv?usp=sharing)
VOLO-D1 | 83.1 | 50.5 | - | - | - | N/A
VOLO-D3 | - | 52.9 | - | - | - | N/A
VOLO-D4 | 84.3 | - | - | - | - | N/A
VOLO-D5 | - | 54.3 | - | - | - | N/A
MaskFormer-Swin-T | - | 48.8 | 42 | - | 55 | N/A
MaskFormer-Swin-S | - | 51.0 | 63 | - | 79 | N/A
MaskFormer-Swin-B | - | 53.9 | 102 | - | 195 | N/A
MaskFormer-Swin-L | - | 55.6 | 212 | - | 375 | N/A
CSWin-T | - | 50.4 | 59.9 | - | - | N/A
CSWin-S | - | 50.8 | 64.4 | - | - | N/A
CSWin-B | - | 51.7 | 109.2 | - | - | N/A
CSWin-L^ | - | 55.2 | 207.7 | - | - | N/A

> Notes: All models' results are from papers with multi-scale inference. Pretrained models are converted from official implementations. `^` means the model is pretrained on ImageNet-21K.

Panoptic Segmentation

Model | COCO Panoptic (PQ) | ADE20K Panoptic (PQ) | Params (M) | FLOPs-COCO (G) | FLOPs-ADE20K (G) | Weights
--- | --- | --- | --- | --- | --- | ---
MaskFormer-Swin-T | 47.7 | - | 42 | 179 | - | N/A
MaskFormer-Swin-S | 49.7 | - | 63 | 259 | - | N/A
MaskFormer-Swin-B | 51.8 | - | 102 | 411 | - | N/A
MaskFormer-Swin-L | 52.7 | - | 212 | 792 | - | N/A

## Configuration 

Create a configuration file in `configs`. Sample configuration for CamVid dataset can be found [here](configs/camvid.yaml). Then edit the fields you think if it is needed. This configuration file is needed for all of training, evaluation and inference scripts.

## Training

```bash
$ python tools/train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

## Evaluation

```bash
$ python tools/eval.py --cfg configs/CONFIG_FILE_NAME.yaml
```

## Inference

```bash
$ python tools/infer.py --cfg configs/CONFIG_FILE_NAME.yaml
```

