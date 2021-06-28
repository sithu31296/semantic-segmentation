# Semantic Segmentation Pipeline

* [Introduction](##Introduction)
* [Datasets, Models, Features](##Features)
* [Models Comparison](##Models-Comparison)
* [Configuration](##Configuration)
* [Training](##Training)
* [Evaluation](##Evaluation)
* [Inference](##Inference)

## Introduction

There are a lot of great repositories for semantic segmentation models but most of them are quite complicated if you want to modify or only need necessary parts. 

In this repository, a complete training, evalaution and inference pipeline is written for the purpose of easy to understand and modify. 

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
    
PyTorch Features:
* [Native TorchVision I/O](https://pytorch.org/vision/stable/io.html)
* [Native AMP](https://pytorch.org/docs/stable/notes/amp_examples.html)
* [Native DDP](https://pytorch.org/docs/stable/notes/ddp.html) (Coming Soon)


## Models Comparison

Model | CityScapes-val (mIoU) | ADE20K val (mIoU) | Params (M) | Weights
--- | --- | --- | --- | ---
SegFormer-B0 | 78.1 | 38.0 | 3.8 | [pretrained](https://drive.google.com/drive/folders/1UUMCAA51zkBBGAGh9zAX79q0YzHaa0p8?usp=sharing) / [backbone](https://drive.google.com/drive/folders/1WApNd_0T5heB5H5zhvwu6HiI1amvA8Hv?usp=sharing)
SegFormer-B1 | 80.0 | 43.1 | 13.7 | [pretrained](https://drive.google.com/drive/folders/1UUMCAA51zkBBGAGh9zAX79q0YzHaa0p8?usp=sharing) / [backbone](https://drive.google.com/drive/folders/1WApNd_0T5heB5H5zhvwu6HiI1amvA8Hv?usp=sharing)
SegFormer-B2 | 82.2 | 47.5 | 27.5 | [pretrained](https://drive.google.com/drive/folders/1UUMCAA51zkBBGAGh9zAX79q0YzHaa0p8?usp=sharing) / [backbone](https://drive.google.com/drive/folders/1WApNd_0T5heB5H5zhvwu6HiI1amvA8Hv?usp=sharing)
SegFormer-B3 | 83.3 | 50.0 | 47.3 | [pretrained](https://drive.google.com/drive/folders/1UUMCAA51zkBBGAGh9zAX79q0YzHaa0p8?usp=sharing) / [backbone](https://drive.google.com/drive/folders/1WApNd_0T5heB5H5zhvwu6HiI1amvA8Hv?usp=sharing)
SegFormer-B4 | 83.9 | 51.1 | 64.1 | [pretrained](https://drive.google.com/drive/folders/1UUMCAA51zkBBGAGh9zAX79q0YzHaa0p8?usp=sharing) / [backbone](https://drive.google.com/drive/folders/1WApNd_0T5heB5H5zhvwu6HiI1amvA8Hv?usp=sharing)
SegFormer-B5 | 84.0 | 51.8 | 84.7 | [pretrained](https://drive.google.com/drive/folders/1UUMCAA51zkBBGAGh9zAX79q0YzHaa0p8?usp=sharing) / [backbone](https://drive.google.com/drive/folders/1WApNd_0T5heB5H5zhvwu6HiI1amvA8Hv?usp=sharing)
VOLO-D1 | 83.1 | 50.5 | - | N/A
VOLO-D3 | - | 52.9 | - | N/A
VOLO-D4 | 84.3 | - | - | N/A
VOLO-D5 | - | 54.3 | - | N/A

> Notes: All models' results are from papers with multi-scale inference. Pretrained models are converted from official implementations.

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

