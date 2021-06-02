# FastSeg

* [Introduction](##Introduction)
* [Datasets, Models, Features](##Features)
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
* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
* [CityScapes](https://www.cityscapes-dataset.com/) (Coming Soon)
* [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) (Coming Soon)

Models
* CNN
    * [BiSeNetv1](https://arxiv.org/abs/1808.00897)
    * [BiSeNetv2](https://arxiv.org/pdf/2004.02147.pdf)
    * [HRNet](https://arxiv.org/abs/1908.07919) (Coming Soon)
    * [Lite-HRNet](https://arxiv.org/abs/2104.06403) (Coming Soon)
* Transformer
    * [Swin Transformer](https://arxiv.org/pdf/2103.14030v1.pdf) (Coming Soon)
    * [Segmenter](https://arxiv.org/pdf/2105.05633v1.pdf) (Coming Soon)
    * [Segmentation Transformer (Updated OCRNet)](https://arxiv.org/pdf/1909.11065.pdf) (Coming Soon)

Features coming soon:
* [Native DDP](https://pytorch.org/docs/stable/notes/ddp.html)
* [Native AMP](https://pytorch.org/docs/stable/notes/amp_examples.html)


## Configuration 

Create a configuration file in `configs`. Sample configuration for CamVid dataset can be found [here](configs/camvid.yaml). Then edit the fields you think if it is needed. This configuration file is needed for all of training, evaluation and prediction scripts.

## Training

```bash
$ python train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

## Evaluation

```bash
$ python eval.py --cfg configs/CONFIG_FILE_NAME.yaml
```

## Inference

```bash
$ python predict.py --cfg configs/CONFIG_FILE_NAME.yaml
```

