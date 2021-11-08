# <div align="center">Semantic Segmentation</div>

<div align="center">
<p>Easy to use and customizable SOTA Semantic Segmentation models with abundant datasets in PyTorch</p>

<a href="https://colab.research.google.com/github/sithu31296/semantic-segmentation/blob/main/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

</div>

![banner](assests/banner.jpg)

## <div align="center">Features</div>

* Applicable to following tasks:
  * Scene Parsing
  * Human Parsing
  * Face Parsing
* 20+ Datasets
* 10+ SOTA Backbones
* 10+ SOTA Semantic Segmentation Models
* PyTorch, ONNX, TFLite and OpenVINO Inference 

</div>


## <div align="center">Model Zoo</div>

Supported Backbones:
* [ResNet](https://arxiv.org/abs/1512.03385) (CVPR 2016)
* [ResNetD](https://arxiv.org/abs/1812.01187) (ArXiv 2018)
* [MobileNetV2](https://arxiv.org/abs/1801.04381) (CVPR 2018)
* [MobileNetV3](https://arxiv.org/abs/1905.02244) (ICCV 2019)
* [PVTv2](https://arxiv.org/abs/2106.13797) (ArXiv 2021)
* [ResT](https://arxiv.org/abs/2105.13677v3) (ArXiv 2021)
* [MicroNet](https://arxiv.org/abs/2108.05894) (ICCV 2021)

Supported Heads/Methods:
* [FCN](https://arxiv.org/abs/1411.4038) (CVPR 2015)
* [UPerNet](https://arxiv.org/abs/1807.10221) (ECCV 2018)
* [BiSeNetv1](https://arxiv.org/abs/1808.00897) (ECCV 2018)
* [FPN](https://arxiv.org/abs/1901.02446) (CVPR 2019)
* [SFNet](https://arxiv.org/abs/2002.10120) (ECCV 2020)
* [SegFormer](https://arxiv.org/abs/2105.15203v2) (ArXiv 2021)
* [FaPN](https://arxiv.org/abs/2108.07058) (ICCV 2021)
* [CondNet](https://arxiv.org/abs/2109.10322) (IEEE SPL 2021)

Supported Standalone Models:
* [FCHarDNet](https://arxiv.org/abs/1909.00948) (ICCV 2019)
* [BiSeNetv2](https://arxiv.org/abs/2004.02147v1) (IJCV 2021)
* [DDRNet](https://arxiv.org/abs/2101.06085) (ArXiv 2021)

Supported Modules:
* [PPM](https://arxiv.org/abs/1612.01105) (CVPR 2017)
* [PSA](https://arxiv.org/abs/2107.00782) (ArXiv 2021)

<details open>
  <summary><strong>ADE20K-val</strong> (Scene Parsing)</summary>

Method | Backbone | mIoU (%) | Params <br><sup>(M) | GFLOPs <br><sup>(512x512) | Weights
--- | --- | --- | --- | --- | --- 
SegFormer | MiT-B1 | 43.1 | 14 | 16 | [pt](https://drive.google.com/file/d/18PN_P3ajcJi_5Q2v8b4BP9O4VdNCpt6m/view?usp=sharing)
|| MiT-B2 | 47.5 | 28 | 62 | [pt](https://drive.google.com/file/d/1AcgEK5aWMJzpe8tsfauqhragR0nBHyPh/view?usp=sharing) 
|| MiT-B3 | 50.0 | 47 | 79 | [pt](https://drive.google.com/file/d/1-OmW3xRD3WAbJTzktPC-VMOF5WMsN8XT/view?usp=sharing)

</details>

<details open>
  <summary><strong>CityScapes-val</strong> (Scene Parsing)</summary>

Method | Backbone | mIoU (%) | Params (M) | GFLOPs | Img Size | Weights
--- | --- | --- | --- | --- | --- | ---
SegFormer | MiT-B0 | 78.1 | 4 | 126 | 1024x1024 | N/A
|| MiT-B1 | 80.0 | 14 | 244 | 1024x1024 | N/A
FaPN | ResNet-50 | 80.0 | 33 | - | 512x1024 | N/A
SFNet | ResNetD-18 | 79.0 | 13 | - | 1024x1024 | N/A
FCHarDNet | HarDNet-70 | 77.7 | 4 | 35 | 1024x1024 | [pt](https://drive.google.com/file/d/1QK1qgaKOPAatx-DHNmv7Mu0S0cE1fHCN/view?usp=sharing)
DDRNet | DDRNet-23slim | 77.8 | 6 | 36 | 1024x2048 | [pt](https://drive.google.com/file/d/17IMQS23StRcIvMXmjNwEcU9uGXuviGY-/view?usp=sharing)
</details>

<details open>
  <summary><strong>HELEN-val</strong> (Face Parsing)</summary>

Method | Backbone | mIoU (%) | Params <br><sup>(M) | GFLOPs <br><sup>(512x512) | FPS <br><sup>(GTX1660ti) | Weights
--- | --- | --- | --- | --- | --- | --- 
BiSeNetv1 | MobileNetV2-1.0 | 58.22 | 5 | 5 | 160 | [pt](https://drive.google.com/file/d/1gE1Ot0O0nzXNABYdTuThRi-BU3_VHuMy/view?usp=sharing)
BiSeNetv1 | ResNet-18 | 58.50 | 14 | 13 | 263 | [pt](https://drive.google.com/file/d/1HMC6OiFPc-aYwhlHlPYoXa-VCR3r2WPQ/view?usp=sharing)
BiSeNetv2 | - | 58.58 | 18 | 15 | 195 | [pt](https://drive.google.com/file/d/1cf-W_2m-vfxMRZ0mFQjEwhOglURpH7m6/view?usp=sharing)
FCHarDNet | HarDNet-70 | 59.38 | 4 | 4 | 130 | [pt](https://drive.google.com/file/d/1XTzpXairzUlQd3i91jOVmSDLs8Zd33jT/view?usp=sharing)
DDRNet | DDRNet-23slim | 61.11 | 6 | 5 | 180 | [pt](https://drive.google.com/file/d/1SdOgVvgYrp8UFztHWN6dHH0MhP8zqnyh/view?usp=sharing)\|[tflite(fp32)](https://drive.google.com/file/d/13yEnXjuv3hKY1cv1ycrjPUFNVyiJM89T/view?usp=sharing)\|[tflite(fp16)](https://drive.google.com/file/d/1IHdjt9wegKiDFvq9l1wglywvM8U0CFdo/view?usp=sharing)\|[tflite(int8)](https://drive.google.com/file/d/1NwPPzz_65YPcSfCfyXx7DiYeCMJMWKeH/view?usp=sharing)
SegFormer | MiT-B0 | 59.31 | 4 | 8 | 75 | [pt](https://drive.google.com/file/d/1uGRjPgX8wcHJlPalFntZOjG0Toxj0msO/view?usp=sharing)
SFNet | ResNetD-18 | 61.00 | 14 | 31 | 56 | [pt](https://drive.google.com/file/d/13w42DgI4PJ05bkWY9XCK_skSGMsmXroj/view?usp=sharing)
</details>

<details open>
  <summary><strong>Backbones</strong></summary>

Model | Variants | ImageNet-1k Top-1 Acc (%) | Params (M) | GFLOPs | Weights
--- | --- | --- | --- | --- | ---
MicroNet | M1\|M2\|M3 | 51.4`\|`59.4`\|`62.5 | 1`\|`2`\|`3 | 6M`\|`12M`\|`21M | [download](https://drive.google.com/drive/folders/1j4JSTcAh94U2k-7jCl_3nwbNi0eduM2P?usp=sharing)
MobileNetV2 | 1.0 | 71.9 | 3 | 300M | [download](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth)
MobileNetV3 | S\|L | 67.7`\|`74.0 | 3`\|`5 | 56M`\|`219M | [S](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth)\|[L](https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth)
||
ResNet | 18\|50\|101 | 69.8`\|`76.1`\|`77.4 | 12`\|`25`\|`44 | 2`\|`4`\|`8 | [download](https://drive.google.com/drive/folders/1MXP3Qx51c91PL9P52Tv89t90SaiTYuaC?usp=sharing)
ResNetD | 18\|50\|101 | - | 12`\|`25`\|`44 | 2`\|`4`\|`8 | [download](https://drive.google.com/drive/folders/1sVyewBDkePlw3kbvhUD4PvUxjro4iKFy?usp=sharing)
MiT | B1\|B2\|B3 | - | 14`\|`25`\|`45 | 2`\|`4`\|`8 | [download](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia)
PVTv2 | B1\|B2\|B4 | 78.7`\|`82.0`\|`83.6 | 14`\|`25`\|`63 | 2`\|`4`\|`10 | [download](https://drive.google.com/drive/folders/10Dd9BEe4wv71dC5BXhsL_C6KeI_Rcxm3?usp=sharing)
ResT | S\|B\|L | 79.6`\|`81.6`\|`83.6 | 14`\|`30`\|`52 | 2`\|`4`\|`8 | [download](https://drive.google.com/drive/folders/1R2cewgHo6sYcQnRGBBIndjNomumBwekr?usp=sharing)

> Notes: Download backbones' weights for [HarDNet-70](https://drive.google.com/file/d/1HAFHvtodAPL_eb4LX_rb0FJZyKTOo4mK/view?usp=sharing) and [DDRNet-23slim](https://drive.google.com/file/d/1TaDJ3yG8ojjcsbQZwkn5LlFMNEcr8vu2/view?usp=sharing).

</details>

## <div align="center">Supported Datasets</div>

[ade20k]: http://sceneparsing.csail.mit.edu/
[cityscapes]: https://www.cityscapes-dataset.com/
[camvid]: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
[cocostuff]: https://github.com/nightrome/cocostuff
[mhp]: https://lv-mhp.github.io/
[lip]: http://sysu-hcp.net/lip/index.php
[atr]: https://github.com/lemondan/HumanParsing-Dataset
[pascalcontext]: https://cs.stanford.edu/~roozbeh/pascal-context/
[pcannos]: https://drive.google.com/file/d/1hOQnuTVYE9s7iRdo-6iARWkN2-qCAoVz/view?usp=sharing
[suim]: http://irvlab.cs.umn.edu/resources/suim-dataset
[mv]: https://www.mapillary.com/dataset/vistas
[sunrgbd]: https://rgbd.cs.princeton.edu/
[helen]: https://www.sifeiliu.net/face-parsing
[celeba]: https://github.com/switchablenorms/CelebAMask-HQ
[lapa]: https://github.com/JDAI-CV/lapa-dataset
[ibugmask]: https://github.com/hhj1897/face_parsing
[facesynthetics]: https://github.com/microsoft/FaceSynthetics
[ccihp]: https://kalisteo.cea.fr/wp-content/uploads/2021/09/README.html

Dataset | Type | Categories | Train <br><sup>Images | Val<br><sup>Images | Test<br><sup>Images | Image Size<br><sup>(HxW)
--- | --- | --- | --- | --- | --- | ---
[COCO-Stuff][cocostuff] | General Scene Parsing | 171 | 118,000 | 5,000 | 20,000 | -
[ADE20K][ade20k] | General Scene Parsing | 150 | 20,210 | 2,000 | 3,352 | -
[PASCALContext][pascalcontext] | General Scene Parsing | 59 | 4,996 | 5,104 | 9,637 | -
||
[SUN RGB-D][sunrgbd] | Indoor Scene Parsing | 37 | 2,666 | 2,619 | 5,050<sup>+labels | -
||
[Mapillary Vistas][mv] | Street Scene Parsing | 65 | 18,000 | 2,000 | 5,000 | 1080x1920
[CityScapes][cityscapes] | Street Scene Parsing | 19 | 2,975 | 500 | 1,525<sup>+labels | 1024x2048
[CamVid][camvid] | Street Scene Parsing | 11 | 367 | 101 | 233<sup>+labels | 720x960
||
[MHPv2][mhp] | Multi-Human Parsing | 59 | 15,403 | 5,000 | 5,000 | -
[MHPv1][mhp] | Multi-Human Parsing | 19 | 3,000 | 1,000 | 980<sup>+labels | -
[LIP][lip] | Multi-Human Parsing | 20 | 30,462 | 10,000 | - | -
[CCIHP][ccihp] | Multi-Human Parsing | 22 | 28,280 | 5,000 | 5,000 | -
[CIHP][lip] | Multi-Human Parsing | 20 | 28,280 | 5,000 | 5,000 | -
[ATR][atr] | Single-Human Parsing | 18 | 16,000 | 700 | 1,000<sup>+labels | -
||
[HELEN][helen] | Face Parsing | 11 | 2,000 | 230 | 100<sup>+labels | - 
[LaPa][lapa] | Face Parsing | 11 | 18,176 | 2,000 | 2,000<sup>+labels | -
[iBugMask][ibugmask] | Face Parsing | 11 | 21,866 | - | 1,000<sup>+labels | -
[CelebAMaskHQ][celeba] | Face Parsing | 19 | 24,183 | 2,993 | 2,824<sup>+labels | 512x512
[FaceSynthetics][facesynthetics] | Face Parsing (Synthetic) | 19 | 100,000 | 1,000 | 100<sup>+labels | 512x512
||
[SUIM][suim] | Underwater Imagery | 8 | 1,525 | - | 110<sup>+labels | -

Check [DATASETS](./DATASETS.md) to find more segmentation datasets.

<details>
  <summary><strong>Datasets Structure</strong> (click to expand)</summary>

Datasets should have the following structure:

```
data
|__ ADEChallenge
    |__ ADEChallengeData2016
        |__ images
            |__ training
            |__ validation
        |__ annotations
            |__ training
            |__ validation

|__ CityScapes
    |__ leftImg8bit
        |__ train
        |__ val
        |__ test
    |__ gtFine
        |__ train
        |__ val
        |__ test

|__ CamVid
    |__ train
    |__ val
    |__ test
    |__ train_labels
    |__ val_labels
    |__ test_labels
    
|__ VOCdevkit
    |__ VOC2010
        |__ JPEGImages
        |__ SegmentationClassContext
        |__ ImageSets
            |__ SegmentationContext
                |__ train.txt
                |__ val.txt
    
|__ COCO
    |__ images
        |__ train2017
        |__ val2017
    |__ labels
        |__ train2017
        |__ val2017

|__ MHPv1
    |__ images
    |__ annotations
    |__ train_list.txt
    |__ test_list.txt

|__ MHPv2
    |__ train
        |__ images
        |__ parsing_annos
    |__ val
        |__ images
        |__ parsing_annos

|__ LIP
    |__ LIP
        |__ TrainVal_images
            |__ train_images
            |__ val_images
        |__ TrainVal_parsing_annotations
            |__ train_segmentations
            |__ val_segmentations

    |__ CIHP/CCIHP
        |__ instance-leve_human_parsing
            |__ Training
                |__ Images
                |__ Category_ids
            |__ Validation
                |__ Images
                |__ Category_ids

    |__ ATR
        |__ humanparsing
            |__ JPEGImages
            |__ SegmentationClassAug

|__ SUIM
    |__ train_val
        |__ images
        |__ masks
    |__ TEST
        |__ images
        |__ masks

|__ SunRGBD
    |__ SUNRGBD
        |__ kv1/kv2/realsense/xtion
    |__ SUNRGBDtoolbox
        |__ traintestSUNRGBD
            |__ allsplit.mat

|__ Mapillary
    |__ training
        |__ images
        |__ labels
    |__ validation
        |__ images
        |__ labels

|__ SmithCVPR2013_dataset_resized (HELEN)
    |__ images
    |__ labels
    |__ exemplars.txt
    |__ testing.txt
    |__ tuning.txt

|__ CelebAMask-HQ
    |__ CelebA-HQ-img
    |__ CelebAMask-HQ-mask-anno
    |__ CelebA-HQ-to-CelebA-mapping.txt

|__ LaPa
    |__ train
        |__ images
        |__ labels
    |__ val
        |__ images
        |__ labels
    |__ test
        |__ images
        |__ labels

|__ ibugmask_release
    |__ train
    |__ test

|__ FaceSynthetics
    |__ dataset_100000
    |__ dataset_1000
    |__ dataset_100
```

> Note: For PASCALContext, download the annotations from [here](pcannos) and put it in VOC2010.

> Note: For CelebAMask-HQ, run the preprocess script. `python3 scripts/preprocess_celebamaskhq.py --root <DATASET-ROOT-DIR>`.

</details>

<br>
<details>
  <summary><strong>Augmentations</strong> (click to expand)</summary>

Check out the notebook [here](.aug_test.ipynb) to test the augmentation effects.

Pixel-level Transforms:
* ColorJitter (Brightness, Contrast, Saturation, Hue)
* Gamma, Sharpness, AutoContrast, Equalize, Posterize
* GaussianBlur, Grayscale

Spatial-level Transforms:
* Affine, RandomRotation
* HorizontalFlip, VerticalFlip
* CenterCrop, RandomCrop
* Pad, ResizePad, Resize
* RandomResizedCrop

</details>

## <div align="center">Usage</div>

<details open>
  <summary><strong>Requirements</strong></summary>

* python >= 3.6
* torch >= 1.8.1
* torchvision >= 0.9.1

Other requirements can be installed with `pip install -r requirements.txt`.

</details>

<br>
<details>
  <summary><strong>Configuration</strong> (click to expand)</summary>

Create a configuration file in `configs`. Sample configuration for ADE20K dataset can be found [here](configs/ade20k.yaml). Then edit the fields you think if it is needed. This configuration file is needed for all of training, evaluation and prediction scripts.

</details>

<br>
<details>
  <summary><strong>Training</strong> (click to expand)</summary>

To train with a single GPU:

```bash
$ python tools/train.py --cfg configs/CONFIG_FILE.yaml
```

To train with multiple gpus, set `DDP` field in config file to `true` and run as follows:

```bash
$ python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train.py --cfg configs/<CONFIG_FILE_NAME>.yaml
```

</details>

<br>
<details>
  <summary><strong>Evaluation</strong> (click to expand)</summary>

Make sure to set `MODEL_PATH` of the configuration file to your trained model directory.

```bash
$ python tools/val.py --cfg configs/<CONFIG_FILE_NAME>.yaml
```

To evaluate with multi-scale and flip, change `ENABLE` field in `MSF` to `true` and run the same command as above.

</details>

<br>
<details open>
  <summary><strong>Inference</strong></summary>

To make an inference, edit the parameters of the config file from below.
* Change `MODEL` >> `NAME` and `VARIANT` to your desired pretrained model.
* Change `DATASET` >> `NAME` to the dataset name depending on the pretrained model.
* Set `TEST` >> `MODEL_PATH` to pretrained weights of the testing model.
* Change `TEST` >> `FILE` to the file or image folder path you want to test.
* Testing results will be saved in `SAVE_DIR`.

```bash
## example using ade20k pretrained models
$ python tools/infer.py --cfg configs/ade20k.yaml
```

Example test results:

![test_result](./assests/infer_result.jpg)

</details>

<br>
<details>
  <summary><strong>Convert to other Frameworks</strong> (ONNX, CoreML, OpenVINO, TFLite)</summary>

To convert to ONNX and CoreML, run:

```bash
$ python tools/export.py --cfg configs/<CONFIG_FILE_NAME>.yaml
```

To convert to OpenVINO and TFLite, see [torch_optimize](https://github.com/sithu31296/torch_optimize).

</details>

<br>
<details>
  <summary><strong>Inference</strong> (ONNX, OpenVINO, TFLite)</summary>

```bash
## ONNX Inference
$ python scripts/onnx_infer.py --model <ONNX_MODEL_PATH> --img-path <TEST_IMAGE_PATH>

## OpenVINO Inference
$ python scripts/openvino_infer.py --model <OpenVINO_MODEL_PATH> --img-path <TEST_IMAGE_PATH>

## TFLite Inference
$ python scripts/tflite_infer.py --model <TFLite_MODEL_PATH> --img-path <TEST_IMAGE_PATH>
```

</details>

<br>
<details>
  <summary><strong>References</strong> (click to expand)</summary>

* https://github.com/CoinCheung/BiSeNet
* https://github.com/open-mmlab/mmsegmentation
* https://github.com/rwightman/pytorch-image-models

</details>

<br>
<details>
  <summary><strong>Citations</strong> (click to expand)</summary>

```
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.15203},
  year={2021}
}

@misc{xiao2018unified,
  title={Unified Perceptual Parsing for Scene Understanding}, 
  author={Tete Xiao and Yingcheng Liu and Bolei Zhou and Yuning Jiang and Jian Sun},
  year={2018},
  eprint={1807.10221},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@article{hong2021deep,
  title={Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes},
  author={Hong, Yuanduo and Pan, Huihui and Sun, Weichao and Jia, Yisong},
  journal={arXiv preprint arXiv:2101.06085},
  year={2021}
}

@misc{zhang2021rest,
  title={ResT: An Efficient Transformer for Visual Recognition}, 
  author={Qinglong Zhang and Yubin Yang},
  year={2021},
  eprint={2105.13677},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{huang2021fapn,
  title={FaPN: Feature-aligned Pyramid Network for Dense Image Prediction}, 
  author={Shihua Huang and Zhichao Lu and Ran Cheng and Cheng He},
  year={2021},
  eprint={2108.07058},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{wang2021pvtv2,
  title={PVTv2: Improved Baselines with Pyramid Vision Transformer}, 
  author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
  year={2021},
  eprint={2106.13797},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@article{Liu2021PSA,
  title={Polarized Self-Attention: Towards High-quality Pixel-wise Regression},
  author={Huajun Liu and Fuqiang Liu and Xinyi Fan and Dong Huang},
  journal={Arxiv Pre-Print arXiv:2107.00782 },
  year={2021}
}

@misc{chao2019hardnet,
  title={HarDNet: A Low Memory Traffic Network}, 
  author={Ping Chao and Chao-Yang Kao and Yu-Shan Ruan and Chien-Hsiang Huang and Youn-Long Lin},
  year={2019},
  eprint={1909.00948},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@inproceedings{sfnet,
  title={Semantic Flow for Fast and Accurate Scene Parsing},
  author={Li, Xiangtai and You, Ansheng and Zhu, Zhen and Zhao, Houlong and Yang, Maoke and Yang, Kuiyuan and Tong, Yunhai},
  booktitle={ECCV},
  year={2020}
}

@article{Li2020SRNet,
  title={Towards Efficient Scene Understanding via Squeeze Reasoning},
  author={Xiangtai Li and Xia Li and Ansheng You and Li Zhang and Guang-Liang Cheng and Kuiyuan Yang and Y. Tong and Zhouchen Lin},
  journal={ArXiv},
  year={2020},
  volume={abs/2011.03308}
}

@ARTICLE{Yucondnet21,
  author={Yu, Changqian and Shao, Yuanjie and Gao, Changxin and Sang, Nong},
  journal={IEEE Signal Processing Letters}, 
  title={CondNet: Conditional Classifier for Scene Segmentation}, 
  year={2021},
  volume={28},
  number={},
  pages={758-762},
  doi={10.1109/LSP.2021.3070472}
}

```

</details>