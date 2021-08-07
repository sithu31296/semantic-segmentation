# <div align="center">Semantic, Panoptic, Entity Segmentation</div>

![banner](assests/banner.jpg)

## <div align="center">Model Zoo</div>

[segformer]: https://arxiv.org/abs/2105.15203v2
[volo]: https://arxiv.org/abs/2106.13112v1
[maskformer]: https://arxiv.org/abs/2107.06278v1
[openworld]: https://arxiv.org/abs/2107.14228
[cswin]: https://arxiv.org/abs/2107.00652

[mit]: https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing

[segformerw]: https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA?usp=sharing

<details open>
  <summary>Semantic Segmentation on ADE20K-val</summary>

Method | Backbone | mIoU <br><sup>(%) | Params <br><sup>(M) | GFLOPs<br><sup>(512x512) | Pretrained
--- | --- | --- | --- | --- | ---
[SegFormer][segformer] | [MiT-B0][mit] | 38.0 | 4 | 8 | [download][segformerw]
| | [MiT-B1][mit] | 43.1 | 14 | 16 | [download][segformerw]
| | [MiT-B2][mit] | 47.5 | 28 | 62 | [download][segformerw]
| | [MiT-B3][mit] | 50.0 | 47 | 79 | [download][segformerw]
| | [MiT-B4][mit] | 51.1 | 64 | 96 | [download][segformerw]
| | [MiT-B5][mit]* | 51.8 | 85 | 183 | [download][segformerw]
||
[VOLO][volo] | VOLO-D1 | 50.5 | - | - | N/A
| | VOLO-D3 | 52.9 | - | - | N/A
| | VOLO-D5 | 54.3 | - | - | N/A
||
[MaskFormer][maskformer] | Swin-T | 48.8 | 42 | 55 | N/A
| | Swin-S | 51.0 | 63 | 79 | N/A
||
[CSWin][cswin] | CSWin-T | 50.4 | 60 | - | N/A
| | CSWin-S | 50.8 | 65 | - | N/A
| | CSWin-B | 51.7 | 109 | - | N/A


> Note: `*` models use 640x640 image size. Only backbones pretrained on ImageNet1k are reported.

</details>

<details>
  <summary>Semantic Segmentation on CityScapes-val (click to expand)</summary>

Method | Backbone | mIoU <br><sup>(%) | Params <br><sup>(M) | GFLOPs<br><sup>(1024x1024) | Pretrained
--- | --- | --- | --- | --- | ---
[SegFormer][segformer] | [MiT-B0][mit] | 78.1 | 4 | 126 | [download][segformerw]
| | [MiT-B1][mit] | 80.0 | 14 | 244 | [download][segformerw]
| | [MiT-B2][mit] | 82.2 | 28 | 717 | [download][segformerw]
| | [MiT-B3][mit] | 83.3 | 47 | 963 | [download][segformerw]
| | [MiT-B4][mit] | 83.9 | 64 | 1241 | [download][segformerw]
| | [MiT-B5][mit] | 84.0 | 85 | 1460 | [download][segformerw]
||
[VOLO][volo] | VOLO-D1 | 83.1 | - | - | N/A
| | VOLO-D4 | 84.3 | - | - | N/A

</details>

<details>
  <summary>Panoptic Segmentation on COCO panoptic-val (click to expand)</summary>

Method | Backbone | PQ | PQ<sup>Th | PQ<sup>St | SQ | RQ | Params <br><sup>(M) | GFLOPs | Pretrained
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
[MaskFormer][maskformer] | Swin-T | 47.7 | 51.7 | 41.7 | 80.4 | 58.3 | 42 | 179 | N/A
| | Swin-S | 49.7 | 54.4 | 42.6 | 80.9 | 60.4 | 63 | 259 | N/A
| | Swin-B | 51.8 | 56.3 | 43.2 | 81.4 | 61.8 | 102 | 411 | N/A

</details>

<details>
  <summary>Entity Segmentation on COCO-val (click to expand)</summary>

Method | Backbone | Entity AP | Mask Rescore | Pretrained
--- | --- | --- | --- | ---
[Open-World Entity][openworld] | MiT-B0 | 28.8 | 30.4 | N/A
| | MiT-B2 | 35.1 | 36.6 | N/A
| | MiT-B3 | 36.9 | 38.5 | N/A
| | MiT-B5 | 37.2 | 38.7 | N/A


</details>

## <div align="center">Datasets</div>

[ade20k]: http://sceneparsing.csail.mit.edu/
[cityscapes]: https://www.cityscapes-dataset.com/
[camvid]: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
[cocostuff]: https://github.com/nightrome/cocostuff
[mhp]: https://lv-mhp.github.io/
[lip]: http://sysu-hcp.net/lip/index.php
[atr]: https://github.com/lemondan/HumanParsing-Dataset
[pascalcontext]: https://cs.stanford.edu/~roozbeh/pascal-context/
[pcannos]: https://drive.google.com/file/d/1hOQnuTVYE9s7iRdo-6iARWkN2-qCAoVz/view?usp=sharing

Dataset | Type | Categories | Train <br><sup>Images | Val<br><sup>Images | Test<br><sup>Images
--- | --- | --- | --- | --- | ---
[ADE20K][ade20k] | Indoor Scene Parsing | 150 | 20,210 | 2,000 | 3,352
[CityScapes][cityscapes] | Street Scene Parsing | 19 | 2,975 | 500 | 1,525<sup>+labels
[CamVid][camvid] | Street Scene Parsing | 11+1 | 367 | 101 | 233<sup>+labels
[PASCALContext][pascalcontext] | General Scene Parsing | 59 | 4,996 | 5,104 | 9,637
[COCO-Stuff][cocostuff] | General Scene Parsing | 171 | 118,000 | 5,000 | 20,000
[MHPv1][mhp] | Multi-Human Parsing | 18+1 | 3,000 | 1,000 | 980<sup>+labels
[MHPv2][mhp] | Multi-Human Parsing | 58+1 | 15,403 | 5,000 | 5,000
[LIP][lip] | Multi-Human Parsing | 19+1 | 30,462 | 10,000 | -
[CIHP][lip] | Multi-Human Parsing | 19+1 | 28,280 | 5,000 | -
[ATR][atr] | Single-Human Parsing | 17+1 | 16,000 | 700 | 1,000<sup>+labels

> Note: `+1` means '+background class'.

<details>
  <summary>Datasets Structure</summary>

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

        |__ CIHP
            |__ instance-leve_human_parsing
                |__ train
                    |__ Images
                    |__ Category_ids
                |__ val
                    |__ Images
                    |__ Category_ids

        |__ ATR
            |__ humanparsing
                |__ JPEGImages
                |__ SegmentationClassAug

```

> Note: For PASCALContext, download the annotations from [here](pcannos) and put it in VOC2010.

</details>

## <div align="center">Usage</div>

<details>
  <summary>Configuration (click to expand)</summary>

Create a configuration file in `configs`. Sample configuration for CityScapes dataset can be found [here](configs/cityscapes.yaml). Then edit the fields you think if it is needed. This configuration file is needed for all of training, evaluation and prediction scripts.

</details>

<details>
  <summary>Training (click to expand)</summary>

Train with 1 GPU:

```bash
$ python tools/train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

Train with 2 GPUs:

```bash
$ python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

</details>


<details>
  <summary>Evaluation (click to expand)</summary>

Make sure to set `MODEL_PATH` of the configuration file to your trained model directory.

```bash
$ python tools/val.py --cfg configs/CONFIG_FILE_NAME.yaml
```

</details>


<details>
  <summary>Inference (click to expand)</summary>

Make sure to set `MODEL_PATH` of the configuration file to model's weights.

```bash
$ python tools/infer.py --cfg configs/CONFIG_FILE_NAME.yaml
```

</details>

<details>
  <summary>Optimization (click to expand)</summary>

For optimizing these models for deployment, see [torch_optimize](https://github.com/sithu31296/torch_optimize).

</details>

<details>
  <summary>References (click to expand)</summary>



</details>