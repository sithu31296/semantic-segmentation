## Scene Parsing

Accurate Models

Method | Backbone | ADE20K <br><sup>(mIoU) | Cityscapes <br><sup>(mIoU) | COCO-Stuff <br><sup>(mIoU) |Params <br><sup>(M) | GFLOPs <br><sup>(512x512) | GFLOPs <br><sup>(1024x1024) | Weights
--- | --- | --- | --- | --- | --- | --- | --- | ---
SegFormer | MiT-B1 | 42.2 | 78.5 | 40.2 | 14 | 16 | 244 | [ade][segformerb1]
|| MiT-B2 | 46.5 | 81.0 | 44.6 | 28 | 62 | 717 | [ade][segformerb2] 
|| MiT-B3 | 49.4 | 81.7 | 45.5 | 47 | 79 | 963 | [ade][segformerb3]
||
Lawin | MiT-B1 | 42.1 | 79.0 | 40.5 | 14 | 13 | 218 | -
|| MiT-B2 | 47.8 | 81.7 | 45.2 | 30 | 45 | 563 | -
|| MiT-B3 | 50.3 | 82.5 | 46.6 | 50 | 62 | 809 | -

* mIoU results are with a single scale from official papers.
* ADE20K image size = 512x512
* Cityscapes image size = 1024x1024
* COCO-Stuff image size = 512x512

Real-time Models

Method | Backbone | CityScapes-val <br><sup>(mIoU) | CamVid <br><sup>(mIoU) | Params (M) | GFLOPs <br><sup>(1024x2048) | Weights
--- | --- | --- | --- | --- | --- | --- 
BiSeNetv1 | ResNet-18 | 74.8 | 68.7 | 14 | 49 | -
BiSeNetv2 | - | 73.4 | 72.4 | 18 | 21 | -
SFNet | ResNetD-18 | 79.0 | - | 13 | - | -
FCHarDNet | HarDNet-70 | 77.7 | - | 4 | 35 | -
DDRNet | DDRNet-23slim | 77.8 | 74.7 | 6 | 36 | [cityscapes][ddrnet]

* mIoU results are with a single scale from official papers.
* Cityscapes image size = 1024x2048 (except BiSeNetv1 & 2 which uses 512x1024)
* CamVid image size = 960x720


## Face Parsing

Method | Backbone | HELEN-val <br><sup>(mIoU) | Params <br><sup>(M) | GFLOPs <br><sup>(512x512) | FPS <br><sup>(GTX1660ti) | Weights
--- | --- | --- | --- | --- | --- | --- 
BiSeNetv1 | MobileNetV2-1.0 | 58.22 | 5 | 5 | 160 | [pt](https://drive.google.com/file/d/1gE1Ot0O0nzXNABYdTuThRi-BU3_VHuMy/view?usp=sharing)
BiSeNetv1 | ResNet-18 | 58.50 | 14 | 13 | 263 | [pt](https://drive.google.com/file/d/1HMC6OiFPc-aYwhlHlPYoXa-VCR3r2WPQ/view?usp=sharing)
BiSeNetv2 | - | 58.58 | 18 | 15 | 195 | [pt](https://drive.google.com/file/d/1cf-W_2m-vfxMRZ0mFQjEwhOglURpH7m6/view?usp=sharing)
FCHarDNet | HarDNet-70 | 59.38 | 4 | 4 | 130 | [pt](https://drive.google.com/file/d/1XTzpXairzUlQd3i91jOVmSDLs8Zd33jT/view?usp=sharing)
DDRNet | DDRNet-23slim | 61.11 | 6 | 5 | 180 | [pt](https://drive.google.com/file/d/1SdOgVvgYrp8UFztHWN6dHH0MhP8zqnyh/view?usp=sharing)\|[tflite(fp32)](https://drive.google.com/file/d/13yEnXjuv3hKY1cv1ycrjPUFNVyiJM89T/view?usp=sharing)\|[tflite(fp16)](https://drive.google.com/file/d/1IHdjt9wegKiDFvq9l1wglywvM8U0CFdo/view?usp=sharing)\|[tflite(int8)](https://drive.google.com/file/d/1NwPPzz_65YPcSfCfyXx7DiYeCMJMWKeH/view?usp=sharing)
SegFormer | MiT-B0 | 59.31 | 4 | 8 | 75 | [pt](https://drive.google.com/file/d/1uGRjPgX8wcHJlPalFntZOjG0Toxj0msO/view?usp=sharing)
SFNet | ResNetD-18 | 61.00 | 14 | 31 | 56 | [pt](https://drive.google.com/file/d/13w42DgI4PJ05bkWY9XCK_skSGMsmXroj/view?usp=sharing)


[ddrnet]: https://drive.google.com/file/d/17IMQS23StRcIvMXmjNwEcU9uGXuviGY-/view?usp=sharing
[segformerb3]: https://drive.google.com/file/d/1-OmW3xRD3WAbJTzktPC-VMOF5WMsN8XT/view?usp=sharing
[segformerb2]: https://drive.google.com/file/d/1AcgEK5aWMJzpe8tsfauqhragR0nBHyPh/view?usp=sharing
[segformerb1]: https://drive.google.com/file/d/18PN_P3ajcJi_5Q2v8b4BP9O4VdNCpt6m/view?usp=sharing