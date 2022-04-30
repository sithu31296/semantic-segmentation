## Scene Parsing

Accurate Models

Method | Backbone | ADE20K <br><sup>(mIoU) | Cityscapes <br><sup>(mIoU) | COCO-Stuff <br><sup>(mIoU) |Params <br><sup>(M) | GFLOPs <br><sup>(512x512) | GFLOPs <br><sup>(1024x1024) | Weights
--- | --- | --- | --- | --- | --- | --- | --- | ---
SegFormer | MiT-B1 | 42.2 | 78.5 | 40.2 | 14 | 16 | 244 | [ade][segformerb1]
|| MiT-B2 | 46.5 | 81.0 | 44.6 | 28 | 62 | 717 | [ade][segformerb2] 
|| MiT-B3 | 49.4 | 81.7 | 45.5 | 47 | 79 | 963 | [ade][segformerb3]
||
Light-Ham | VAN-S | 45.7 | - | - | 15 | 21 | - | -
|| VAN-B | 49.6 | - | - | 27 | 34 | - | -
|| VAN-L | 51.0 | - | - | 46 | 55 | - | -
||
Lawin | MiT-B1 | 42.1 | 79.0 | 40.5 | 14 | 13 | 218 | -
|| MiT-B2 | 47.8 | 81.7 | 45.2 | 30 | 45 | 563 | -
|| MiT-B3 | 50.3 | 82.5 | 46.6 | 50 | 62 | 809 | -
||
TopFormer | TopFormer-T | 34.6 | - | - | 1.4 | 0.6 | - | -
|| TopFormer-S | 37.0 | - | - | 3.1 | 1.2 | - | -
|| TopFormer-B | 39.2 | - | - | 5.1 | 1.8 | - | -

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
DDRNet | DDRNet-23slim | 77.8 | 74.7 | 6 | 36 | [city][ddrnet]

* mIoU results are with a single scale from official papers.
* Cityscapes image size = 1024x2048 (except BiSeNetv1 & 2 which uses 512x1024)
* CamVid image size = 960x720


## Face Parsing

Method | Backbone | HELEN-val <br><sup>(mIoU) | Params <br><sup>(M) | GFLOPs <br><sup>(512x512) | FPS <br><sup>(GTX1660ti) | Weights
--- | --- | --- | --- | --- | --- | --- 
BiSeNetv1 | ResNet-18 | 58.50 | 14 | 13 | 263 | [HELEN](https://drive.google.com/file/d/1HMC6OiFPc-aYwhlHlPYoXa-VCR3r2WPQ/view?usp=sharing)
BiSeNetv2 | - | 58.58 | 18 | 15 | 195 | [HELEN](https://drive.google.com/file/d/1cf-W_2m-vfxMRZ0mFQjEwhOglURpH7m6/view?usp=sharing)
DDRNet | DDRNet-23slim | 61.11 | 6 | 5 | 180 | [HELEN](https://drive.google.com/file/d/1SdOgVvgYrp8UFztHWN6dHH0MhP8zqnyh/view?usp=sharing)
SFNet | ResNetD-18 | 61.00 | 14 | 31 | 56 | [HELEN](https://drive.google.com/file/d/13w42DgI4PJ05bkWY9XCK_skSGMsmXroj/view?usp=sharing)


[ddrnet]: https://drive.google.com/file/d/1VdE3OkrIlIzLRPuT-2So-Xq_5gPaxm0t/view?usp=sharing
[segformerb3]: https://drive.google.com/file/d/1-OmW3xRD3WAbJTzktPC-VMOF5WMsN8XT/view?usp=sharing
[segformerb2]: https://drive.google.com/file/d/1AcgEK5aWMJzpe8tsfauqhragR0nBHyPh/view?usp=sharing
[segformerb1]: https://drive.google.com/file/d/18PN_P3ajcJi_5Q2v8b4BP9O4VdNCpt6m/view?usp=sharing
[topformert]: https://drive.google.com/file/d/1OnS3_PwjJuNMWCKisreNxw_Lma8uR8bV/view?usp=sharing
[topformers]: https://drive.google.com/file/d/19041fMb4HuDyNhIYdW1r5612FyzpexP0/view?usp=sharing
[topformerb]: https://drive.google.com/file/d/1m7CxYKWAyJzl5W3cj1vwsW4DfqAb_rqz/view?usp=sharing