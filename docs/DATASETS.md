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
