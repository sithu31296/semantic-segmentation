import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class COCOStuff(Dataset):
    """
    https://github.com/nightrome/cocostuff

    COCO-Stuff - 164k images
        based on COCO 2017
        118k train, 5k val, 20k test-dev
        num_classes: 171 (80 thing classes, 91 stuff classes)
    """
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'parking meter', 'bench', 'bird', 
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'shoe', 'eye glasses', 'handbag', 'suitcase', 'skis', 'snowboard', 
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'bowl', 
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 
        'window', 'desk', 'toilet', 'door', 'laptop', 'mouse', 'keyboard', 'microwave', 'oven', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 
        'teddy bear', 'hair drier', 'toothbrush', 'hair brush', 'banner', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-tile', 
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 
        'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 
        'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 
        'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 
        'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 
        'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood'
    ]

    PALETTE = torch.tensor([
        [178,  55, 178], [ 82, 178, 178], [  0,  87, 178], [178,  22, 178], [178, 178, 142], [177, 178,   0], [178,  32, 178], [ 62, 178, 110], [178,  76, 178], [ 72, 178,  99],
        [  0, 178, 178], [178, 147, 178], [ 70, 178, 178], [  0, 141, 178], [178,   0,  53], [178,   0,   3], [178,   0, 129], [178, 178,   0], [154, 178,  18], [  0,   0, 175],
        [  0,  99, 178], [115,   0, 178], [  0, 156, 178], [  0,   0, 146], [164, 178,   8], [151, 178,   0], [  0,   0, 132], [ 86, 178,  89], [178,   0,   0], [178,  86, 178],
        [  7, 178, 178], [147, 178,  28], [133, 178,  38], [  0, 150, 178], [178,  96, 178], [ 52, 178, 120], [145, 178, 178], [178, 178,   0], [ 94, 178,   0], [178, 133, 178],
        [ 82, 178,  89], [113, 178,  59], [143, 178,  28], [178, 167, 177], [178, 178,   0], [173,   1, 178], [178, 158,   0], [138,   0, 178], [178, 178,   0], [  0,  24, 178],
        [167, 178,   8], [165, 178,   0], [178,  35, 178], [178,   0, 116], [127, 178,  49], [  0, 127, 178], [  0, 124, 178], [178,  45, 178], [108, 178,   0], [  0, 178, 178],
        [178,  15, 178], [174, 169,   0], [178, 178,   0], [178,   0,  40], [  0,   0, 103], [  0,  11, 178], [120, 178, 178], [178, 134,   0], [178,  62, 178], [122, 178,   0],
        [178, 123,   0], [ 76, 178,  99], [  0,   0,  89], [123, 178,  49], [103, 178,  69], [  0, 112, 178], [  0,  49, 178], [178,  93, 178], [  0,   0, 178], [  0,  36, 178],
        [127,   0, 178], [178,   0,   0], [  0, 170, 178], [ 55, 178, 120], [178,   5, 166], [158, 174, 178], [ 11, 175, 160], [ 32, 178, 178], [ 42, 178, 130], [157, 178,  18],
        [178,  82, 178], [  0, 178, 178], [178,  11, 178], [ 57, 178, 178], [137, 178,  38], [  0, 178, 178], [178,   0, 103], [178,   0, 141], [178,   0,  15], [178,  72, 178],
        [178,  52, 178], [178, 137, 178], [178, 178,   0], [178, 178, 107], [178, 157, 178], [ 66, 178, 110], [178,  66, 178], [178,   0,  78], [178,   0,   0], [178, 116, 178],
        [  0,  98, 178], [  0, 178, 178], [178,   0,  66], [ 19, 178, 178], [  1, 162, 171], [178,   0, 154], [178, 178,   0], [178, 178,   0], [  0, 113, 178], [ 45, 178, 178],
        [178, 178,   0], [ 25, 178, 150], [ 96, 178,  79], [178, 143, 178], [171, 164, 178], [178, 178, 154], [  0, 178, 178], [178,  25, 178], [  0,   0, 117], [178,   0,   0],
        [ 95, 178, 178], [  5, 178, 171], [178,  42, 178], [178,   0,   0], [  0, 137, 178], [150,   0, 178], [ 35, 178, 140], [ 15, 178, 160], [178, 146,   0], [108, 178, 178],
        [178, 178,   0], [178, 103, 178], [178, 178,   0], [178, 111,   0], [178, 178,   0], [  0,  61, 178], [ 22, 178, 150], [178,   0,   0], [178,   0,  91], [178, 178, 119],
        [178, 127, 178], [178,   0,  28], [178, 178, 131], [178,   0,   0], [  0,  74, 178], [178, 177, 166], [ 45, 178, 130], [162,   0, 178], [178, 113, 178], [178, 123, 178],
        [ 93, 178,  79], [178, 154, 178], [178, 178,   0], [178, 106, 178], [  0, 178, 178], [  0,   0, 160], [133, 178, 178], [106, 178,  69], [136, 178,   0], [ 32, 178, 140], [116, 178,  59]
    ])
    UNUSEID = [11, 25, 28, 29, 44, 65, 67, 68, 70, 82, 90]

    def __init__(self, root: str, split: str = 'train', transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        self.use_id = [id for id in range(182) if id not in self.UNUSEID]
        self.label_map = np.arange(256)
        for i, id in enumerate(self.use_id):
            self.label_map[id] = i

        img_path = Path(root) / 'images' / f"{split}2017"
        self.files = list(img_path.glob('*.jpg'))

        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('images', 'labels').replace('.jpg', '.png')

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)

        if self.transform:
            image, label = self.transform(image, label)
        return image, self.encode(label.squeeze().numpy()).long()

    def encode(self, label: Tensor) -> Tensor:
        label = self.label_map[label]
        return torch.from_numpy(label)


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(COCOStuff, '/home/sithu/datasets/COCO')
