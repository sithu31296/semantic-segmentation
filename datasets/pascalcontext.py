import torch 
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import io
from pathlib import Path
from typing import Tuple


class PASCALContext(Dataset):
    """
    https://cs.stanford.edu/~roozbeh/pascal-context/
    based on PASCAL VOC 2010
    num_classes: 59
    10,100 train+val
    9,637 test
    """
    CLASSES = [
        'aeroplane', 'bag', 'bed', 'bedclothes', 'bench',
        'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
        'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
        'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
        'floor', 'flower', 'food', 'grass', 'ground', 'horse',
        'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person',
        'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep',
        'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table',
        'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water',
        'window', 'wood'
    ]

    PALETTE = torch.tensor([
        [180, 120, 120], [6, 230, 230], [80, 50, 50],
        [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
        [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
        [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
        [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
        [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
        [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
        [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
        [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
        [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
        [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
        [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
        [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
        [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
        [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]
    ])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = -1

        self.images, self.labels = self.get_files(root, split)

        print(f"Found {len(self.images)} {split} images.")

    def get_files(self, root: str, split: str):
        root = Path(root)
        flist = root / 'ImageSets' / 'SegmentationContext' / f'{split}.txt'
        with open(flist) as f:
            files = f.read().splitlines()
        images, labels = [], []
    
        for fi in files:
            images.append(str(root / 'JPEGImages' / f'{fi}.jpg'))
            labels.append(str(root / 'SegmentationClassContext' / f'{fi}.png'))
        
        return images, labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = self.images[index]
        lbl_path = self.labels[index]

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)

        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long() - 1    # remove background class
        
    def decode(self, label: Tensor) -> Tensor:
        return self.PALETTE[label.to(int)]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision import transforms as T
    from torchvision.utils import make_grid
    from transforms import Compose, RandomResizedCrop, Normalize

    root = 'C:\\Users\\sithu\\Documents\\Datasets\\VOCdevkit\\VOC2010'
    transform = Compose([RandomResizedCrop((512, 512)), Normalize()])

    dataset = PASCALContext(root, split="train", transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4)
    image, label = next(iter(dataloader))
    print(image.shape, label.shape)
    print(label.unique())
    label[label == -1] = 0
    labels = [dataset.decode(lbl).permute(2, 0, 1) for lbl in label]
    labels = torch.stack(labels)

    inv_normalize = T.Normalize(
        mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
        std=(1/0.229, 1/0.224, 1/0.225)
    )
    image = inv_normalize(image)
    image *= 255
    images = torch.vstack([image, labels])
    
    plt.imshow(make_grid(images, nrow=4).to(torch.uint8).numpy().transpose((1, 2, 0)))
    plt.show()