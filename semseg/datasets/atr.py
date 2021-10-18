import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class ATR(Dataset):
    """Single Person Fashion Dataset
    https://openaccess.thecvf.com/content_iccv_2015/papers/Liang_Human_Parsing_With_ICCV_2015_paper.pdf

    https://github.com/lemondan/HumanParsing-Dataset
    num_classes: 17+background
    16000 train images
    700 val images
    1000 test images with labels
    """
    CLASSES = ['background', 'hat', 'hair', 'sunglass', 'upper-clothes', 'skirt', 'pants', 'dress', 'belt', 'left-shoe', 'right-shoe', 'face', 'left-leg', 'right-leg', 'left-arm', 'right-arm', 'bag', 'scarf']
    PALETTE = torch.tensor([[0, 0, 0], [127, 0, 0], [254, 0, 0], [0, 84, 0], [169, 0, 50], [254, 84, 0], [255, 0, 84], [0, 118, 220], [84, 84, 0], [0, 84, 84], [84, 50, 0], [51, 85, 127], [0, 127, 0], [0, 0, 254], [50, 169, 220], [0, 254, 254], [84, 254, 169], [169, 254, 84]])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        img_path = Path(root) / 'humanparsing' / 'JPEGImages' 
        self.files = list(img_path.glob('*.jpg'))
        if split == 'train':
            self.files = self.files[:16000]
        elif split == 'val':
            self.files = self.files[16000:16700]
        else:
            self.files = self.files[16700:17700]
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('JPEGImages', 'SegmentationClassAug').replace('.jpg', '.png')

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)

        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long() 


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(ATR, '/home/sithu/datasets/LIP/ATR')