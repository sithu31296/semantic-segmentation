import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class LaPa(Dataset):
    CLASSES = ['background', 'skin', 'l-brow', 'r-brow', 'l-eye', 'r-eye', 'nose', 'u-lip', 'i-mouth', 'l-lip', 'hair']
    PALETTE = torch.tensor([[0, 0, 0], [0, 153, 255], [102, 255, 153], [0, 204, 153], [255, 255, 102], [255, 255, 204], [255, 153, 0], [255, 102, 255], [102, 0, 51], [255, 204, 255], [255, 0, 102]])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        
        img_path = Path(root) / split / 'images'
        self.files = list(img_path.glob('*.jpg'))

        if not self.files: raise Exception(f"No images found in {root}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('images', 'labels').replace('.jpg', '.png')
        image = io.read_image(str(img_path))
        label = io.read_image(str(lbl_path))

        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long()


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(LaPa, '/home/sithu/datasets/LaPa')