import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class iBugMask(Dataset):
    CLASSES = ['background', 'skin', 'l-brow', 'r-brow', 'l-eye', 'r-eye', 'nose', 'u-lip', 'i-mouth', 'l-lip', 'hair']
    PALETTE = torch.tensor([[0, 0, 0], [255, 255, 0], [139, 76, 57], [139, 54, 38], [0, 205, 0], [0, 138, 0], [154, 50, 205], [72, 118, 255], [255, 165, 0], [0, 0, 139], [255, 0, 0]])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        split = 'train' if split == 'train' else 'test'
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        
        img_path = Path(root) / split 
        self.files = list(img_path.glob('*.jpg'))

        if not self.files: raise Exception(f"No images found in {root}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('.jpg', '.png')
        image = io.read_image(str(img_path))
        label = io.read_image(str(lbl_path))

        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long()


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(iBugMask, '/home/sithu/datasets/ibugmask_release')