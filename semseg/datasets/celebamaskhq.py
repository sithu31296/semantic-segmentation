import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
from torchvision import transforms as T


class CelebAMaskHQ(Dataset):
    CLASSES = [
        'background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 
        'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth'
    ]
    PALETTE = torch.tensor([
        [0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], 
        [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]
    ])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.root = Path(root)
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.resize = T.Resize((512, 512))

        with open(self.root / f'{split}_list.txt') as f:
            self.files = f.read().splitlines()

        if not self.files: 
            raise Exception(f"No images found in {root}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = self.root / 'CelebA-HQ-img' / f"{self.files[index]}.jpg"
        lbl_path = self.root / 'CelebAMask-HQ-label' / f"{self.files[index]}.png"
        image = io.read_image(str(img_path))
        image = self.resize(image)
        label = io.read_image(str(lbl_path))

        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long()


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(CelebAMaskHQ, '/home/sithu/datasets/CelebAMask-HQ')