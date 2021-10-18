import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class HELEN(Dataset):
    CLASSES = ['background', 'skin', 'l-brow', 'r-brow', 'l-eye', 'r-eye', 'nose', 'u-lip', 'i-mouth', 'l-lip', 'hair']
    PALETTE = torch.tensor([[0, 0 ,0], [127, 0, 0], [254, 0, 0], [0, 84, 0], [169, 0, 50], [254, 84, 0], [255, 0, 84], [0, 118, 220], [84, 84, 0], [0, 84, 84], [84, 50, 0]])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        self.files = self.get_files(root, split)
        if not self.files: raise Exception(f"No images found in {root}")
        print(f"Found {len(self.files)} {split} images.")

    def get_files(self, root: str, split: str):
        root = Path(root)
        if split == 'train':
            split = 'exemplars'
        elif split == 'val':
            split = 'tuning'
        else:
            split = 'testing'
        with open(root / f'{split}.txt') as f:
            lines = f.read().splitlines()
        
        split_names = [line.split(',')[-1].strip() for line in lines if line != '']
        files = (root / 'images').glob("*.jpg")
        files = list(filter(lambda x: x.stem in split_names, files))
        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).split('.')[0].replace('images', 'labels')
        image = io.read_image(img_path)
        label = self.encode(lbl_path)

        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long()

    def encode(self, label_path: str) -> Tensor:
        mask_paths = sorted(list(Path(label_path).glob('*.png')))
        for i, mask_path in enumerate(mask_paths):
            mask = io.read_image(str(mask_path)).squeeze()
            if i == 0:
                label = torch.zeros(self.n_classes, *mask.shape)
            label[i, ...] = mask
        label = label.argmax(dim=0).unsqueeze(0)
        return label 


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(HELEN, '/home/sithu/datasets/SmithCVPR2013_dataset_resized')