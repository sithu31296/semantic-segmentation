import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class MHPv2(Dataset):
    """
    25,403 images each with at least 2 persons (average 3)
    15,403 images for training
    5000 images for validation
    5000 images for testing
    num_classes: 58+background
    """
    CLASSES = ['background', 'cap/hat', 'helmet', 'face', 'hair', 'left-arm', 'right-arm', 'left-hand', 'right-hand', 'protector', 'bikini/bra', 'jacket/windbreaker/hoodie', 't-shirt', 'polo-shirt', 'sweater', 'singlet', 'torso-skin', 'pants', 'shorts/swim-shorts', 'skirt', 'stockings', 'socks', 'left-boot', 'right-boot', 'left-shoe', 'right-shoe', 'left-highheel', 'right-highheel', 'left-sandal', 'right-sandal', 'left-leg', 'right-leg', 'left-foot', 'right-foot', 'coat', 'dress', 'robe', 'jumpsuits', 'other-full-body-clothes', 'headware', 'backpack', 'ball', 'bats', 'belt', 'bottle', 'carrybag', 'cases', 'sunglasses', 'eyeware', 'gloves', 'scarf', 'umbrella', 'wallet/purse', 'watch', 'wristband', 'tie', 'other-accessories', 'other-upper-body-clothes', 'other-lower-body-clothes']
    PALETTE = torch.tensor([[0, 0, 0], [255, 114, 196], [63, 31, 34], [253, 1, 0], [254, 26, 1], [253, 54, 0], [253, 82, 0], [252, 110, 0], [253, 137, 0], [253, 166, 1], [254, 191, 0], [253, 219, 0], [252, 248, 0], [238, 255, 1], [209, 255, 0], [182, 255, 0], [155, 255, 0], [133, 254, 0], [102, 254, 0], [78, 255, 0], [55, 254, 1], [38, 255, 0], [30, 255, 13], [34, 255, 35], [35, 254, 64], [36, 254, 87], [37, 252, 122], [37, 255, 143], [35, 255, 172], [35, 255, 200], [40, 253, 228], [40, 255, 255], [37, 228, 255], [33, 198, 254], [31, 170, 254], [22, 145, 255], [26, 112, 255], [20, 86, 253], [22, 53, 255], [19, 12, 253], [19, 1, 246], [30, 1, 252], [52, 0, 254], [72, 0, 255], [102, 0, 255], [121, 1, 252], [157, 1, 245], [182, 0, 253], [210, 0, 254], [235, 0, 255], [253, 1, 246], [254, 0, 220], [255, 0, 191], [254, 0, 165], [252, 0, 137], [248, 2, 111], [253, 0, 81], [255, 0, 54], [253, 1, 26]])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        self.images, self.labels = self.get_files(root, split)
        print(f"Found {len(self.images)} {split} images.")

    def get_files(self, root: str, split: str):
        root = Path(root)
        all_labels = list((root / split / 'parsing_annos').rglob('*.png'))
        images = list((root / split / 'images').rglob('*.jpg'))
        labels = []
        
        for f in images:
            labels_per_images = list(filter(lambda x: x.stem.split('_', maxsplit=1)[0] == f.stem, all_labels))
            assert labels_per_images != []
            labels.append(labels_per_images)

        assert len(images) == len(labels)
        return images, labels

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.images[index])
        lbl_paths = self.labels[index]

        image = io.read_image(img_path)
        label = self.read_label(lbl_paths)

        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long()

    def read_label(self, lbl_paths: list) -> Tensor:
        labels = None
        label_idx = None

        for lbl_path in lbl_paths:
            label = io.read_image(str(lbl_path)).squeeze().numpy()
            if label.ndim != 2:
                label = label[0]
            if label_idx is None:
                label_idx = np.zeros(label.shape, dtype=np.uint8)
            label = np.ma.masked_array(label, mask=label_idx)
            label_idx += np.minimum(label, 1)
            if labels is None:
                labels = label
            else:
                labels += label
        return torch.from_numpy(labels.data).unsqueeze(0).to(torch.uint8)


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(MHPv2, '/home/sithu/datasets/LV-MHP-v2')