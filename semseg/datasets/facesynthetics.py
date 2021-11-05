import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class FaceSynthetics(Dataset):
    CLASSES = ['background', 'skin', 'nose', 'r-eye', 'l-eye', 'r-brow', 'l-brow', 'r-ear', 'l-ear', 'i-mouth', 't-lip', 'b-lip', 'neck', 'hair', 'beard', 'clothing', 'glasses', 'headwear', 'facewear']
    PALETTE = torch.tensor([
        [0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], 
        [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]
    ])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        if split == 'train':
            split = 'dataset_100000'
        elif split == 'val':
            split = 'dataset_1000'
        else:
            split = 'dataset_100'
        
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        
        img_path = Path(root) / split
        images = img_path.glob('*.png')
        self.files = [path for path in images if '_seg' not in path.name]

        if not self.files: raise Exception(f"No images found in {root}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('.png', '_seg.png')
        image = io.read_image(str(img_path))
        label = io.read_image(str(lbl_path))

        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(FaceSynthetics, 'C:\\Users\\sithu\\Documents\\Datasets')