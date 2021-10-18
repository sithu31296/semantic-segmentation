import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class LIP(Dataset):
    """
    num_classes: 19+background
    30462 train images
    10000 val images
    """
    CLASSES = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes', 'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt', 'face', 'left-arm', 'right-arm', 'left-leg', 'right-leg', 'left-shoe', 'right-shoe']
    PALETTE = torch.tensor([[0, 0, 0], [127, 0, 0], [254, 0, 0], [0, 84, 0], [169, 0, 50], [254, 84, 0], [255, 0, 84], [0, 118, 220], [84, 84, 0], [0, 84, 84], [84, 50, 0], [51, 85, 127], [0, 127, 0], [0, 0, 254], [50, 169, 220], [0, 254, 254], [84, 254, 169], [169, 254, 84], [254, 254, 0], [254, 169, 0]])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.split = split
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        img_path = Path(root) / 'TrainVal_images' / f'{split}_images' 
        self.files = list(img_path.glob('*.jpg'))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('TrainVal_images', 'TrainVal_parsing_annotations').replace(f'{self.split}_images', f'{self.split}_segmentations').replace('.jpg', '.png')

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)

        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long() 


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(LIP, '/home/sithu/datasets/LIP/LIP')