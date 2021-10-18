import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from scipy import io as sio
from pathlib import Path
from typing import Tuple


class SunRGBD(Dataset):
    """
    num_classes: 37
    """
    CLASSES = [
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 
        'floor mat', 'clothes', 'ceiling', 'books', 'fridge', 'tv', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag'
    ]

    PALETTE = torch.tensor([
        (119, 119, 119), (244, 243, 131), (137, 28, 157), (150, 255, 255), (54, 114, 113), (0, 0, 176), (255, 69, 0), (87, 112, 255), (0, 163, 33),
        (255, 150, 255), (255, 180, 10), (101, 70, 86), (38, 230, 0), (255, 120, 70), (117, 41, 121), (150, 255, 0), (132, 0, 255), (24, 209, 255),
        (191, 130, 35), (219, 200, 109), (154, 62, 86), (255, 190, 190), (255, 0, 255), (152, 163, 55), (192, 79, 212), (230, 230, 230), (53, 130, 64),
        (155, 249, 152), (87, 64, 34), (214, 209, 175), (170, 0, 59), (255, 0, 0), (193, 195, 234), (70, 72, 115), (255, 255, 0), (52, 57, 131), (12, 83, 45)
    ])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['alltrain', 'train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = -1
        self.files, self.labels = self.get_data(root, split)
        print(f"Found {len(self.files)} {split} images.")

    def get_data(self, root: str, split: str):
        root = Path(root)
        files, labels = [], []
        split_path = root / 'SUNRGBDtoolbox' / 'traintestSUNRGBD' / 'allsplit.mat'
        split_mat = sio.loadmat(split_path, squeeze_me=True, struct_as_record=False)
        if split == 'train':
            file_lists = split_mat['trainvalsplit'].train
        elif split == 'val':
            file_lists = split_mat['trainvalsplit'].val
        elif split == 'test':
            file_lists = split_mat['alltest']
        else:
            file_lists = split_mat['alltrain']
        
        for fl in file_lists:
            real_fl = root / fl.split('/n/fs/sun3d/data/')[-1]
            files.append(str(list((real_fl / 'image').glob('*.jpg'))[0]))
            labels.append(real_fl / 'seg.mat')

        assert len(files) == len(labels)
        return files, labels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = io.read_image(self.files[index], io.ImageReadMode.RGB)
        label = sio.loadmat(self.labels[index], squeeze_me=True, struct_as_record=False)['seglabel']
        label = torch.from_numpy(label.astype(np.uint8)).unsqueeze(0)
        
        if self.transform:
            image, label = self.transform(image, label)
        return image, self.encode(label.squeeze()).long() - 1    # subtract -1 to remove void class

    def encode(self, label: Tensor) -> Tensor:
        label[label > self.n_classes] = 0
        return label


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(SunRGBD, '/home/sithu/datasets/sunrgbd')