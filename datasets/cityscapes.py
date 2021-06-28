import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision import io
from pathlib import Path
from typing import Tuple, Union, List


class CityScapes(Dataset):
    CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 
                'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    PALETTE = torch.tensor([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], 
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    ID2TRAINID = {0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255,
                  17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: -1}
    
    def __init__(self, root: str, split: str = 'train', img_size: Union[int, Tuple[int], List[int]] = 512, transforms = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']

        self.transforms = transforms
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        self.image_transforms = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.Normalize((0.3257, 0.3690, 0.3223), (0.2112, 0.2148, 0.2115))
        ])
        self.label_transforms = T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST)

        if split != 'test':
            img_path = Path(root) / 'leftImg8bit' / split
            self.files = list(img_path.rglob('*.png'))
        
            if not self.files:
                raise Exception(f"No images found in {img_path}")

            print(f"Found {len(self.files)} {split} images.")


    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('leftImg8bit', 'gtFine').replace('.png', '_labelIds.png')

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)
        
        if self.transforms:
            image, label = self.transforms(image, label)

        image, label = self.transform(image, label)
        return image, label
        

    def transform(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image = image.float()
        image /= 255
        return self.image_transforms(image), self.encode(self.label_transforms(label).squeeze()).long()


    def encode(self, label: torch.Tensor) -> torch.Tensor:
        for id, trainid in self.ID2TRAINID.items():
            label[label == id] = trainid
        return label

    
    def decode(self, label: torch.Tensor) -> torch.Tensor:
        return self.PALETTE[label.to(int)]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    root = 'C:\\Users\\sithu\\Documents\\Datasets\\CityScapes'

    dataset = CityScapes(root, split="train", img_size=(480, 640))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4)
    image, label = next(iter(dataloader))
    print(image.shape, label.shape)
    print(label.unique())
    # labels = [dataset.decode(lbl).permute(2, 0, 1) for lbl in label]
    # labels = torch.stack(labels)

    # inv_normalize = T.Normalize(
    #     mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
    #     std=(1/0.229, 1/0.224, 1/0.225)
    # )
    # image = inv_normalize(image)
    # image *= 255
    # images = torch.vstack([image, labels])
    
    # plt.imshow(make_grid(images, nrow=4).to(torch.uint8).numpy().transpose((1, 2, 0)))
    # plt.show()
