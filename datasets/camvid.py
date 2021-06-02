import os
import torch 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from torchvision import transforms
from torchvision import io
from glob import glob
from typing import Tuple, Union


class CamVid(Dataset):
    def __init__(self, root: str, mode: str = 'train', img_size: Union[int, Tuple[int]] = 512, augmentations = None) -> None:
        super().__init__()
        self.mode = mode
        self.augmentations = augmentations
        self.n_classes = 12
        self.ignore_label = 255
        self.root = root
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        self.class_names, self.colors = self.colorMap()
        self.colors_tensor = torch.tensor(self.colors)
        
        self.image_transforms = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.BILINEAR),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.label_transforms = transforms.Resize(img_size, interpolation=Image.NEAREST)
        self.files = os.listdir(os.path.join(root, mode))
    
        if not self.files:
            raise Exception(f"No images found in {os.path.join(root, mode)}")

        print(f"Found {len(self.files)} {mode} images.")


    def __len__(self) -> int:
        return len(self.files)
    

    def __getitem__(self, index: int) -> tuple:
        item = self.files[index]
        img_path = os.path.join(self.root, self.mode, item)
        lbl_path = os.path.join(self.root, self.mode+'_labels', os.path.basename(item)[:-4] + "_L.png")

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)
        
        if self.augmentations:
            image, label = self.augmentations(image, label)

        image, label = self.transform(image, label)
        return image, label
        

    def transform(self, image, label):
        image = image.float()
        image /= 255
        return self.image_transforms(image), self.encode(self.label_transforms(label)).long()


    def encode(self, label: torch.Tensor) -> torch.Tensor:
        label = label.permute(1, 2, 0)
        mask = torch.zeros(label.shape[:-1])

        for index, color in enumerate(self.colors):
            bool_mask = torch.eq(label, torch.tensor(color))
            class_map = torch.all(bool_mask, dim=-1)
            mask[class_map] = index
        return mask

    
    def decode(self, label: torch.Tensor) -> torch.Tensor:
        return self.colors_tensor[label.to(int)]

    @classmethod
    def colorMap(self) -> Tuple[dict, np.array]:
        """Return the colormap and color values of CamVid dataset
        """
        camVidColorMap = {
            "Unlabelled": [0, 0, 0],
            "Sky": [128, 128, 128],
            "Building": [128, 0, 0],
            "Pole": [192, 192, 128],
            "Road": [128, 64, 128],
            "Pavement": [0, 0, 192],
            "Tree": [128, 128, 0],
            "SignSymbol": [192, 128, 128],
            "Fence": [64, 64, 128],
            "Car": [64, 0, 128],
            "Pedestrian": [64, 64, 0],
            "Bicyclist": [0, 128, 192]
        }

        return list(camVidColorMap.keys()), list(camVidColorMap.values())


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    # from augmentations import Compose, randomCrop, randomRotate, randomHorizontalFlip

    root = "../datasets/CamVid"

    # augment = Compose([
    #     randomHorizontalFlip(p=0.5),
    #     randomCrop((256, 256), p=0.5),
    # ])
    augment = None

    dataset = CamVid(root, mode="train", img_size=(480, 640), augmentations=augment)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4)
    image, label = next(iter(dataloader))
    labels = [dataset.decode(lbl).permute(2, 0, 1) for lbl in label]
    labels = torch.stack(labels)

    inv_normalize = transforms.Normalize(
        mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
        std=(1/0.229, 1/0.224, 1/0.225)
    )
    image = inv_normalize(image)
    image *= 255
    images = torch.vstack([image, labels])
    
    plt.imshow(make_grid(images, nrow=4).to(torch.uint8).numpy().transpose((1, 2, 0)))
    plt.show()