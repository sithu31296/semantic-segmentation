import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision import io
from pathlib import Path
from typing import Tuple, Union, List


class CamVid(Dataset):
    CLASSES = ['Unlabelled', 'Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']
    
    PALETTE = torch.tensor([[0, 0, 0], [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128], [0, 0, 192], [128, 128, 0], [192, 128, 128], [64, 64, 128], [64, 0, 128], [64, 64, 0], [0, 128, 192]])
    
    def __init__(self, root: str, split: str = 'train', img_size: Union[int, Tuple[int], List[int]] = 512, transforms = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.split = split
        self.transforms = transforms
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        self.image_transforms = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.label_transforms = T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST)
        
        if split != 'test':
            img_path = Path(root) / split
            self.files = list(img_path.glob("*.png"))
        
            if not self.files:
                raise Exception(f"No images found in {img_path}")

            print(f"Found {len(self.files)} {split} images.")


    def __len__(self) -> int:
        return len(self.files)
    

    def __getitem__(self, index: int) -> tuple:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace(self.split, self.split + '_labels').replace('.png', '_L.png')

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)
        
        if self.transforms:
            image, label = self.transforms(image, label)

        image, label = self.transform(image, label)
        return image, label
        

    def transform(self, image, label):
        image = image.float()
        image /= 255
        return self.image_transforms(image), self.encode(self.label_transforms(label)).long()


    def encode(self, label: torch.Tensor) -> torch.Tensor:
        label = label.permute(1, 2, 0)
        mask = torch.zeros(label.shape[:-1])

        for index, color in enumerate(self.PALETTE):
            bool_mask = torch.eq(label, color)
            class_map = torch.all(bool_mask, dim=-1)
            mask[class_map] = index
        return mask

    
    def decode(self, label: torch.Tensor) -> torch.Tensor:
        return self.PALETTE[label.to(int)]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    root = "C:\\Users\\sithu\\Documents\\Datasets\\CamVid"

    dataset = CamVid(root, split="train", img_size=(480, 640), transforms=None)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4)
    image, label = next(iter(dataloader))
    labels = [dataset.decode(lbl).permute(2, 0, 1) for lbl in label]
    labels = torch.stack(labels)

    inv_normalize = T.Normalize(
        mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
        std=(1/0.229, 1/0.224, 1/0.225)
    )
    image = inv_normalize(image)
    image *= 255
    images = torch.vstack([image, labels])
    
    plt.imshow(make_grid(images, nrow=4).to(torch.uint8).numpy().transpose((1, 2, 0)))
    plt.show()