import torch 
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import io
from pathlib import Path
from typing import Tuple


class SunRGBD(Dataset):
    """
    num_classes: 37
    """
    CLASSES = ['void', 'wall', 'floor', 'cabinet', 'bed', 'chair',
                           'sofa', 'table', 'door', 'window', 'bookshelf',
                           'picture', 'counter', 'blinds', 'desk', 'shelves',
                           'curtain', 'dresser', 'pillow', 'mirror',
                           'floor mat', 'clothes', 'ceiling', 'books',
                           'fridge', 'tv', 'paper', 'towel', 'shower curtain',
                           'box', 'whiteboard', 'person', 'night stand',
                           'toilet', 'sink', 'lamp', 'bathtub', 'bag']

    PALETTE = torch.tensor([
        (0, 0, 0), (119, 119, 119), (244, 243, 131),
        (137, 28, 157), (150, 255, 255), (54, 114, 113),
        (0, 0, 176), (255, 69, 0), (87, 112, 255), (0, 163, 33),
        (255, 150, 255), (255, 180, 10), (101, 70, 86),
        (38, 230, 0), (255, 120, 70), (117, 41, 121),
        (150, 255, 0), (132, 0, 255), (24, 209, 255),
        (191, 130, 35), (219, 200, 109), (154, 62, 86),
        (255, 190, 190), (255, 0, 255), (152, 163, 55),
        (192, 79, 212), (230, 230, 230), (53, 130, 64),
        (155, 249, 152), (87, 64, 34), (214, 209, 175),
        (170, 0, 59), (255, 0, 0), (193, 195, 234), (70, 72, 115),
        (255, 255, 0), (52, 57, 131), (12, 83, 45)
    ])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        split = 'train' if split == 'train' else 'test'

        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = -1

        img_path = Path(root) / 'images' / split 
        self.files = list(img_path.glob('*.jpg'))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")

        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('images', 'annotations').replace('.jpg', '.png')

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)
        
        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long() - 1    # subtract -1 since original label index starts from 1
        
    def decode(self, label: Tensor) -> Tensor:
        return self.PALETTE[label.to(int)]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision import transforms as T
    from torchvision.utils import make_grid
    from transforms import Compose, ResizePad, Normalize, RandomResizedCrop

    root = 'C:\\Users\\sithu\\Documents\\Datasets\\ADEChallenge\\ADEChallengeData2016'
    transform = Compose([RandomResizedCrop((512, 512)), Normalize()])

    dataset = ADE20K(root, split="train", transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4)
    image, label = next(iter(dataloader))
    print(image.shape, label.shape)
    print(label.unique())
    label[label == -1] = 0
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