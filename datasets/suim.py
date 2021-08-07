import torch 
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import io
from pathlib import Path
from typing import Tuple
from PIL import Image
from torchvision.transforms import functional as TF


class SUIM(Dataset):
    """
    num_classes: 8
    """
    CLASSES = ['water', 'human divers', 'aquatic plants and sea-grass', 'wrecks and ruins', 'robots (AUVs/ROVs/instruments)', 'reefs and invertebrates', 'fish and vertebrates', 'sea-floor and rocks']
    PALETTE = torch.tensor([[0, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255], [255, 0, 0], [255, 0, 255], [255, 255, 0], [255, 255, 255]])
    
    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.split = 'train_val' if split == 'train' else 'TEST'
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = -100    # train all classes 

        img_path = Path(root) / self.split / 'images'
        self.files = list(img_path.glob("*.jpg"))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")

        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('images', 'masks').replace('.jpg', '.bmp')

        image = io.read_image(img_path)
        label = TF.pil_to_tensor(Image.open(lbl_path).convert('RGB'))

        if self.transform:
            image, label = self.transform(image, label)
        return image, self.encode(label).long()

    def encode(self, label: Tensor) -> Tensor:
        label = label.permute(1, 2, 0)
        mask = torch.zeros(label.shape[:-1])

        for index, color in enumerate(self.PALETTE):
            bool_mask = torch.eq(label, color)
            class_map = torch.all(bool_mask, dim=-1)
            mask[class_map] = index
        return mask
    
    def decode(self, label: Tensor) -> Tensor:
        return self.PALETTE[label.to(int)]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision import transforms as T
    from torchvision.utils import make_grid
    from transforms import Compose, RandomResizedCrop, Normalize

    root = "C:\\Users\\sithu\\Documents\\Datasets\\SUIM"
    transform = Compose([RandomResizedCrop((512, 512)), Normalize()])

    dataset = SUIM(root, split="train", transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4)
    image, label = next(iter(dataloader))
    print(image.shape, label.shape)
    print(label.unique().tolist())
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