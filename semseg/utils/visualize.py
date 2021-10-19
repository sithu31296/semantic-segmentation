import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid
from semseg.augmentations import Compose, Normalize, RandomResizedCrop


def visualize_dataset_sample(dataset, root, split='val', batch_size=4):
    transform = Compose([
        RandomResizedCrop((512, 512), scale=(1.0, 1.0)),
        Normalize()
    ])

    dataset = dataset(root, split=split, transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    image, label = next(iter(dataloader))

    print(f"Image Shape\t: {image.shape}")
    print(f"Label Shape\t: {label.shape}")
    print(f"Classes\t\t: {label.unique().tolist()}")

    label[label == -1] = 0
    label[label == 255] = 0
    labels = [dataset.PALETTE[lbl.to(int)].permute(2, 0, 1) for lbl in label]
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


def generate_palette(num_classes, background: bool = False):
    if background:
        palette = [[0, 0, 0]]
        palette += [[random.randint(0, 255)]*3 for _ in range(num_classes - 1)]
    else:
        palette = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(num_classes)]
    return np.array(palette)