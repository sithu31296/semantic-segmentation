from typing import Tuple, Union, List
from .augmentations import *
from .camvid import CamVid
from .cityscapes import CityScapes
from .ade20k import ADE20K
from .mhpv1 import MHPv1
from .mhpv2 import MHPv2
from .lip import LIP
from .cihp import CIHP
from .atr import ATR
from .pascalcontext import PASCALContext
from .cocostuff import COCOStuff

__all__ = {
    'camvid': CamVid,
    'cityscapes': CityScapes,
    'ade20k': ADE20K,
    'mhpv1': MHPv1,
    'mhpv2': MHPv2,
    'lip': LIP,
    'cihp': CIHP,
    'atr': ATR,
    'pascalconteext': PASCALContext,
    'cocostuff': COCOStuff
}

augs = {
    'colorjitter': ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    'gamma': AdjustGamma(gamma=0.2, gain=1),
    'sharpness': RandomAdjustSharpness(sharpness_factor=0.1, p=0.5),
    'contrast': RandomAutoContrast(p=0.5),
    'hflip': RandomHorizontalFlip(p=0.5),
    'vflip': RandomVerticalFlip(p=0.5),
    'blur': RandomGaussianBlur((3, 3), p=0.5),
    'grayscale': RandomGrayscale(p=0.5),
    'rotate': RandomRotation(degrees=10, p=0.2),

    'centercrop': CenterCrop((640, 640)),
    'randomcrop': RandomCrop((640, 640), p=0.2),
    'randomresizedcrop': RandomResizedCrop((640, 640), scale=(0.08, 1), ratio=(0.5, 1.25)),
    'pad': Pad(10, fill=114),
}

def get_dataset(dataset_name: str, root: str, split: str, transform = None):
    assert dataset_name in __all__.keys(), f"Only {list(__all__.keys())} datasets are supported as of now."
    return __all__[dataset_name](root, split, transform)


def get_train_transform(img_size: Union[int, Tuple[int], List[int]], aug_names: list):
    return Compose(
        *[augs[aug] for aug in aug_names if aug in augs],
        Resize(img_size),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    )


def get_val_transform(img_size: Union[int, Tuple[int], List[int]]):
    return Compose([
        Resize(img_size),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])