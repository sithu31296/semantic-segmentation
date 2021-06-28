from torch.utils import data
from typing import Tuple, Union, List
from .camvid import CamVid
from .cityscapes import CityScapes
from .ade20k import ADE20K

__all__ = {
    'camvid': CamVid,
    'cityscapes': CityScapes,
    'ade20k': ADE20K
}

def choose_datasets(dataset_name: str, root: str, split: str, img_size: Union[int, Tuple[int], List[int]] = 512, transforms = None):
    assert dataset_name in __all__.keys(), f"Only {list(__all__.keys())} datasets are supported as of now."
    return __all__[dataset_name](root, split, img_size, transforms)