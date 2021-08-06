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


def get_dataset(dataset_name: str, root: str, split: str, transform = None):
    assert dataset_name in __all__.keys(), f"Only {list(__all__.keys())} datasets are supported as of now."
    return __all__[dataset_name](root, split, transform)