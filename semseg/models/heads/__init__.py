from .upernet import UPerHead
from .segformer import SegFormerHead
from .sfnet import SFHead
from .fpn import FPNHead
from .fapn import FaPNHead
from .fcn import FCNHead
from .condnet import CondHead
from .lawin import LawinHead

__all__ = ['UPerHead', 'SegFormerHead', 'SFHead', 'FPNHead', 'FaPNHead', 'FCNHead', 'CondHead', 'LawinHead']