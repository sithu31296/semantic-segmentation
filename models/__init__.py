from .segformer import SegFormer, segformer_settings
from .ddrnet import DDRNet, ddrnet_settings
from .hardnet import HarDNet, hardnet_settings
from .sfnet import SFNet, sfnet_settings

__all__ = ['SegFormer', 'DDRNet', 'HarDNet', 'SFNet']

def get_model(model_name: str, variant: str, num_classes: int):
    assert model_name in __all__, f"Only {__all__} models are supported."
    return eval(model_name)(variant, num_classes)