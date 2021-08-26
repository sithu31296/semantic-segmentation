from .segformer import SegFormer, segformer_settings
from .ddrnet import DDRNet, ddrnet_settings
from .hardnet import HarDNet, hardnet_settings

__all__ = {
    'segformer': SegFormer,
    'ddrnet': DDRNet,
    'hardnet': HarDNet,
}

def get_model(model_name: str, variant: str, num_classes: int):
    assert model_name in __all__.keys(), f"Only {list(__all__.keys())} models are supported."
    return __all__[model_name](variant, num_classes)