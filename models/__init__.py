from .segformer import SegFormer
from .volo import VOLO

__all__ = {
    'segformer': SegFormer,
    'volo': VOLO
}

def get_model(model_name: str, backbone: str, variant: str, num_classes: int, image_size):
    assert model_name in __all__.keys(), f"Only {list(__all__.keys())} models are supported."
    return __all__[model_name](variant, num_classes, image_size)