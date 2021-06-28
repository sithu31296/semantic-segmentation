from .segformer import SegFormer
from .volo import VOLO

__all__ = {
    'segformer': SegFormer,
    'volo': VOLO
}

def choose_models(model_name: str, variant: str, num_classes: int, image_size: int):
    assert model_name in __all__.keys(), f"Only {list(__all__.keys())} models are supported."
    return __all__[model_name](variant, num_classes, image_size)