from .segformer import SegFormer
from .custom_cnn import CustomCNN
from .custom_vit import CustomVIT
from .ddrnet import DDRNet

__all__ = {
    'customcnn': CustomCNN,
    'customvit': CustomVIT,
    'segformer': SegFormer,
    'ddrnet': DDRNet,
}

def get_model(model_name: str, variant: str, num_classes: int):
    assert model_name in __all__.keys(), f"Only {list(__all__.keys())} models are supported."
    return __all__[model_name](variant, num_classes)