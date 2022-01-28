import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.backbones import *
from semseg.models.backbones.layers import trunc_normal_
from semseg.models.heads import LawinHead


class Lawin(nn.Module):
    """
    Notes::::: This implementation has larger params and FLOPs than the results reported in the paper.
    Will update the code and weights if the original author releases the full code.
    """
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__()
        backbone, variant = backbone.split('-')
        self.backbone = eval(backbone)(variant)
        self.decode_head = LawinHead(self.backbone.embed_dims, 256 if variant == 'B0' else 512, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y


if __name__ == '__main__':
    model = Lawin('MiT-B0')
    x = torch.zeros(2, 3, 512, 512)
    y = model(x)
    print(y.shape)