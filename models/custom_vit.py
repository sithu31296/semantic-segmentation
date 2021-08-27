import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F
from .backbones import ResT
from .backbones.layers import trunc_normal_
from .heads import UPerHead


class CustomVIT(nn.Module):
    def __init__(self, variant: str = 'S', num_classes: int = 19) -> None:
        super().__init__()
        self.backbone = ResT(variant)
        self.decode_head = UPerHead(self.backbone.embed_dims, 128, num_classes)
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
    model = CustomVIT('S', 19)
    model.init_pretrained('checkpoints/backbones/rest/rest_small.pth')
    x = torch.zeros(2, 3, 512, 512)
    y = model(x)
    print(y.shape)
        

