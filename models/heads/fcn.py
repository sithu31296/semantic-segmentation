import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class FCNHead(nn.Module):
    def __init__(self, c1, c2, num_classes: int = 19):
        super().__init__()
        self.conv = ConvModule(c1, c2, 1)
        self.cls = nn.Conv2d(c2, num_classes, 1)

    def forward(self, features) -> Tensor:
        x = self.conv(features[-1])
        x = self.cls(x)
        return x


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from models.backbones.resnet import ResNet
    backbone = ResNet('50')
    head = FCNHead(2048, 256, 19)
    x = torch.randn(2, 3, 224, 224)
    features = backbone(x)
    out = head(features)
    print(out.shape)
    out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
    print(out.shape)
