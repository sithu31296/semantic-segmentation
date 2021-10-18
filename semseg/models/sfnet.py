import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.backbones import *
from semseg.models.heads import SFHead


class SFNet(nn.Module):
    def __init__(self, backbone: str = 'ResNetD-18', num_classes: int = 19):
        super().__init__()
        backbone, variant = backbone.split('-')
        self.backbone = eval(backbone)(variant)
        self.head = SFHead(self.backbone.channels, 128 if variant == '18' else 256, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x: Tensor) -> Tensor:
        outs = self.backbone(x)
        out = self.head(outs)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return out


if __name__ == '__main__':
    model = SFNet('ResNetD-18')
    model.init_pretrained('checkpoints/backbones/resnetd/resnetd18.pth')
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)