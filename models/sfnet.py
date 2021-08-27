import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .backbones import ResNetD
from .heads import SFHead



sfnet_settings = {
    '18': [[64, 128, 256, 512], 128],
    '50': [[256, 512, 1024, 2048], 256],
    '101': [[256, 512, 1024, 2048], 256]
}


class SFNet(nn.Module):
    def __init__(self, variant: str = '18', num_classes: int = 19):
        super().__init__()
        assert variant in sfnet_settings.keys(), f"SFNet model variant should be in {list(sfnet_settings.keys())}"
        in_channels, fpn_channel = sfnet_settings[variant]

        self.backbone = ResNetD(variant)
        self.head = SFHead(in_channels, fpn_channel, num_classes)

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
    model = SFNet('18')
    model.init_pretrained('checkpoints/backbones/resnetd/resnetd18.pth')
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)