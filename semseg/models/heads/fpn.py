import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.layers import ConvModule


class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks
    https://arxiv.org/abs/1901.02446
    """
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.lateral_convs = nn.ModuleList([])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[::-1]:
            self.lateral_convs.append(ConvModule(ch, channel, 1))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.lateral_convs[0](features[0])
        
        for i in range(1, len(features)):
            out = F.interpolate(out, scale_factor=2.0, mode='nearest')
            out = out + self.lateral_convs[i](features[i])
            out = self.output_convs[i](out)
        out = self.conv_seg(self.dropout(out))
        return out


if __name__ == '__main__':
    from semseg.models.backbones import ResNet
    backbone = ResNet('50')
    head = FPNHead([256, 512, 1024, 2048], 128, 19)
    x = torch.randn(2, 3, 224, 224)
    features = backbone(x)
    out = head(features)
    out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
    print(out.shape)