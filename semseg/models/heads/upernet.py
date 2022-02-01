import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple
from semseg.models.layers import ConvModule
from semseg.models.modules import PPM


class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    scales: Pooling scales used in PPM module applied on the last feature
    """
    def __init__(self, in_channels, channel=128, num_classes: int = 19, scales=(1, 2, 3, 6)):
        super().__init__()
        # PPM Module
        self.ppm = PPM(in_channels[-1], channel, scales)

        # FPN Module
        self.fpn_in = nn.ModuleList()
        self.fpn_out = nn.ModuleList()

        for in_ch in in_channels[:-1]: # skip the top layer
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))

        self.bottleneck = ConvModule(len(in_channels)*channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channel, num_classes, 1)


    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features)-1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=False)
 
        output = self.bottleneck(torch.cat(fpn_features, dim=1))
        output = self.conv_seg(self.dropout(output))
        return output


if __name__ == '__main__':
    model = UPerHead([64, 128, 256, 512], 128)
    x1 = torch.randn(2, 64, 56, 56)
    x2 = torch.randn(2, 128, 28, 28)
    x3 = torch.randn(2, 256, 14, 14)
    x4 = torch.randn(2, 512, 7, 7)
    y = model([x1, x2, x3, x4])
    print(y.shape)