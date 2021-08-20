import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)        # use SyncBN in original

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.conv(x)))


class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding
    Code is mostly copied from mmsegmentation
    https://arxiv.org/abs/1807.10221
    pool_scales: Pooling scales used in PPM module applied on the last feature
    """
    def __init__(self, in_channels, ch=128, pool_scales=(1, 2, 3, 6), num_classes: int = 19):
        super().__init__()
        self.psp_modules = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(in_channels[-1], ch, 1)
            )
        for scale in pool_scales])
        self.bottleneck = ConvModule(in_channels[-1]+len(pool_scales)*ch, ch, 3, 1, 1)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels[:-1]: # skip the top layer
            self.lateral_convs.append(ConvModule(in_ch, ch, 1))
            self.fpn_convs.append(ConvModule(ch, ch, 3, 1, 1))

        self.fpn_bottleneck = ConvModule(len(in_channels)*ch, ch, 3, 1, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(ch, num_classes, 1)

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        for ppm in self.psp_modules:
            psp_outs.append(F.interpolate(ppm(x), size=x.shape[2:], mode='bilinear', align_corners=False))
        psp_outs = torch.cat(psp_outs, dim=1)
        return self.bottleneck(psp_outs)

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels-1, 0, -1):
            laterals[i-1] += F.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode='bilinear', align_corners=False)
        
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels-1)]
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels-1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=False)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.dropout(output)
        output = self.conv_seg(output)
        return output

