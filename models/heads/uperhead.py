import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Literal, Tuple


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s=s, p=p, bias=False)
        self.bn = nn.BatchNorm2d(c2)        # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class PPM(nn.ModuleList):
    def __init__(self, c1, c2, pool_scales):
        super().__init__()

        for pool_scale in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),
                ConvModule(c1, c2, 1)
            ))

    def forward(self, x: Tensor):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            ppm_out = F.interpolate(ppm_out, size=x.shape[2:], mode='bilinear', align_corners=False)
            ppm_outs.append(ppm_out)
        return ppm_outs


class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221

    pool_scales: Pooling scales used in PPM module applied on the last feature
    """
    def __init__(self, dims, channels=128, pool_scales=(1, 2, 3, 6), num_classes: int = 19):
        super().__init__()
        # PSP Module
        self.psp_modules = PPM(dims[-1], channels, pool_scales)
        self.bottleneck = ConvModule(dims[-1]+len(pool_scales)*channels, channels, 3, 1, 1)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in dims[:-1]: # skip the top layer
            self.lateral_convs.append(ConvModule(in_ch, channels, 1))
            self.fpn_convs.append(ConvModule(channels, channels, 3, 1, 1))

        self.fpn_bottleneck = ConvModule(len(dims)*channels, channels, 3, 1, 1)

        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        laterals = []

        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))

        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        
        laterals.append(self.bottleneck(psp_outs))

        used_backbone_levels = len(laterals)

        for i in range(used_backbone_levels-1, 0, -1):
            laterals[i-1] += F.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode='bilinear', align_corners=False)
        
        fpn_outs = []

        for i in range(used_backbone_levels-1):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))

        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels-1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=False)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.dropout(output)
        output = self.conv_seg(output)
        return output

