from os.path import expanduser
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBnReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, d=1, g=1, bn=True, relu=True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, d, g, bias=False)
        self.has_bn = bn
        self.has_relu = relu
        if self.has_bn: self.bn = nn.BatchNorm2d(out_ch)
        if self.has_relu: self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn: x = self.bn(x)
        if self.has_relu: x = self.relu(x)
        return x