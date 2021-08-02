import torch
from torch import nn, Tensor
from typing import Tuple
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)        # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    def __init__(self, dims, embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        self.linear_c1 = MLP(dims[0], embed_dim)
        self.linear_c2 = MLP(dims[1], embed_dim)
        self.linear_c3 = MLP(dims[2], embed_dim)
        self.linear_c4 = MLP(dims[3], embed_dim)

        self.linear_fuse = ConvModule(embed_dim*4, embed_dim, 1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        c1, c2, c3, c4 = inputs
        B = c1.shape[0]
        c1f = self.linear_c1(c1).permute(0, 2, 1).reshape(B, -1, c1.shape[2], c1.shape[3])
        
        c2f = self.linear_c2(c2).permute(0, 2, 1).reshape(B, -1, c2.shape[2], c2.shape[3])
        c2f = F.interpolate(c2f, size=c1.shape[2:], mode='bilinear', align_corners=False)

        c3f = self.linear_c3(c3).permute(0, 2, 1).reshape(B, -1, c3.shape[2], c3.shape[3])
        c3f = F.interpolate(c3f, size=c1.shape[2:], mode='bilinear', align_corners=False)

        c4f = self.linear_c4(c4).permute(0, 2, 1).reshape(B, -1, c4.shape[2], c4.shape[3])
        c4f = F.interpolate(c4f, size=c1.shape[2:], mode='bilinear', align_corners=False)

        c = self.linear_fuse(torch.cat([c4f, c3f, c2f, c1f], dim=1))
        c = self.dropout(c)
        c = self.linear_pred(c)
        return c