import torch 
from torch import nn, Tensor


class HSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + 3) / 6


class HSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = HSigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return x * self.sigmoid(x)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()

        channels_per_group = c // self.groups

        # reshape
        x = x.view(b, self.groups, channels_per_group, h, w)

        x = torch.transpose(x, 1, 2).contiguous()
        out = x.view(b, -1, h, w)

        return out


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DYShiftMax(nn.Module):
    def __init__(self, c1, c2, 
        init_a=[0.0, 0.0],
        init_b=[0.0, 0.0],
        act_relu=True,
        g=None,
        reduction=4,
        expansion=False
    ):
        super().__init__()
        self.exp = 4 if act_relu else 2
        self.init_a = init_a
        self.init_b = init_b
        self.c2 = c2

        self.avg_pool = nn.Sequential(
            nn.Sequential(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        squeeze = _make_divisible(c1 // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(c1, squeeze),
            nn.ReLU(True),
            nn.Linear(squeeze, c2 * self.exp),
            HSigmoid()
        )

        g = g[1]
        if g != 1 and expansion:
            g = c1 // g
        
        gc = c1 // g
        index = torch.Tensor(range(c1)).view(1, c1, 1, 1)
        index = index.view(1, g, gc, 1, 1)
        indexgs = torch.split(index, [1, g-1], dim=1)
        indexgs = torch.cat([indexgs[1], indexgs[0]], dim=1)
        indexs = torch.split(indexgs, [1, gc-1], dim=2)
        indexs = torch.cat([indexs[1], indexs[0]], dim=2)
        self.index = indexs.view(c1).long()
        
        
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x_out = x
        
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, -1, 1, 1)
        y = (y - 0.5) * 4.0

        x2 = x_out[:, self.index, :, :]

        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.c2, dim=1)

            a1 = a1 + self.init_a[0]
            a2 = a2 + self.init_b[1]
            b1 = b1 + self.init_b[0]
            b2 = b2 + self.init_b[1]

            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2

            out = torch.max(z1, z2)

        elif self.exp == 2:
            a1, b1 = torch.split(y, self.c2, dim=1)
            a1 = a1 + self.init_a[0]
            b1 = b1 + self.init_b[0]
            out = x_out * a1 + x2 * b1

        return out


class SwishLinear(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(c1, c2),
            nn.BatchNorm1d(c2),
            HSwish()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class SpatialSepConvSF(nn.Module):
    def __init__(self, c1, outs, k, s):
        super().__init__()
        o1, o2 = outs
        self.conv = nn.Sequential(
            nn.Conv2d(c1, o1, (k, 1), (s, 1), (k//2, 0), bias=False),
            nn.BatchNorm2d(o1),
            nn.Conv2d(o1, o1*o2, (1, k), (1, s), (0, k//2), groups=o1, bias=False),
            nn.BatchNorm2d(o1*o2),
            ChannelShuffle(o1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Stem(nn.Module):
    def __init__(self, c1, c2, s, g=(4, 4)):
        super().__init__()
        self.stem = nn.Sequential(
            SpatialSepConvSF(c1, g, 3, s),
            nn.ReLU6(True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stem(x)


class DepthSpatialSepConv(nn.Module):
    def __init__(self, c1, expand, k, s):
        super().__init__()
        exp1, exp2 = expand
        ch = c1 * exp1
        c2 = c1 * exp1 * exp2

        self.conv = nn.Sequential(
            nn.Conv2d(c1, ch, (k, 1), (s, 1), (k//2, 0), groups=c1, bias=False),
            nn.BatchNorm2d(ch),
            nn.Conv2d(ch, c2, (1, k), (1, s), (0, k//2), groups=ch, bias=False),
            nn.BatchNorm2d(c2)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class PWConv(nn.Module):
    def __init__(self, c1, c2, g=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, groups=g[0], bias=False),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class MicroBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, 
        t1=(2, 2), 
        gs1=4, 
        groups_1x1=(1, 1), 
        dy=(2, 0, 1),
        r = 1,
        init_a=(1.0, 1.0),
        init_b=(0.0, 0.0),
    ):
        super().__init__()
        self.identity = s == 1 and c1 == c2
        y1, y2, y3 = dy
        _, g1, g2 = groups_1x1
        reduction = 8 * r
        ch2 = c1 * t1[0] * t1[1]

        if gs1[0] == 0:
            self.layers = nn.Sequential(
                DepthSpatialSepConv(c1, t1, k, s),
                DYShiftMax(ch2, ch2, init_a, init_b, True if y2 == 2 else False, gs1, reduction) if y2 > 0 else nn.ReLU6(True),
                ChannelShuffle(gs1[1]),
                ChannelShuffle(ch2 // 2) if y2 != 0 else nn.Sequential(),
                PWConv(ch2, c2, (g1, g2)),
                DYShiftMax(c2, c2, [1.0, 0.0], [0.0, 0.0], False, (g1, g2), reduction//2) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2),
                ChannelShuffle(c2 // 2) if c2%2 == 0 and y3 != 0 else nn.Sequential()
            )
        elif g2 == 0:
            self.layers = nn.Sequential(
                PWConv(c1, ch2, gs1),
                DYShiftMax(ch2, ch2, [1.0, 0.0], [0.0, 0.0], False, gs1, reduction) if y3 > 0 else nn.Sequential()
            )
        else:
            self.layers = nn.Sequential(
                PWConv(c1, ch2, gs1),
                DYShiftMax(ch2, ch2, init_a, init_b, True if y1 == 2 else False, gs1, reduction) if y1 > 0 else nn.ReLU6(True),
                ChannelShuffle(gs1[1]),
                DepthSpatialSepConv(ch2, (1, 1), k, s),
                nn.Sequential(),
                DYShiftMax(ch2, ch2, init_a, init_b, True if y2 == 2 else False, gs1, reduction, True) if y2 > 0 else nn.ReLU6(True),
                ChannelShuffle(ch2 // 4) if y1 != 0 and y2 != 0 else nn.Sequential() if y1 == 0 and y2 == 0 else ChannelShuffle(ch2 // 2),
                PWConv(ch2, c2, (g1, g2)),
                DYShiftMax(c2, c2, [1.0, 0.0], [0.0, 0.0], False, (g1, g2), reduction=reduction//2 if c2 < ch2 else reduction) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2),
                ChannelShuffle(c2 // 2) if y3 != 0 else nn.Sequential()
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.layers(x)
        if self.identity:
            out += identity
        return out


micronet_settings = {
    'M1': [
        6,              # stem_ch
        [3, 2],         # stem_groups
        960,            # out_ch
        [1.0, 1.0],     # init_a
        [0.0, 0.0],     # init_b
        [1, 2, 4, 7],   # out indices
        [8, 16, 32, 576],
        [
            #s, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r
            [2,   8, 3, 2, 2,  0,  6,   8,  2,  2, 2, 0, 1, 1],  #6->12(0, 0)->24  ->8(4,2)->8
            [2,  16, 3, 2, 2,  0,  8,  16,  4,  4, 2, 2, 1, 1], #8->16(0, 0)->32  ->16(4,4)->16
            [2,  16, 5, 2, 2,  0, 16,  16,  4,  4, 2, 2, 1, 1], #16->32(0, 0)->64  ->16(8,2)->16
            [1,  32, 5, 1, 6,  4,  4,  32,  4,  4, 2, 2, 1, 1], #16->16(2,8)->96 ->32(8,4)->32
            [2,  64, 5, 1, 6,  8,  8,  64,  8,  8, 2, 2, 1, 1], #32->32(2,16)->192 ->64(12,4)->64
            [1,  96, 3, 1, 6,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(3,16)->384 ->96(16,6)->96
            [1, 576, 3, 1, 6, 12, 12,   0,  0,  0, 2, 2, 1, 2], #96->96(4,24)->576
        ],
    ],  
    'M2': [
        8, 
        [4, 2],
        1024,
        [1.0, 1.0],
        [0.0, 0.0],
        [1, 3, 6, 9],
        [12, 24, 64, 768],
        [
            #s,  c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r
            [2,  12, 3, 2, 2,  0,  8,  12,  4,  4, 2, 0, 1, 1], #8->16(0, 0)->32  ->12(4,3)->12
            [2,  16, 3, 2, 2,  0, 12,  16,  4,  4, 2, 2, 1, 1], #12->24(0,0)->48  ->16(8, 2)->16
            [1,  24, 3, 2, 2,  0, 16,  24,  4,  4, 2, 2, 1, 1], #16->16(0, 0)->64  ->24(8,3)->24
            [2,  32, 5, 1, 6,  6,  6,  32,  4,  4, 2, 2, 1, 1], #24->24(2, 12)->144  ->32(16,2)->32
            [1,  32, 5, 1, 6,  8,  8,  32,  4,  4, 2, 2, 1, 2], #32->32(2,16)->192 ->32(16,2)->32
            [1,  64, 5, 1, 6,  8,  8,  64,  8,  8, 2, 2, 1, 2], #32->32(2,16)->192 ->64(12,4)->64
            [2,  96, 5, 1, 6,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(4,12)->384 ->96(16,5)->96
            [1, 128, 3, 1, 6, 12, 12, 128,  8,  8, 2, 2, 1, 2], #96->96(5,16)->576->128(16,8)->128
            [1, 768, 3, 1, 6, 16, 16,   0,  0,  0, 2, 2, 1, 2], #128->128(4,32)->768
        ],
    ],
    'M3': [
        12, 
        [4, 3],
        1024,
        [1.0, 0.5],
        [0.0, 0.5],
        [1, 3, 8, 12],
        [16, 24, 80, 864],
        [
            #s,  c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r
            [2,  16, 3, 2, 2,  0, 12,  16,  4,  4, 0, 2, 0, 1], #12->24(0, 0)->48  ->16(8,2)->16
            [2,  24, 3, 2, 2,  0, 16,  24,  4,  4, 0, 2, 0, 1], #16->32(0, 0)->64  ->24(8,3)->24
            [1,  24, 3, 2, 2,  0, 24,  24,  4,  4, 0, 2, 0, 1], #24->48(0, 0)->96  ->24(8,3)->24
            [2,  32, 5, 1, 6,  6,  6,  32,  4,  4, 0, 2, 0, 1], #24->24(2,12)->144  ->32(16,2)->32
            [1,  32, 5, 1, 6,  8,  8,  32,  4,  4, 0, 2, 0, 2], #32->32(2,16)->192 ->32(16,2)->32
            [1,  64, 5, 1, 6,  8,  8,  48,  8,  8, 0, 2, 0, 2], #32->32(2,16)->192 ->48(12,4)->48
            [1,  80, 5, 1, 6,  8,  8,  80,  8,  8, 0, 2, 0, 2], #48->48(3,16)->288 ->80(16,5)->80
            [1,  80, 5, 1, 6, 10, 10,  80,  8,  8, 0, 2, 0, 2], #80->80(4,20)->480->80(20,4)->80
            [2, 120, 5, 1, 6, 10, 10, 120, 10, 10, 0, 2, 0, 2], #80->80(4,20)->480->128(16,8)->128
            [1, 120, 5, 1, 6, 12, 12, 120, 10, 10, 0, 2, 0, 2], #120->128(4,32)->720->128(32,4)->120
            [1, 144, 3, 1, 6, 12, 12, 144, 12, 12, 0, 2, 0, 2], #120->128(4,32)->720->160(32,5)->144
            [1, 864, 3, 1, 6, 12, 12,   0,  0,  0, 0, 2, 0, 2], #144->144(5,32)->864
        ],
    ]
}


class MicroNet(nn.Module):
    def __init__(self, variant: str = 'M1') -> None:
        super().__init__()
        self.inplanes = 64

        assert variant in micronet_settings.keys(), f"MicroNet model name should be in {list(micronet_settings.keys())}"
        input_channel, stem_groups, _, init_a, init_b, out_indices, channels, cfgs = micronet_settings[variant]
        self.out_indices = out_indices
        self.channels = channels

        self.features = nn.ModuleList([Stem(3, input_channel, 2, stem_groups)])

        for s, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r in cfgs:
            self.features.append(MicroBlock(input_channel, c, ks, s, (c1, c2), (g1, g2), (c3, g3, g4), (y1, y2, y3), r, init_a, init_b))
            input_channel = c

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


if __name__ == '__main__':
    model = MicroNet('M3')
    # model.load_state_dict(torch.load('checkpoints/backbones/micronet/micronet-m2.pth', map_location='cpu'), strict=False)
    x = torch.zeros(1, 3, 224, 224)
    # outs = model(x)
    # for y in outs:
    #     print(y.shape)
    from fvcore.nn import flop_count_table, FlopCountAnalysis
    print(flop_count_table(FlopCountAnalysis(model, x)))