import torch
import torch.nn as nn

from .layers import ConvBnReLU

class DetailBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.S1 = nn.Sequential(
            ConvBnReLU(3, 64, 3, 2, 1),
            ConvBnReLU(64, 64, 3, 1, 1)
        )
        self.S2 = nn.Sequential(
            ConvBnReLU(64, 64, 3, 2, 1),
            ConvBnReLU(64, 64, 3, 1, 1),
            ConvBnReLU(64, 64, 3, 1, 1)
        )
        self.S3 = nn.Sequential(
            ConvBnReLU(64, 128, 3, 2, 1),
            ConvBnReLU(128, 128, 3, 1, 1),
            ConvBnReLU(128, 128, 3, 1, 1)
        )

    
    def forward(self, x):
        return self.S3(self.S2(self.S1(x)))


class StemBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_3x3 = ConvBnReLU(3, 16, 3, 2, 1)
        self.left = nn.Sequential(
            ConvBnReLU(16, 8, 1, 1, 0),             # 8 is not mentioned in the paper
            ConvBnReLU(8, 16, 3, 2, 1)
        )
        self.right = nn.MaxPool2d(3, 2, 1, ceil_mode=False)

        self.fuse = ConvBnReLU(32, 16, 3, 1, 1)

    def forward(self, x):
        x = self.conv_3x3(x)
        x_left = self.left(x)
        x_right = self.right(x)
        y = torch.cat([x_left, x_right], dim=1)
        return self.fuse(y)


class ContextEmbeddingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(128),
            ConvBnReLU(128, 128, 1, 1, 0)           # in paper, there is broadcast term
        )
        self.conv = ConvBnReLU(128, 128, 3, 1, 1)   # in paper, normal Conv not Conv+BN+ReLU

    def forward(self, x):
        y = self.inner(x)
        out = x + y
        return self.conv(out)


class GatherExpansionLayerv1(nn.Module):
    def __init__(self, in_ch, out_ch, e=6) -> None:
        super().__init__()

        self.inner = nn.Sequential(
            ConvBnReLU(in_ch, in_ch, 3, 1, 1),
            ConvBnReLU(in_ch, in_ch*e, 3, 1, 1, g=in_ch),       # no groups and relu in paper
            ConvBnReLU(in_ch*e, out_ch, 1, 1, 0, relu=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.inner(x)
        out = x + y
        return self.relu(out)


class GatherExpansionLayerv2(nn.Module):
    def __init__(self, in_ch, out_ch, e=6) -> None:
        super().__init__()

        self.inner = nn.Sequential(
            ConvBnReLU(in_ch, in_ch, 3, 1, 1),
            ConvBnReLU(in_ch, in_ch*e, 3, 2, 1, g=in_ch, relu=False),       # no groups in paper
            ConvBnReLU(in_ch*e, in_ch*e, 3, 1, 1, g=in_ch*e),               # no relu in paper
            ConvBnReLU(in_ch*e, out_ch, 1, 1, 0, relu=False)
        )

        self.inner[-1].bn.last_bn = True

        self.outer = nn.Sequential(
            ConvBnReLU(in_ch, in_ch, 3, 2, 1, g=in_ch, relu=False),
            ConvBnReLU(in_ch, out_ch, 1, 1, 0, relu=False)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.inner(x)
        x2 = self.outer(x)
        out = x1 + x2
        return self.relu(out)


class SemanticBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GatherExpansionLayerv2(16, 32),
            GatherExpansionLayerv1(32, 32)
        )
        self.S4 = nn.Sequential(
            GatherExpansionLayerv2(32, 64),
            GatherExpansionLayerv1(64, 64)
        )
        self.S5_1 = nn.Sequential(
            GatherExpansionLayerv2(64, 128),
            GatherExpansionLayerv1(128, 128),
            GatherExpansionLayerv1(128, 128),
            GatherExpansionLayerv1(128, 128)
        )
        self.S5_2 = ContextEmbeddingBlock()

    def forward(self, x):
        x2 = self.S1S2(x)
        x3 = self.S3(x2)
        x4 = self.S4(x3)
        x5_1 = self.S5_1(x4)
        x5_2 = self.S5_2(x5_1)
        return x2, x3, x4, x5_1, x5_2


class AggregationLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left1 = nn.Sequential(
            ConvBnReLU(128, 128, 3, 1, 1, g=128, relu=False),
            nn.Conv2d(128, 128, 1, 1, 0, bias=False)
        )
        self.left2 = nn.Sequential(
            ConvBnReLU(128, 128, 3, 2, 1, relu=False),
            nn.AvgPool2d(3, 2, 1, ceil_mode=False)
        )

        self.right1 = nn.Sequential(
            ConvBnReLU(128, 128, 3, 1, 1, relu=False),
            nn.Upsample(scale_factor=4),
            nn.Sigmoid()
        )
        self.right2 = nn.Sequential(
            ConvBnReLU(128, 128, 3, 1, 1, g=128, relu=False),
            nn.Conv2d(128, 128, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.up = nn.Upsample(scale_factor=4)           # not shown in paper but needed
        self.conv = ConvBnReLU(128, 128, 3, 1, 1)       # no relu in paper

    
    def forward(self, x_d, x_s):
        x1 = self.left1(x_d)
        x2 = self.left2(x_d)
        x3 = self.right1(x_s)
        x4 = self.right2(x_s)

        left = x1 * x3
        right = x2 * x4
        right = self.up(right)
        out = left + right

        return self.conv(out)



## Check this not mentioned like that in paper
class SegHead(nn.Module):
    def __init__(self, in_ch, mid_ch, n_classes, upscale_factor=8, is_aux=True) -> None:
        super().__init__()

        out_ch = n_classes * upscale_factor * upscale_factor
        
        self.conv_3x3 = ConvBnReLU(in_ch, mid_ch, 3, 1, 1)
        self.drop = nn.Dropout(0.1)

        if is_aux:
            self.conv_out = nn.Sequential(
                ConvBnReLU(mid_ch, upscale_factor * upscale_factor, 3, 1, 1),
                nn.Conv2d(upscale_factor*upscale_factor, out_ch, 1, 1, 0),
                nn.PixelShuffle(upscale_factor)
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(mid_ch, out_ch, 1, 1, 0),
                nn.PixelShuffle(upscale_factor)
            )
    
    def forward(self, x):
        out = self.conv_3x3(x)
        out = self.drop(out)
        return self.conv_out(out)


class BiSeNetv2(nn.Module):
    def __init__(self, n_classes, is_training=False) -> None:
        super().__init__()

        self.is_training = is_training

        self.detail_branch = DetailBranch()
        self.semantic_branch = SemanticBranch()
        self.aggregation_layer = AggregationLayer()

        self.output_head = SegHead(128, 1024, n_classes, upscale_factor=8, is_aux=False)

        if is_training:
            self.aux2_head = SegHead(16, 128, n_classes, upscale_factor=4)
            self.aux3_head = SegHead(32, 128, n_classes, upscale_factor=8)
            self.aux4_head = SegHead(64, 128, n_classes, upscale_factor=16)
            self.aux5_head = SegHead(128, 128, n_classes, upscale_factor=32)

        self._init_weights()

    
    def _init_weights(self, pretrained: str = None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, 'last_bn') and m.last_bn:
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    
    def forward(self, x):
        x_d = self.detail_branch(x)
        aux2, aux3, aux4, aux5, x_s = self.semantic_branch(x)

        output = self.aggregation_layer(x_d, x_s)
        output = self.output_head(output)

        if self.is_training:
            aux2 = self.aux2_head(aux2)
            aux3 = self.aux3_head(aux3)
            aux4 = self.aux4_head(aux4)
            aux5 = self.aux5_head(aux5)
            return output, aux2, aux3, aux4, aux5
        
        return output.argmax(dim=1)


if __name__ == '__main__':
    bisenet = BiSeNetv2(20, is_training=True)
    image = torch.randn((4, 3, 1024, 2048))
    output, aux2, aux3, aux4, aux5 = bisenet(image)
    print(output.shape, aux2.shape, aux3.shape, aux4.shape, aux5.shape)