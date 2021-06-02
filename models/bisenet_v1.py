import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d 
from torchvision import models
from .layers import ConvBnReLU


class SpatialPath(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        hidden_ch = 64
        self.conv_7x7 = ConvBnReLU(in_ch, hidden_ch, 7, 2, 3)
        self.conv_3x3_1 = ConvBnReLU(hidden_ch, hidden_ch, 3, 2, 1)
        self.conv_3x3_2 = ConvBnReLU(hidden_ch, hidden_ch, 3, 2, 1)
        self.conv_1x1 = ConvBnReLU(hidden_ch, out_ch, 1, 1, 0)

    def forward(self, x):           
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        return self.conv_1x1(x)


class Resnet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = models.resnet18()
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        down4 = self.layer1(x)                  # 1 / 4
        down8 = self.layer2(down4)              # 1 / 8
        down16 = self.layer3(down8)             # 1 / 16
        down32 = self.layer4(down16)            # 1 / 32

        return down16, down32


class ContextPath(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = Resnet18()

        self.arm16 = AttentionRefinmentModule(256, 128)
        self.arm32 = AttentionRefinmentModule(512, 128)

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnReLU(512, 128, 1, 1, 0)
        )

        self.up16 = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)

        self.refine16 = ConvBnReLU(128, 128, 3, 1, 1)
        self.refine32 = ConvBnReLU(128, 128, 3, 1, 1)


    def forward(self, x):
        down16, down32 = self.resnet(x)                 # 4x256x64x128, 4x512x32x64

        arm_down16 = self.arm16(down16)                 # 4x128x64x128
        arm_down32 = self.arm32(down32)                 # 4x128x32x64

        global_down32 = self.global_context(down32)     # 4x128x1x1
        global_down32 = F.interpolate(global_down32, size=down32.size()[2:], mode='bilinear', align_corners=True)   # 4x128x32x64
        
        arm_down32 += global_down32                         # 4x128x32x64
        arm_down32 = self.up32(arm_down32)                  # 4x128x64x128
        arm_down32 = self.refine32(arm_down32)              # 4x128x64x128

        arm_down16 += arm_down32                            # 4x128x64x128
        arm_down16 = self.up16(arm_down16)                  # 4x128x128x256
        arm_down16 = self.refine16(arm_down16)              # 4x128x128x256      

        return arm_down16, arm_down32
        
        
class AttentionRefinmentModule(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.conv_3x3 = ConvBnReLU(in_ch, out_ch, 3, 1, 1)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnReLU(out_ch, out_ch, 1, 1, 0, relu=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.attention(fm)
        return fm * fm_se


class FeatureFusionModule(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=1) -> None:
        super().__init__()
        self.conv_1x1 = ConvBnReLU(in_ch, out_ch, 1, 1, 0)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnReLU(out_ch, out_ch // reduction, 1, 1, 0, bn=False),
            ConvBnReLU(out_ch // reduction, out_ch, 1, 1, 0, bn=False, relu=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.attention(fm)

        return fm + fm * fm_se


class Head(nn.Module):
    def __init__(self, in_ch, n_classes, upscale_factor, is_aux=False) -> None:
        super().__init__()
        if is_aux:
            hidden_ch = 256
        else:
            hidden_ch = 64

        out_ch = n_classes * upscale_factor * upscale_factor
            
        self.conv_3x3 = ConvBnReLU(in_ch, hidden_ch, 3, 1, 1)
        self.conv_1x1 = nn.Conv2d(hidden_ch, out_ch, 1, 1, 0)

        self.upscale = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv_1x1(self.conv_3x3(x))
        return self.upscale(x)



class BiSeNetv1(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.num_aux_heads = 2

        self.context_path = ContextPath()
        self.spatial_path = SpatialPath(3, 128)
        self.ffm = FeatureFusionModule(256, 256)

        self.output_head = Head(256, n_classes, upscale_factor=8, is_aux=False)
        self.context16_head = Head(128, n_classes, upscale_factor=8, is_aux=True)
        self.context32_head = Head(128, n_classes, upscale_factor=16, is_aux=True)

    def forward(self, x):                                       # 4x3x1024x2048           
        spatial_out = self.spatial_path(x)                      # 4x128x128x256
        context16, context32 = self.context_path(x)             # 4x128x128x256, 4x128x64x128

        fm_fuse = self.ffm(spatial_out, context16)              # 4x256x128x256

        output = self.output_head(fm_fuse)                      # 4xn_classesx1024x2048

        if self.training:
            context_out16 = self.context16_head(context16)      # 4xn_classesx1024x2048
            context_out32 = self.context32_head(context32)      # 4xn_classesx1024x2048
            return output, context_out16, context_out32
        
        # return F.log_softmax(output, dim=1)                     # 4xn_classesx1024x2048       
        return output.argmax(dim=1)                             # 4x1024x1028     


    def init_weights(self, pretrained: str = None):
        for m in self.children():
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

        if pretrained:
            backbone_state_dict = torch.load(pretrained, map_location='cpu')
            state_dict = self.context_path.resnet.state_dict()

            for k, v in backbone_state_dict.items():
                if 'fc' in k: continue
                state_dict.update({k: v})

            self.context_path.resnet.load_state_dict(state_dict)


if __name__ == '__main__':
    from torch.nn import CrossEntropyLoss

    bisenet = BiSeNetv1(20, is_training=True)
    image = torch.randn((4, 3, 1024, 2048))
    labels = torch.randint(0, 19, (4, 1024, 2048)).long()
    criteria = CrossEntropyLoss()
    output, aux1, aux2 = bisenet(image)
    print(output.shape)
    print(labels.shape)
    loss = criteria(output, labels)
    print(loss.item())