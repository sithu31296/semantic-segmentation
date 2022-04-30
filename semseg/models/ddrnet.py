import torch
from torch import nn, Tensor
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, c1, c2, s=1, downsample= None, no_relu=False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return out if self.no_relu else F.relu(out)


class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self, c1, c2, s=1, downsample=None, no_relu=False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, s, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c2 * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2 * self.expansion)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return out if self.no_relu else F.relu(out)


class ConvBN(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2)
        )


class Conv2BN(nn.Sequential):
    def __init__(self, c1, ch, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, ch, k, s, p, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2)
        )


class Stem(nn.Sequential):
    def __init__(self, c1, c2):
        super().__init__(
            nn.Conv2d(c1, c2, 3, 2, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2, c2, 3, 2, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class Scale(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.AvgPool2d(k, s, p),
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, 1, bias=False)
        )


class ScaleLast(nn.Sequential):
    def __init__(self, c1, c2, k):
        super().__init__(
            nn.AdaptiveAvgPool2d(k),
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, 1, bias=False)
        )


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, k, s, p, bias=False)
        )


class DAPPM(nn.Module):
    def __init__(self, c1, ch, c2):
        super().__init__()
        self.scale1 = Scale(c1, ch, 5, 2, 2)
        self.scale2 = Scale(c1, ch, 9, 4, 4)
        self.scale3 = Scale(c1, ch, 17, 8, 8)
        self.scale4 = ScaleLast(c1, ch, 1)
        self.scale0 = ConvModule(c1, ch, 1)
        self.process1 = ConvModule(ch, ch, 3, 1, 1)
        self.process2 = ConvModule(ch, ch, 3, 1, 1)
        self.process3 = ConvModule(ch, ch, 3, 1, 1)
        self.process4 = ConvModule(ch, ch, 3, 1, 1)
        self.compression = ConvModule(ch*5, c2, 1)
        self.shortcut = ConvModule(c1, c2, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = [self.scale0(x)]
        outs.append(self.process1((F.interpolate(self.scale1(x), size=x.shape[-2:], mode='bilinear', align_corners=False) + outs[-1])))
        outs.append(self.process2((F.interpolate(self.scale2(x), size=x.shape[-2:], mode='bilinear', align_corners=False) + outs[-1])))
        outs.append(self.process3((F.interpolate(self.scale3(x), size=x.shape[-2:], mode='bilinear', align_corners=False) + outs[-1])))
        outs.append(self.process4((F.interpolate(self.scale4(x), size=x.shape[-2:], mode='bilinear', align_corners=False) + outs[-1])))
        out = self.compression(torch.cat(outs, dim=1)) + self.shortcut(x)
        return out


class SegHead(nn.Module):
    def __init__(self, c1, ch, c2, scale_factor=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv1 = nn.Conv2d(c1, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, c2, 1)
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(x)))

        if self.scale_factor is not None:
            H, W = x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


class DDRNet(nn.Module):
    def __init__(self, backbone: str = None, num_classes: int = 19) -> None:
        super().__init__()
        planes, spp_planes, head_planes = [32, 64, 128, 256, 512], 128, 64

        self.conv1 = Stem(3, planes[0])

        self.layer1 = self._make_layer(BasicBlock, planes[0], planes[0], 2)
        self.layer2 = self._make_layer(BasicBlock, planes[0], planes[1], 2, 2)
        self.layer3 = self._make_layer(BasicBlock, planes[1], planes[2], 2, 2)
        self.layer4 = self._make_layer(BasicBlock, planes[2], planes[3], 2, 2)
        self.layer5 = self._make_layer(Bottleneck, planes[3], planes[3], 1)

        self.layer3_ = self._make_layer(BasicBlock, planes[1], planes[1], 2)
        self.layer4_ = self._make_layer(BasicBlock, planes[1], planes[1], 2)
        self.layer5_ = self._make_layer(Bottleneck, planes[1], planes[1], 1)

        self.compression3 = ConvBN(planes[2], planes[1], 1)
        self.compression4 = ConvBN(planes[3], planes[1], 1)

        self.down3 = ConvBN(planes[1], planes[2], 3, 2, 1)
        self.down4 = Conv2BN(planes[1], planes[2], planes[3], 3, 2, 1)

        self.spp = DAPPM(planes[-1], spp_planes, planes[2])
        self.seghead_extra = SegHead(planes[1], head_planes, num_classes, 8)
        self.final_layer = SegHead(planes[2], head_planes, num_classes, 8)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu')['model'], strict=False)

    def _make_layer(self, block, inplanes, planes, depths, s=1) -> nn.Sequential:
        downsample = None
        if inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, s, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(inplanes, planes, s, downsample)]
        inplanes = planes * block.expansion

        for i in range(1, depths):
            if i == depths - 1:
                layers.append(block(inplanes, planes, no_relu=True))
            else:
                layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2] // 8, x.shape[-1] // 8
        layers = []

        x = self.conv1(x)   
        x = self.layer1(x) 
        layers.append(x) 

        x = self.layer2(F.relu(x))  
        layers.append(x)

        x = self.layer3(F.relu(x))  
        layers.append(x)
        x_ = self.layer3_(F.relu(layers[1]))
        x = x + self.down3(F.relu(x_))
        x_ = x_ + F.interpolate(self.compression3(F.relu(layers[2])), size=(H, W), mode='bilinear', align_corners=False)

        if self.training: x_aux = self.seghead_extra(x_)

        x = self.layer4(F.relu(x))   
        layers.append(x)
        x_ = self.layer4_(F.relu(x_))
        x = x + self.down4(F.relu(x_))
        x_ = x_ + F.interpolate(self.compression4(F.relu(layers[3])), size=(H, W), mode='bilinear', align_corners=False)

        x_ = self.layer5_(F.relu(x_))
        x = F.interpolate(self.spp(self.layer5(F.relu(x))), size=(H, W), mode='bilinear', align_corners=False)
        x_ = self.final_layer(x + x_)

        return (x_, x_aux) if self.training else x_


if __name__ == '__main__':
    model = DDRNet()
    # # model.init_pretrained('checkpoints/backbones/ddrnet/ddrnet_23slim_imagenet.pth')
    # model.load_state_dict(torch.load('checkpoints/pretrained/ddrnet/ddrnet_23slim_city.pth', map_location='cpu'))
    x = torch.zeros(2, 3, 224, 224)
    outs = model(x)
    for y in outs:
        print(y.shape)