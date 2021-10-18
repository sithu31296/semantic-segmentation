import torch
from torch import nn, Tensor
from torch.nn import functional as F


class PSAP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        ch = c2 // 2
        self.conv_q_right = nn.Conv2d(c1, 1, 1, bias=False)
        self.conv_v_right = nn.Conv2d(c1, ch, 1, bias=False)
        self.conv_up = nn.Conv2d(ch, c2, 1, bias=False)

        self.conv_q_left = nn.Conv2d(c1, ch, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(c1, ch, 1, bias=False)

    def spatial_pool(self, x: Tensor) -> Tensor:
        input_x = self.conv_v_right(x)               # [B, C, H, W]
        context_mask = self.conv_q_right(x)         # [B, 1, H, W]
        B, C, _, _ = input_x.shape

        input_x = input_x.view(B, C, -1)
        context_mask = context_mask.view(B, 1, -1).softmax(dim=2)

        context = input_x @ context_mask.transpose(1, 2)
        context = self.conv_up(context.unsqueeze(-1)).sigmoid()
        x *= context
        return x

    def channel_pool(self, x: Tensor) -> Tensor:
        g_x = self.conv_q_left(x)
        B, C, H, W = g_x.shape
        avg_x = self.avg_pool(g_x).view(B, C, -1).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(B, C, -1)

        context = avg_x @ theta_x
        context = context.softmax(dim=2).view(B, 1, H, W).sigmoid()
        x *= context
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial_pool(x) + self.channel_pool(x)



class PSAS(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        ch = c2 // 2
        self.conv_q_right = nn.Conv2d(c1, 1, 1, bias=False)
        self.conv_v_right = nn.Conv2d(c1, ch, 1, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(ch, ch // 4, 1),
            nn.LayerNorm([ch // 4, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(ch // 4, c2, 1)
        )

        self.conv_q_left = nn.Conv2d(c1, ch, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(c1, ch, 1, bias=False)

    def spatial_pool(self, x: Tensor) -> Tensor:
        input_x = self.conv_v_right(x)               # [B, C, H, W]
        context_mask = self.conv_q_right(x)     # [B, 1, H, W]
        B, C, _, _ = input_x.shape

        input_x = input_x.view(B, C, -1)
        context_mask = context_mask.view(B, 1, -1).softmax(dim=2)

        context = input_x @ context_mask.transpose(1, 2)
        context = self.conv_up(context.unsqueeze(-1)).sigmoid()
        x *= context
        return x

    def channel_pool(self, x: Tensor) -> Tensor:
        g_x = self.conv_q_left(x)
        B, C, H, W = g_x.shape
        avg_x = self.avg_pool(g_x).view(B, C, -1).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(B, C, -1).softmax(dim=2)

        context = avg_x @ theta_x
        context = context.view(B, 1, H, W).sigmoid()
        x *= context
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.channel_pool(self.spatial_pool(x))


"""
PSA Module Usage

class BasicBlock(nn.Module):
    # 2 Layer No Expansion Block
    expansion: int = 1
    def __init__(self, c1, c2, s=1, downsample= None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.deattn = PSAS(c2, c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.deattn(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return F.relu(out)


class Bottleneck(nn.Module):
    # 3 Layer 4x Expansion Block
    expansion: int = 4
    def __init__(self, c1, c2, s=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, s, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.deattn = PSAP(c2, c2)
        self.conv3 = nn.Conv2d(c2, c2 * self.expansion, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(c2 * self.expansion)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.deattn(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return F.relu(out)


resnet_settings = {
    '18': [BasicBlock, [2, 2, 2, 2]],
    '34': [BasicBlock, [3, 4, 6, 3]],
    '50': [Bottleneck, [3, 4, 6, 3]],
    '101': [Bottleneck, [3, 4, 23, 3]],
    '152': [Bottleneck, [3, 8, 36, 3]]
}


class ResNet(nn.Module):
    def __init__(self, model_name: str = '50') -> None:
        super().__init__()
        assert model_name in resnet_settings.keys(), f"ResNet model name should be in {list(resnet_settings.keys())}"
        block, depths = resnet_settings[model_name]

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, depths[0], s=1)
        self.layer2 = self._make_layer(block, 128, depths[1], s=2)
        self.layer3 = self._make_layer(block, 256, depths[2], s=2)
        self.layer4 = self._make_layer(block, 512, depths[3], s=2)


    def _make_layer(self, block, planes, depth, s=1) -> nn.Sequential:
        downsample = None
        if s != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, s, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = nn.Sequential(
            block(self.inplanes, planes, s, downsample),
            *[block(planes * block.expansion, planes) for _ in range(1, depth)]
        )
        self.inplanes = planes * block.expansion
        return layers


    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))   # [1, 64, H/4, W/4]
        x1 = self.layer1(x)  # [1, 64/256, H/4, W/4]   
        x2 = self.layer2(x1)  # [1, 128/512, H/8, W/8]
        x3 = self.layer3(x2)  # [1, 256/1024, H/16, W/16]
        x4 = self.layer4(x3)  # [1, 512/2048, H/32, W/32]
        return x1, x2, x3, x4


if __name__ == '__main__':
    model = ResNet('18')
    x = torch.zeros(2, 3, 224, 224)
    outs = model(x)
    for y in outs:
        print(y.shape) 

"""
        
