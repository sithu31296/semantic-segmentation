import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU6(True)
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, ch, squeeze_factor=4):
        super().__init__()
        squeeze_ch = _make_divisible(ch // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(ch, squeeze_ch, 1)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Conv2d(squeeze_ch, ch, 1)

    def _scale(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc2(self.relu(self.fc1(scale)))
        return F.hardsigmoid(scale, True)

    def forward(self, x: Tensor) -> Tensor:
        scale = self._scale(x)
        return scale * x


class InvertedResidualConfig:
    def __init__(self, c1, c2, k, expanded_ch, use_se) -> None:
        pass


class InvertedResidual(nn.Module):
    def __init__(self, c1, c2, s, expand_ratio):
        super().__init__()
        ch = int(round(c1 * expand_ratio))
        self.use_res_connect = s == 1 and c1 == c2

        layers = []

        if expand_ratio != 1:
            layers.append(ConvModule(c1, ch, 1))

        layers.extend([
            ConvModule(ch, ch, 3, s, 1, g=ch),
            nn.Conv2d(ch, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        ])

        self.conv = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


mobilenetv3_settings = {
    'S': [],
    'L': []
}


class MobileNetV3(nn.Module):
    def __init__(self, variant: str = None):
        super().__init__()
        self.out_indices = [3, 6, 13, 17]
        self.channels = [24, 32, 96, 320]
        input_channel = 32
        
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.features = nn.ModuleList([ConvModule(3, input_channel, 3, 2, 1)])

        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
    
    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


if __name__ == '__main__':
    model = MobileNetV3()
    # model.load_state_dict(torch.load('checkpoints/backbones/mobilenet_v2.pth', map_location='cpu'), strict=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    # outs = model(x)
    # for y in outs:
    #     print(y.shape)

    from fvcore.nn import flop_count_table, FlopCountAnalysis
    print(flop_count_table(FlopCountAnalysis(model, x)))