import torch
from torch import nn, Tensor


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU6(True)
        )


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


mobilenetv2_settings = {
    '1.0': []
}


class MobileNetV2(nn.Module):
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
    model = MobileNetV2()
    # model.load_state_dict(torch.load('checkpoints/backbones/mobilenet_v2.pth', map_location='cpu'), strict=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    # outs = model(x)
    # for y in outs:
    #     print(y.shape)

    from fvcore.nn import flop_count_table, FlopCountAnalysis
    print(flop_count_table(FlopCountAnalysis(model, x)))