import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k//2, bias=False)
        self.norm = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU6(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.norm(self.conv(x)))


def get_link(layer, base_ch, growth_rate):
    if layer == 0:
        return base_ch, 0, []

    link = []
    out_channels = growth_rate

    for i in range(10):
        dv = 2 ** i
        if layer % dv == 0:
            link.append(layer - dv)

            if i > 0: out_channels *= 1.7

    out_channels = int((out_channels + 1) / 2) * 2
    in_channels = 0

    for i in link:
        ch, _, _ = get_link(i, base_ch, growth_rate)
        in_channels += ch

    return out_channels, in_channels, link


class HarDBlock(nn.Module):
    def __init__(self, c1, growth_rate, n_layers):
        super().__init__()
        self.links = []
        layers = []
        self.out_channels = 0

        for i in range(n_layers):
            out_ch, in_ch, link = get_link(i+1, c1, growth_rate)
            self.links.append(link)

            layers.append(ConvModule(in_ch, out_ch))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += out_ch

        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        layers = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []

            for i in link:
                tin.append(layers[i])

            if len(tin) > 1:
                x = torch.cat(tin, dim=1)
            else:
                x = tin[0]

            out = self.layers[layer](x)
            layers.append(out)

        t = len(layers)
        outs = []
        for i in range(t):
            if (i == t - 1) or (i % 2 == 1):
                outs.append(layers[i])

        out = torch.cat(outs, dim=1)
        return out


class FCHarDNet(nn.Module):
    def __init__(self, backbone: str = None, num_classes: int = 19) -> None:
        super().__init__()
        first_ch, ch_list, gr, n_layers = [16, 24, 32, 48], [64, 96, 160, 224, 320], [10, 16, 18, 24, 32], [4, 4, 8, 8, 8]

        self.base = nn.ModuleList([])

        # stem
        self.base.append(ConvModule(3, first_ch[0], 3, 2))
        self.base.append(ConvModule(first_ch[0], first_ch[1], 3))
        self.base.append(ConvModule(first_ch[1], first_ch[2], 3, 2))
        self.base.append(ConvModule(first_ch[2], first_ch[3], 3))

        self.shortcut_layers = []
        skip_connection_channel_counts = []
        ch = first_ch[-1]

        for i in range(len(n_layers)):
            blk = HarDBlock(ch, gr[i], n_layers[i])
            ch = blk.out_channels

            skip_connection_channel_counts.append(ch)
            self.base.append(blk)

            if i < len(n_layers) - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvModule(ch, ch_list[i], k=1))
            ch = ch_list[i]
            
            if i < len(n_layers) - 1:
                self.base.append(nn.AvgPool2d(2, 2))

        prev_block_channels = ch
        self.n_blocks = len(n_layers) - 1

        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])

        for i in range(self.n_blocks-1, -1, -1):
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            blk = HarDBlock(cur_channels_count // 2, gr[i], n_layers[i])
            prev_block_channels = blk.out_channels
            
            self.conv1x1_up.append(ConvModule(cur_channels_count, cur_channels_count//2, 1))
            self.denseBlocksUp.append(blk)

        self.finalConv = nn.Conv2d(prev_block_channels, num_classes, 1, 1, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2:]
        skip_connections = []
        for i, layer in enumerate(self.base):
            x = layer(x)
            if i in self.shortcut_layers:
                skip_connections.append(x)

        out = x

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = F.interpolate(out, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            out = torch.cat([out, skip], dim=1)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        return out


if __name__ == '__main__':
    model = FCHarDNet()
    # model.init_pretrained('checkpoints/backbones/hardnet/hardnet_70.pth')
    # model.load_state_dict(torch.load('checkpoints/pretrained/hardnet/hardnet70_cityscapes.pth', map_location='cpu'))
    x = torch.zeros(1, 3, 224, 224)
    outs = model(x)
    print(outs.shape)
