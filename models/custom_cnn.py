import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .backbones import ResNet
from .heads import UPerHead


head_settings = {
    '18': [[64, 128, 256, 512], 128],
    '34': [[64, 128, 256, 512], 256],
    '50': [[256, 512, 1024, 2048], 512],
    '101': [[256, 512, 1024, 2048], 512],
    '152': [[256, 512, 1024, 2048], 768],
}


class CustomCNN(nn.Module):
    def __init__(self, variant: str = '50', num_classes: int = 19):
        super().__init__()
        in_channels, channels = head_settings[variant]
        self.backbone = ResNet(variant)
        self.decode_head = UPerHead(in_channels, channels, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y

    
if __name__ == '__main__':
    model = CustomCNN('18', 19)
    model.init_pretrained('checkpoints/backbones/resnet/resnet18.pth')
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)