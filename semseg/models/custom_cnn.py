import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.backbones import *
from semseg.models.heads import UPerHead


class CustomCNN(nn.Module):
    def __init__(self, backbone: str = 'ResNet-50', num_classes: int = 19):
        super().__init__()
        backbone, variant = backbone.split('-')
        self.backbone = eval(backbone)(variant)
        self.decode_head = UPerHead(self.backbone.channels, 256, num_classes)
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
    model = CustomCNN('ResNet-18', 19)
    model.init_pretrained('checkpoints/backbones/resnet/resnet18.pth')
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)