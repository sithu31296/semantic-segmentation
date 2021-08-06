import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .backbones.mit import MiT
from .heads.segformerhead import SegFormerHead


head_settings = {
    'B0': 256,        # head_dim
    'B1': 256,
    'B2': 768,
    'B3': 768,
    'B4': 768,
    'B5': 768
}


class SegFormer(nn.Module):
    def __init__(self, model_name: str = 'B0', num_classes: int = 19) -> None:
        super().__init__()
        self.backbone = MiT(model_name)
        self.decode_head = SegFormerHead(self.backbone.embed_dims, head_settings[model_name], num_classes)

    def init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y


if __name__ == '__main__':
    model = SegFormer('B0', 19)
    model.load_state_dict(torch.load('checkpoints/pretrained/segformer/segformer.b0.1024x1024.city.160k.pth', map_location='cpu')['state_dict'], strict=False)
    x = torch.zeros(1, 3, 512, 512)
    y = model(x)
    print(y.shape)
        

