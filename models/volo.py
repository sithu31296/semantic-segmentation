import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple



volo_settings = {
    'D1': [],
    'D2': [],
    'D3': [],
    'D4': [],
    'D5': []
}


class VOLO(nn.Module):
    def __init__(self, model_name: str = 'D1', num_classes: int = 19, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in volo_settings.keys(), f"SegFormer model name should be in {list(volo_settings.keys())}"
        dims, layers = volo_settings[model_name]

    def init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


if __name__ == '__main__':
    model = VOLO('D1', image_size=224)
    x = torch.zeros(1, 3, 768, 768)
    y = model(x)
    print(y.shape)
        

