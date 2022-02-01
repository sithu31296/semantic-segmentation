import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.layers import ConvModule


class CondHead(nn.Module):
    def __init__(self, in_channel: int = 2048, channel: int = 512, num_classes: int = 19):
        super().__init__()
        self.num_classes = num_classes
        self.weight_num = channel * num_classes
        self.bias_num = num_classes

        self.conv = ConvModule(in_channel, channel, 1)
        self.dropout = nn.Dropout2d(0.1)

        self.guidance_project = nn.Conv2d(channel, num_classes, 1)
        self.filter_project = nn.Conv2d(channel*num_classes, self.weight_num + self.bias_num, 1, groups=num_classes)

    def forward(self, features) -> Tensor:
        x = self.dropout(self.conv(features[-1]))
        B, C, H, W = x.shape
        guidance_mask = self.guidance_project(x)
        cond_logit = guidance_mask
        
        key = x
        value = x
        guidance_mask = guidance_mask.softmax(dim=1).view(*guidance_mask.shape[:2], -1)
        key = key.view(B, C, -1).permute(0, 2, 1)

        cond_filters = torch.matmul(guidance_mask, key)
        cond_filters /= H * W
        cond_filters = cond_filters.view(B, -1, 1, 1)
        cond_filters = self.filter_project(cond_filters)
        cond_filters = cond_filters.view(B, -1)

        weight, bias = torch.split(cond_filters, [self.weight_num, self.bias_num], dim=1)
        weight = weight.reshape(B * self.num_classes, -1, 1, 1)
        bias = bias.reshape(B * self.num_classes)

        value = value.view(-1, H, W).unsqueeze(0)
        seg_logit = F.conv2d(value, weight, bias, 1, 0, groups=B).view(B, self.num_classes, H, W)
        
        if self.training:
            return cond_logit, seg_logit
        return seg_logit


if __name__ == '__main__':
    from semseg.models.backbones import ResNetD
    backbone = ResNetD('50')
    head = CondHead()
    x = torch.randn(2, 3, 224, 224)
    features = backbone(x)
    outs = head(features)
    for out in outs:
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        print(out.shape)