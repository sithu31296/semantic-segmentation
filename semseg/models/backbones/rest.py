import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.layers import DropPath


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio=1):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio 
        self.scale = (dim // head) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio+1, sr_ratio, sr_ratio//2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        self.apply_transform = head > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv2d(head, head, 1, 1)
            self.transform_norm = nn.InstanceNorm2d(head)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.sr_norm(x)
        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PA(nn.Module):        # positional embedding
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        
    def forward(self, x):
        return x * self.pa_conv(x).sigmoid()


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=64):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, 2, 1)
        self.norm = nn.BatchNorm2d(c2)
        self.pos = PA(c2)

    def forward(self, x: Tensor):
        x = self.pos(self.norm(self.conv(x)))
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class Stem(nn.Module):
    def __init__(self, c1=3, c2=64):
        super().__init__()
        ch = c2 // 2
        self.conv1 = nn.Conv2d(c1, ch, 3, 2, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(ch)
        self.conv3 = nn.Conv2d(ch, c2, 3, 2, 1, bias=False)
        self.act = nn.ReLU()
        self.pos = PA(c2)

    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        x = self.pos(self.conv3(x))
        _, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        return x, H, W


rest_settings = {
    'S': [[64, 128, 256, 512], [2, 2, 6, 2], 0.1],    # [embed_dims, depths, dpr]
    'B': [[96, 192, 384, 768], [2, 2, 6, 2], 0.2],
    'L': [[96, 192, 384, 768], [2, 2, 18, 2], 0.3]
}


class ResT(nn.Module):  
    def __init__(self, model_name: str = 'S') -> None:
        super().__init__()
        assert model_name in rest_settings.keys(), f"ResT model name should be in {list(rest_settings.keys())}"
        embed_dims, depths, drop_path_rate = rest_settings[model_name]
        self.channels = embed_dims
        self.stem = Stem(3, embed_dims[0])

        # patch_embed
        self.patch_embed_2 = PatchEmbed(embed_dims[0], embed_dims[1])
        self.patch_embed_3 = PatchEmbed(embed_dims[1], embed_dims[2])
        self.patch_embed_4 = PatchEmbed(embed_dims[2], embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # transformer encoder
        cur = 0
        self.stage1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur+i]) for i in range(depths[0])])

        cur += depths[0]
        self.stage2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur+i]) for i in range(depths[1])])

        cur += depths[1]
        self.stage3 = nn.ModuleList([Block(embed_dims[2], 4, 2, dpr[cur+i]) for i in range(depths[2])])
        
        cur += depths[2]
        self.stage4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur+i]) for i in range(depths[3])])
        self.norm = nn.LayerNorm(embed_dims[-1])


    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        
        # stage 1
        x, H, W = self.stem(x)
        for blk in self.stage1:
            x = blk(x, H, W)
        x1 = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # stage 2
        x, H, W = self.patch_embed_2(x1)
        for blk in self.stage2:
            x = blk(x, H, W)
        x2 = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # stage 3
        x, H, W = self.patch_embed_3(x2)
        for blk in self.stage3:
            x = blk(x, H, W)
        x3 = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # stage 4
        x, H, W = self.patch_embed_4(x3)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = self.norm(x)
        x4 = x.permute(0, 2, 1).reshape(B, -1, H, W)

        return x1, x2, x3, x4


if __name__ == '__main__':
    model = ResT('S')
    model.load_state_dict(torch.load('checkpoints/backbones/rest/rest_small.pth', map_location='cpu'), strict=False)
    x = torch.zeros(1, 3, 224, 224)
    outs = model(x)
    for y in outs:
        print(y.shape)