import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_ch=3, dim=96, type='pool') -> None:
        super().__init__()
        self.patch_size = patch_size
        self.type = type
        self.dim = dim
        
        if type == 'conv':
            self.proj = nn.Conv2d(in_ch, dim, patch_size, patch_size, groups=patch_size*patch_size)
        else:
            self.proj = nn.ModuleList([
                nn.MaxPool2d(patch_size, patch_size),
                nn.AvgPool2d(patch_size, patch_size)
            ])
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        if W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - W % self.patch_size))
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size - H % self.patch_size))

        if self.type == 'conv':
            x = self.proj(x)
        else:
            x = 0.5 * (self.proj[0](x) + self.proj[1](x))
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.dim, Wh, Ww)
        return x


class LawinAttn(nn.Module):
    def __init__(self, in_ch=512, head=4, patch_size=8, reduction=2) -> None:
        super().__init__()
        self.head = head

        self.position_mixing = nn.ModuleList([
            nn.Linear(patch_size * patch_size, patch_size * patch_size)
        for _ in range(self.head)])

        self.inter_channels = max(in_ch // reduction, 1)
        self.g = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch)
        )


    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        B, C, H, W = context.shape
        context = context.reshape(B, C, -1)
        context_mlp = []

        for i, pm in enumerate(self.position_mixing):
            context_crt = context[:, (C//self.head)*i:(C//self.head)*(i+1), :]
            context_mlp.append(pm(context_crt))

        context_mlp = torch.cat(context_mlp, dim=1)
        context = context + context_mlp
        context = context.reshape(B, C, H, W)

        g_x = self.g(context).view(B, self.inter_channels, -1)
        g_x = rearrange(g_x, "b (h dim) n -> (b h) dim n", h=self.head)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(query).view(B, self.inter_channels, -1)
        theta_x = rearrange(theta_x, "b (h dim) n -> (b h) dim n", h=self.head)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(context).view(B, self.inter_channels, -1)
        phi_x = rearrange(phi_x, "b (h dim) n -> (b h) dim n", h=self.head)

        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)

        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, "(b h) n dim -> b n (h dim)", h=self.head)
        y = y.permute(0, 2, 1).contiguous().reshape(B, self.inter_channels, *query.shape[-2:])

        output = query + self.conv_out(y)
        return output
        

class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)        # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class LawinHead(nn.Module):
    def __init__(self, in_channels: list, embed_dim=512, num_classes=19) -> None:
        super().__init__()
        for i, dim in enumerate(in_channels):
            self.add_module(f"linear_c{i+1}", MLP(dim, 48 if i == 0 else embed_dim))

        self.lawin_8 = LawinAttn(embed_dim, 64)
        self.lawin_4 = LawinAttn(embed_dim, 16)
        self.lawin_2 = LawinAttn(embed_dim, 4)
        self.ds_8 = PatchEmbed(8, embed_dim, embed_dim)
        self.ds_4 = PatchEmbed(4, embed_dim, embed_dim)
        self.ds_2 = PatchEmbed(2, embed_dim, embed_dim)
    
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(embed_dim, embed_dim)
        )
        self.linear_fuse = ConvModule(embed_dim*3, embed_dim)
        self.short_path = ConvModule(embed_dim, embed_dim)
        self.cat = ConvModule(embed_dim*5, embed_dim)

        self.low_level_fuse = ConvModule(embed_dim+48, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)
    
    def get_lawin_att_feats(self, x: Tensor, patch_size: int):
        _, _, H, W = x.shape
        query = F.unfold(x, patch_size, stride=patch_size)
        query = rearrange(query, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size, pw=patch_size, nh=H//patch_size, nw=W//patch_size)
        outs = []

        for r in [8, 4, 2]:
            context = F.unfold(x, patch_size*r, stride=patch_size, padding=int((r-1)/2*patch_size))
            context = rearrange(context, "b (c ph pw) (nh nw) -> (b nh nw) c ph pw", ph=patch_size*r, pw=patch_size*r, nh=H//patch_size, nw=W//patch_size)
            context = getattr(self, f"ds_{r}")(context)
            output = getattr(self, f"lawin_{r}")(query, context)
            output = rearrange(output, "(b nh nw) c ph pw -> b c (nh ph) (nw pw)", ph=patch_size, pw=patch_size, nh=H//patch_size, nw=W//patch_size)
            outs.append(output)
        return outs


    def forward(self, features):
        B, _, H, W = features[1].shape
        outs = [self.linear_c2(features[1]).permute(0, 2, 1).reshape(B, -1, *features[1].shape[-2:])]

        for i, feature in enumerate(features[2:]):
            cf = eval(f"self.linear_c{i+3}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        feat = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        B, _, H, W = feat.shape

        ## Lawin attention spatial pyramid pooling
        feat_short = self.short_path(feat)
        feat_pool = F.interpolate(self.image_pool(feat), size=(H, W), mode='bilinear', align_corners=False)
        feat_lawin = self.get_lawin_att_feats(feat, 8)
        output = self.cat(torch.cat([feat_short, feat_pool, *feat_lawin], dim=1))

        ## Low-level feature enhancement
        c1 = self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])
        output = F.interpolate(output, size=features[0].shape[-2:], mode='bilinear', align_corners=False)
        fused = self.low_level_fuse(torch.cat([output, c1], dim=1))

        seg = self.linear_pred(self.dropout(fused))
        return seg
