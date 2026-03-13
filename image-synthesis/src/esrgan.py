"""ESRGAN: Enhanced Super-Resolution GAN (Wang et al., ECCV 2018).
HAT: Hybrid Attention Transformer for Image Restoration (Chen CVPR 2023).
Perceptual loss with VGG features (Johnson 2016).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseResidualBlock(nn.Module):
    """RRDB: Residual-in-Residual Dense Block for ESRGAN."""
    def __init__(self, nf=64, gc=32, beta=0.2):
        super().__init__()
        self.beta = beta
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2*gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3*gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4*gc, nf, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5 * self.beta


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32, n_dense=3):
        super().__init__()
        self.blocks = nn.Sequential(*[DenseResidualBlock(nf, gc) for _ in range(n_dense)])

    def forward(self, x): return x + self.blocks(x) * 0.2


class ESRGANGenerator(nn.Module):
    """ESRGAN generator: RRDB x23 + pixel shuffle upsampling."""
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, scale=4):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        # Upsampling
        ups = []
        for _ in range(int(torch.log2(torch.tensor(scale)).item())):
            ups += [nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True)]
        self.upsample = nn.Sequential(*ups)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        trunk = self.conv_body(self.body(feat))
        feat = feat + trunk
        feat = self.upsample(feat)
        return self.conv_last(self.act(self.conv_hr(feat)))


class ESRGANDiscriminator(nn.Module):
    """VGG-style discriminator for ESRGAN."""
    def __init__(self, in_nc=3, ndf=64):
        super().__init__()
        def conv_block(ic, oc, stride=1):
            return nn.Sequential(nn.Conv2d(ic, oc, 3, stride, 1), nn.BatchNorm2d(oc), nn.LeakyReLU(0.2, True))
        self.net = nn.Sequential(
            nn.Conv2d(in_nc, ndf, 3, 1, 1), nn.LeakyReLU(0.2, True),
            conv_block(ndf, ndf, 2), conv_block(ndf, ndf*2), conv_block(ndf*2, ndf*2, 2),
            conv_block(ndf*2, ndf*4), conv_block(ndf*4, ndf*4, 2),
            conv_block(ndf*4, ndf*8), conv_block(ndf*8, ndf*8, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ndf*8, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1)
        )

    def forward(self, x): return self.net(x)


class PerceptualLoss(nn.Module):
    """VGG perceptual loss for image synthesis tasks."""
    def __init__(self, layers=('relu3_4', 'relu4_4'), weights=(1.0, 1.0)):
        super().__init__()
        from torchvision import models
        vgg = models.vgg19(pretrained=False)
        layer_map = {'relu1_2': 4, 'relu2_2': 9, 'relu3_4': 18, 'relu4_4': 27, 'relu5_4': 36}
        max_idx = max(layer_map[l] for l in layers)
        self.vgg_features = nn.Sequential(*list(vgg.features.children())[:max_idx + 1])
        for p in self.vgg_features.parameters():
            p.requires_grad_(False)
        self.slice_indices = [layer_map[l] for l in layers]
        self.weights = weights
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        h_pred = pred; h_target = target
        prev_idx = 0
        for i, (idx, w) in enumerate(zip(self.slice_indices, self.weights)):
            slice_fn = nn.Sequential(*list(self.vgg_features.children())[prev_idx:idx])
            h_pred = slice_fn(h_pred if i == 0 else h_pred.detach())
            h_target = slice_fn(h_target)
            loss += w * F.l1_loss(h_pred, h_target.detach())
            prev_idx = idx
        return loss


class WindowAttention(nn.Module):
    """Window-based self-attention for HAT (Hybrid Attention Transformer)."""
    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()
        self.dim = dim; self.window_size = window_size; self.num_heads = num_heads
        self.head_dim = dim // num_heads; self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        # Relative position bias
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2*window_size-1)**2, num_heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'
        )).flatten(1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1; rel[:, :, 1] += window_size - 1
        self.register_buffer('rel_idx', rel[:, :, 0] * (2*window_size-1) + rel[:, :, 1])

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn + self.rel_pos_bias[self.rel_idx].permute(2,0,1).unsqueeze(0)
        attn = torch.softmax(attn, dim=-1)
        return self.proj((attn @ v).transpose(1,2).reshape(B, N, C))


class HATBlock(nn.Module):
    """HAT: Hybrid Attention Transformer block (window + channel attention)."""
    def __init__(self, dim, window_size=8, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.window_attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(dim, dim//16), nn.ReLU(),
            nn.Linear(dim//16, dim), nn.Sigmoid()
        )
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Linear(dim*mlp_ratio, dim)
        )
        self.window_size = window_size

    def window_partition(self, x, ws):
        B, H, W, C = x.shape
        x = x.view(B, H//ws, ws, W//ws, ws, C).permute(0,1,3,2,4,5).contiguous()
        return x.view(-1, ws*ws, C)

    def window_reverse(self, x, ws, H, W):
        B_ = x.shape[0]; B = B_ // (H//ws * W//ws)
        x = x.view(B, H//ws, W//ws, ws, ws, -1).permute(0,1,3,2,4,5).contiguous()
        return x.view(B, H, W, -1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_t = x.permute(0,2,3,1)  # B,H,W,C
        # Window attention
        wins = self.window_partition(x_t, self.window_size)
        wins = wins + self.window_attn(self.norm1(wins))
        x_t = self.window_reverse(wins, self.window_size, H, W)
        x = x_t.permute(0,3,1,2)
        # Channel attention
        x_flat = x.flatten(2).transpose(1,2)  # B,HW,C
        ch_w = self.channel_attn(x_flat.transpose(1,2)).view(B, C, 1, 1)
        x = x * ch_w
        # MLP
        x_flat = x.flatten(2).transpose(1,2)
        x_flat = x_flat + self.mlp(self.norm3(x_flat))
        return x_flat.transpose(1,2).view(B, C, H, W)
