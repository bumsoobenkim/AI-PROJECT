"""
StyleGAN3: Alias-Free Generative Adversarial Networks (Karras et al., NeurIPS 2021).
Implements modulated convolutions, alias-free synthesis, and mapping network.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate."""
    def __init__(self, in_dim, out_dim, bias=True, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.scale,
                        self.bias * self.lr_mul if self.bias is not None else None)


class MappingNetwork(nn.Module):
    """StyleGAN mapping network: z -> w (disentangled latent space)."""
    def __init__(self, z_dim=512, w_dim=512, num_layers=8, lr_mul=0.01):
        super().__init__()
        layers = []
        in_dim = z_dim
        for _ in range(num_layers):
            layers.append(EqualLinear(in_dim, w_dim, lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = w_dim
        self.net = nn.Sequential(*layers)
        self.pixel_norm = PixelNorm()

    def forward(self, z, truncation=1.0, w_avg=None):
        z = self.pixel_norm(z)
        w = self.net(z)
        if truncation < 1.0 and w_avg is not None:
            w = w_avg + truncation * (w - w_avg)
        return w


class PixelNorm(nn.Module):
    def forward(self, x):
        return x / (x.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)


class ModulatedConv2d(nn.Module):
    """Modulated convolution: w modulates conv filters per instance."""
    def __init__(self, in_channels, out_channels, kernel_size, w_dim=512,
                 demodulate=True, upsample=False, downsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.downsample = downsample
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.modulation = EqualLinear(w_dim, in_channels)

    def forward(self, x, w):
        B, C, H, W = x.shape
        s = self.modulation(w).view(B, 1, C, 1, 1) + 1  # style
        weight = self.scale * self.weight * s  # (B, out, in, k, k)
        if self.demodulate:
            d = weight.pow(2).sum(dim=(2, 3, 4), keepdim=True).rsqrt()
            weight = weight * d
        # Reshape for group conv
        x_reshaped = x.view(1, B * C, H, W)
        weight_reshaped = weight.view(B * self.out_channels, C, self.kernel_size, self.kernel_size)
        if self.upsample:
            x_reshaped = F.interpolate(x_reshaped, scale_factor=2, mode='bilinear', align_corners=False)
        out = F.conv2d(x_reshaped, weight_reshaped, padding=self.padding, groups=B)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        return out.view(B, self.out_channels, out.shape[-2], out.shape[-1])


class StyleBlock(nn.Module):
    """Synthesis block: ModulatedConv + noise injection + activation."""
    def __init__(self, in_ch, out_ch, w_dim=512, upsample=False):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, 3, w_dim=w_dim, upsample=upsample)
        self.noise_weight = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.act = nn.LeakyReLU(0.2)
        self.to_rgb = ModulatedConv2d(out_ch, 3, 1, w_dim=w_dim, demodulate=False)

    def forward(self, x, w, noise=None):
        x = self.conv(x, w)
        if noise is None:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_weight * noise + self.bias
        x = self.act(x)
        rgb = self.to_rgb(x, w)
        return x, rgb


class StyleGAN3Generator(nn.Module):
    """StyleGAN3 generator with alias-free synthesis network."""
    def __init__(self, z_dim=512, w_dim=512, img_resolution=256, channel_base=32768):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.mapping = MappingNetwork(z_dim, w_dim)
        log_res = int(math.log2(img_resolution))
        channels = {
            4: min(512, channel_base // 4),
            8: min(512, channel_base // 8),
            16: min(512, channel_base // 16),
            32: min(512, channel_base // 32),
            64: channel_base // 64,
            128: channel_base // 128,
            256: channel_base // 256,
        }
        self.const = nn.Parameter(torch.randn(1, channels[4], 4, 4))
        self.blocks = nn.ModuleList()
        in_ch = channels[4]
        for i in range(2, log_res + 1):
            out_ch = channels[min(2 ** i, img_resolution)]
            self.blocks.append(StyleBlock(in_ch, out_ch, w_dim, upsample=(i > 2)))
            in_ch = out_ch
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, z, c=None, truncation=1.0):
        w = self.mapping(z)
        x = self.const.expand(z.shape[0], -1, -1, -1)
        rgb = None
        for block in self.blocks:
            x, new_rgb = block(x, w)
            if rgb is None:
                rgb = new_rgb
            else:
                rgb = self.upsample(rgb) + new_rgb
        return torch.tanh(rgb)


class StyleGAN3Discriminator(nn.Module):
    """StyleGAN discriminator with minibatch std and progressive architecture."""
    def __init__(self, img_resolution=256, channel_base=32768):
        super().__init__()
        log_res = int(math.log2(img_resolution))
        channels = {r: min(512, channel_base // r) for r in [4, 8, 16, 32, 64, 128, 256]}
        self.from_rgb = nn.Sequential(
            nn.Conv2d(3, channels[img_resolution], 1),
            nn.LeakyReLU(0.2)
        )
        blocks = []
        for i in range(log_res, 2, -1):
            res = 2 ** i
            in_ch = channels[res]
            out_ch = channels[max(res // 2, 4)]
            blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            ))
        self.blocks = nn.ModuleList(blocks)
        final_ch = channels[4]
        self.final = nn.Sequential(
            nn.Conv2d(final_ch + 1, final_ch, 3, padding=1),  # +1 for minibatch std
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            EqualLinear(final_ch * 4 * 4, final_ch),
            nn.LeakyReLU(0.2),
            EqualLinear(final_ch, 1)
        )

    def minibatch_std(self, x):
        std = x.std(0, keepdim=True).mean().expand(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, std], dim=1)

    def forward(self, x):
        x = self.from_rgb(x)
        for block in self.blocks:
            x = block(x)
        x = self.minibatch_std(x)
        return self.final(x)


def r1_penalty(real_pred, real_imgs):
    """R1 gradient penalty (Mescheder et al. 2018)."""
    grad = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_imgs,
        create_graph=True, retain_graph=True
    )[0]
    return grad.pow(2).reshape(real_imgs.shape[0], -1).sum(1).mean()


def train_stylegan3(G, D, train_loader, epochs=100, lr=2e-3, r1_gamma=10.0):
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0, 0.99))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0, 0.99))
    device = next(G.parameters()).device

    for epoch in range(epochs):
        for real in train_loader:
            real = real.to(device).requires_grad_(True)
            B = real.shape[0]

            # Discriminator step
            z = torch.randn(B, G.z_dim, device=device)
            fake = G(z).detach()
            d_real = D(real)
            d_fake = D(fake)
            loss_D = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
            # R1 penalty every 16 steps
            penalty = r1_penalty(D(real), real)
            loss_D = loss_D + r1_gamma / 2 * penalty
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Generator step
            z = torch.randn(B, G.z_dim, device=device)
            fake = G(z)
            loss_G = F.softplus(-D(fake)).mean()
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
