"""Latent Diffusion Model (Rombach et al., CVPR 2022).
Implements VQVAE encoder/decoder, UNet denoiser, CLIP text conditioning.
Foundation for Stable Diffusion pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VectorQuantizer(nn.Module):
    """VQ-VAE codebook (van den Oord 2017) for LDM encoder."""
    def __init__(self, n_embed=8192, embed_dim=4, beta=0.25):
        super().__init__()
        self.n_embed = n_embed; self.embed_dim = embed_dim; self.beta = beta
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1/n_embed, 1/n_embed)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embed_dim)
        d = (z_flat.pow(2).sum(1, keepdim=True) + self.embedding.weight.pow(2).sum(1)
             - 2 * z_flat @ self.embedding.weight.T)
        min_idx = d.argmin(1)
        z_q = self.embedding(min_idx).view(z.shape)
        loss = F.mse_loss(z_q.detach(), z) + self.beta * F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach()  # straight-through
        z_q = z_q.permute(0, 3, 1, 2)
        return z_q, loss, min_idx


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.time_emb = nn.Linear(time_dim, out_ch) if time_dim else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x, t=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if t is not None and self.time_emb:
            h = h + self.time_emb(F.silu(t))[:, :, None, None]
        h = self.drop(self.conv2(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SpatialAttention(nn.Module):
    """Self-attention in spatial dimension for UNet middle block."""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.unbind(1)
        attn = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale, dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v).reshape(B, C, H, W)
        return x + self.proj(out)


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning in UNet."""
    def __init__(self, query_dim, context_dim=768, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5; self.heads = heads
        self.norm = nn.LayerNorm(query_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None):
        B, N, D = x.shape
        h_norm = self.norm(x)
        ctx = context if context is not None else h_norm
        q = self.to_q(h_norm).reshape(B, N, self.heads, -1).transpose(1, 2)
        k = self.to_k(ctx).reshape(B, ctx.shape[1], self.heads, -1).transpose(1, 2)
        v = self.to_v(ctx).reshape(B, ctx.shape[1], self.heads, -1).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x + self.to_out(out)


class LDMUNet(nn.Module):
    """Conditional UNet for LDM: (noisy_latent, t, text_emb) -> noise_pred."""
    def __init__(self, in_channels=4, base_ch=128, ch_mult=(1, 2, 4, 4),
                 context_dim=768, num_res=2):
        super().__init__()
        time_dim = base_ch * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_ch, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        channels = [base_ch * m for m in ch_mult]
        self.in_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = channels[0]
        for c in channels:
            block = nn.ModuleList([ResidualBlock(ch, c, time_dim) for _ in range(num_res)])
            self.down_blocks.append(block)
            self.down_samples.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
            ch = c
        # Middle
        self.mid = nn.ModuleList([
            ResidualBlock(ch, ch, time_dim), SpatialAttention(ch), ResidualBlock(ch, ch, time_dim)
        ])
        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for c in reversed(channels):
            block = nn.ModuleList([ResidualBlock(ch + c, c, time_dim) for _ in range(num_res)])
            self.up_blocks.append(block)
            self.up_samples.append(nn.ConvTranspose2d(c, c, 2, stride=2))
            ch = c
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        # Cross-attention layers for text conditioning
        self.cross_attns = nn.ModuleList([CrossAttention(c, context_dim) for c in channels])

    def sinusoidal_emb(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x, t, context=None):
        t_emb = self.time_embed(self.sinusoidal_emb(t, self.time_embed[0].in_features))
        h = self.in_conv(x)
        skips = []
        for i, (blocks, down) in enumerate(zip(self.down_blocks, self.down_samples)):
            for blk in blocks:
                h = blk(h, t_emb)
                if context is not None:
                    B, C, H, W = h.shape
                    h_flat = h.flatten(2).transpose(1, 2)
                    h_flat = self.cross_attns[i](h_flat, context)
                    h = h_flat.transpose(1, 2).view(B, C, H, W)
                skips.append(h)
            h = down(h)
        for blk in self.mid:
            h = blk(h) if not isinstance(blk, ResidualBlock) else blk(h, t_emb)
        for blocks, up in zip(self.up_blocks, self.up_samples):
            h = up(h)
            for blk in blocks:
                h = blk(torch.cat([h, skips.pop()], dim=1), t_emb)
        return self.out_conv(F.silu(self.out_norm(h)))


class DDPMScheduler:
    """DDPM noise scheduler with linear/cosine beta schedule."""
    def __init__(self, num_steps=1000, beta_start=0.00085, beta_end=0.012, schedule='scaled_linear'):
        self.num_steps = num_steps
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule == 'scaled_linear':
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2
        elif schedule == 'cosine':
            t = torch.arange(num_steps + 1) / num_steps
            alphas_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            self.betas = (1 - alphas_bar[1:] / alphas_bar[:-1]).clamp(max=0.999)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = self.alphas.cumprod(0)

    def add_noise(self, x0, noise, t):
        abar = self.alphas_bar[t].to(x0.device).view(-1, 1, 1, 1)
        return abar.sqrt() * x0 + (1 - abar).sqrt() * noise

    def step(self, model_output, t, xt):
        """DDPM reverse step: x_{t-1} from x_t and predicted noise."""
        abar_t = self.alphas_bar[t].to(xt.device)
        abar_prev = self.alphas_bar[t - 1].to(xt.device) if t > 0 else torch.ones(1)
        alpha_t = self.alphas[t].to(xt.device)
        x0_pred = (xt - (1 - abar_t).sqrt() * model_output) / abar_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        mean = (abar_prev.sqrt() * self.betas[t] / (1 - abar_t)) * x0_pred + \
               (alpha_t.sqrt() * (1 - abar_prev) / (1 - abar_t)) * xt
        var = self.betas[t] * (1 - abar_prev) / (1 - abar_t)
        noise = torch.randn_like(xt) if t > 0 else 0
        return mean + var.sqrt() * noise


class LatentDiffusionModel(nn.Module):
    """Full LDM pipeline: VAE compression + diffusion in latent space."""
    def __init__(self, vae, unet, text_encoder, scheduler=None, latent_scale=0.18215):
        super().__init__()
        self.vae = vae; self.unet = unet; self.text_encoder = text_encoder
        self.scheduler = scheduler or DDPMScheduler()
        self.latent_scale = latent_scale

    def encode(self, x):
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample() * self.latent_scale
        return latent

    def decode(self, z):
        with torch.no_grad():
            return self.vae.decode(z / self.latent_scale).sample

    def training_loss(self, x, text_tokens):
        z = self.encode(x)
        noise = torch.randn_like(z)
        t = torch.randint(0, self.scheduler.num_steps, (z.shape[0],), device=z.device)
        z_noisy = self.scheduler.add_noise(z, noise, t)
        context = self.text_encoder(text_tokens)[0]
        noise_pred = self.unet(z_noisy, t, context)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def generate(self, text_tokens, shape=(1, 4, 64, 64), steps=50, guidance_scale=7.5):
        device = next(self.unet.parameters()).device
        context = self.text_encoder(text_tokens)[0].to(device)
        uncond = self.text_encoder(torch.zeros_like(text_tokens))[0].to(device)
        z = torch.randn(*shape, device=device)
        timesteps = torch.linspace(self.scheduler.num_steps - 1, 0, steps, dtype=torch.long)
        for t_val in timesteps:
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            noise_cond = self.unet(z, t, context)
            noise_uncond = self.unet(z, t, uncond)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            z = self.scheduler.step(noise_pred, t_val.item(), z)
        return self.decode(z)
