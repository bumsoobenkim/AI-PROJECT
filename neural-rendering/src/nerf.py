"""NeRF: Representing Scenes as Neural Radiance Fields (Mildenhall ECCV 2020).
Instant-NGP: Instant Neural Graphics Primitives (Muller NeurIPS 2022).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(x, num_levels=10):
    freqs = 2.0 ** torch.arange(num_levels, dtype=x.dtype, device=x.device) * torch.pi
    xf = x.unsqueeze(-1) * freqs
    enc = torch.cat([torch.sin(xf), torch.cos(xf)], dim=-1)
    return torch.cat([x, enc.flatten(-2)], dim=-1)


class NeRF(nn.Module):
    """Vanilla NeRF MLP: (xyz, dir) -> (RGB, sigma)."""
    def __init__(self, pos_levels=10, dir_levels=4, hidden=256, skips=(4,)):
        super().__init__()
        self.pos_levels = pos_levels
        self.dir_levels = dir_levels
        self.skips = skips
        pos_dim = 3 + 3 * 2 * pos_levels
        dir_dim = 3 + 3 * 2 * dir_levels
        self.pts_linears = nn.ModuleList(
            [nn.Linear(pos_dim, hidden)] +
            [nn.Linear(hidden, hidden) if i not in skips
             else nn.Linear(hidden + pos_dim, hidden) for i in range(1, 8)]
        )
        self.sigma_linear = nn.Linear(hidden, 1)
        self.feature_linear = nn.Linear(hidden, hidden)
        self.rgb_linear = nn.Sequential(
            nn.Linear(hidden + dir_dim, hidden // 2),
            nn.ReLU(), nn.Linear(hidden // 2, 3)
        )

    def forward(self, pts, dirs):
        pts_enc = positional_encoding(pts, self.pos_levels)
        dirs_enc = positional_encoding(F.normalize(dirs, dim=-1), self.dir_levels)
        h = pts_enc
        for i, lin in enumerate(self.pts_linears):
            inp = h if i not in self.skips else torch.cat([h, pts_enc], -1)
            h = F.relu(lin(inp))
        sigma = F.softplus(self.sigma_linear(h))
        feat = self.feature_linear(h)
        rgb = torch.sigmoid(self.rgb_linear(torch.cat([feat, dirs_enc], -1)))
        return rgb, sigma


class VolumeRenderer(nn.Module):
    """Hierarchical volume rendering for NeRF."""
    def __init__(self, coarse, fine=None, near=2.0, far=6.0, n_coarse=64, n_fine=128):
        super().__init__()
        self.coarse = coarse
        self.fine = fine or coarse
        self.near = near; self.far = far
        self.n_coarse = n_coarse; self.n_fine = n_fine

    def sample_rays(self, rays_o, rays_d, n, perturb=True):
        B = rays_o.shape[0]
        t = torch.linspace(self.near, self.far, n, device=rays_o.device).expand(B, n)
        if perturb:
            mids = 0.5 * (t[..., 1:] + t[..., :-1])
            upper = torch.cat([mids, t[..., -1:]], -1)
            lower = torch.cat([t[..., :1], mids], -1)
            t = lower + (upper - lower) * torch.rand_like(t)
        pts = rays_o[:, None, :] + rays_d[:, None, :] * t[:, :, None]
        return pts, t

    def importance_sample(self, rays_o, rays_d, t_c, weights, n_fine):
        weights = weights + 1e-5
        pdf = weights / weights.sum(-1, keepdim=True)
        cdf = pdf.cumsum(-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
        u = torch.rand(*cdf.shape[:-1], n_fine, device=cdf.device)
        inds = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
        below = (inds - 1).clamp(0)
        above = inds.clamp(max=cdf.shape[-1] - 1)
        t_mid = 0.5 * (t_c.gather(-1, below) + t_c.gather(-1, above))
        t_fine, _ = torch.sort(torch.cat([t_c, t_mid], -1), -1)
        pts = rays_o[:, None, :] + rays_d[:, None, :] * t_fine[:, :, None]
        return pts, t_fine

    def render(self, model, pts, dirs, t):
        B, N, _ = pts.shape
        dirs_exp = dirs[:, None, :].expand(B, N, 3)
        rgb, sigma = model(pts.reshape(-1, 3), dirs_exp.reshape(-1, 3))
        rgb = rgb.view(B, N, 3); sigma = sigma.view(B, N)
        delta = t[..., 1:] - t[..., :-1]
        delta = torch.cat([delta, torch.full_like(delta[..., :1], 1e10)], -1)
        alpha = 1 - torch.exp(-sigma * delta)
        T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1 - alpha + 1e-10], -1), -1)[..., :-1]
        weights = T * alpha
        color = (weights[:, :, None] * rgb).sum(-2)
        depth = (weights * t).sum(-1)
        return color, depth, weights

    def forward(self, rays_o, rays_d, perturb=True):
        pts_c, t_c = self.sample_rays(rays_o, rays_d, self.n_coarse, perturb)
        color_c, depth_c, weights_c = self.render(self.coarse, pts_c, rays_d, t_c)
        pts_f, t_f = self.importance_sample(rays_o, rays_d, t_c, weights_c.detach(), self.n_fine)
        color_f, depth_f, _ = self.render(self.fine, pts_f, rays_d, t_f)
        return {"coarse": color_c, "fine": color_f, "depth": depth_f}


class HashEncoder(nn.Module):
    """Instant-NGP multi-resolution hash grid encoding."""
    def __init__(self, n_levels=16, n_features=2, log2_table=19, base_res=16, max_res=2048):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        T = 2 ** log2_table
        growth = (max_res / base_res) ** (1.0 / (n_levels - 1))
        self.resolutions = [int(base_res * growth ** i) for i in range(n_levels)]
        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(min(T, r ** 3), n_features) * 1e-4)
            for r in self.resolutions
        ])

    def _hash(self, coords, T):
        primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.long, device=coords.device)
        return ((coords * primes).sum(-1) % T).abs()

    def _interp(self, x, table, res):
        x = x * res
        x0 = x.long().clamp(0, res - 1)
        x1 = (x0 + 1).clamp(0, res - 1)
        w1 = x - x0.float(); w0 = 1.0 - w1
        T = table.shape[0]
        out = torch.zeros(x.shape[0], table.shape[1], device=x.device)
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    c = torch.stack([x0[:, 0] + dx, x0[:, 1] + dy, x0[:, 2] + dz], -1).clamp(0, res - 1)
                    idx = self._hash(c, T)
                    w = (w1[:, 0] if dx else w0[:, 0]) * (w1[:, 1] if dy else w0[:, 1]) * (w1[:, 2] if dz else w0[:, 2])
                    out = out + table[idx] * w[:, None]
        return out

    def forward(self, x):
        return torch.cat([self._interp(x, t, r) for t, r in zip(self.tables, self.resolutions)], -1)


class InstantNGP(nn.Module):
    """Instant-NGP: hash encoding + tiny MLP."""
    def __init__(self, n_levels=16, n_features=2, hidden=64):
        super().__init__()
        self.encoder = HashEncoder(n_levels, n_features)
        enc_dim = n_levels * n_features
        self.sigma_net = nn.Sequential(nn.Linear(enc_dim, hidden), nn.ReLU(), nn.Linear(hidden, 16))
        self.rgb_net = nn.Sequential(nn.Linear(16 + 27, hidden), nn.ReLU(), nn.Linear(hidden, 3), nn.Sigmoid())

    def forward(self, pts, dirs_sh):
        h = self.sigma_net(self.encoder(pts))
        sigma = F.softplus(h[:, :1])
        rgb = self.rgb_net(torch.cat([h[:, 1:], dirs_sh], -1))
        return rgb, sigma
