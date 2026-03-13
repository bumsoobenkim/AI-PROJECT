"""PointNet++ Hierarchical Point Set Learning (Qi et al., NeurIPS 2017).
OccNet: Occupancy Networks for implicit 3D surface representation (Mescheder CVPR 2019).
Neural SDF with eikonal regularization (Gropp et al., ICML 2020).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """Pairwise squared distances between two point sets."""
    return (src.unsqueeze(2) - dst.unsqueeze(1)).pow(2).sum(-1)


def farthest_point_sampling(xyz, n_samples):
    """FPS: iteratively select farthest point from current set."""
    B, N, _ = xyz.shape
    device = xyz.device
    selected = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    dist = torch.full((B, N), float('inf'), device=device)
    idx = torch.randint(0, N, (B,), device=device)
    for i in range(n_samples):
        selected[:, i] = idx
        ctr = xyz[torch.arange(B), idx].unsqueeze(1)
        d = (xyz - ctr).pow(2).sum(-1)
        dist = torch.minimum(dist, d)
        idx = dist.argmax(1)
    return selected


def ball_query(xyz, query, radius, n_samples):
    """Find n_samples neighbors within radius for each query point."""
    B, N, _ = xyz.shape; B, M, _ = query.shape
    dists = square_distance(query, xyz)
    sorted_idx = dists.argsort(dim=-1)[:, :, :n_samples]
    mask = dists.gather(-1, sorted_idx) > radius ** 2
    sorted_idx[mask] = sorted_idx[:, :, :1].expand_as(sorted_idx)[mask]
    return sorted_idx


class PointNetSetAbstraction(nn.Module):
    """PointNet++ SA module: FPS + ball query + PointNet MLP."""
    def __init__(self, n_centers, radius, n_samples, in_dim, mlp_dims):
        super().__init__()
        self.n_centers = n_centers; self.radius = radius; self.n_samples = n_samples
        layers = []
        last = in_dim + 3
        for d in mlp_dims:
            layers += [nn.Conv2d(last, d, 1), nn.BatchNorm2d(d), nn.ReLU(True)]
            last = d
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features=None):
        B, N, _ = xyz.shape
        fps_idx = farthest_point_sampling(xyz, self.n_centers)
        new_xyz = xyz[torch.arange(B).unsqueeze(1), fps_idx]
        ball_idx = ball_query(xyz, new_xyz, self.radius, self.n_samples)
        grouped_xyz = xyz.unsqueeze(1).expand(-1, self.n_centers, -1, -1)
        grouped_xyz = grouped_xyz.gather(2, ball_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)
        if features is not None:
            grouped_feat = features.unsqueeze(1).expand(-1, self.n_centers, -1, -1)
            grouped_feat = grouped_feat.gather(2, ball_idx.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1]))
            x = torch.cat([grouped_xyz, grouped_feat], -1)
        else:
            x = grouped_xyz
        x = x.permute(0, 3, 1, 2)  # B, C, M, K
        x = self.mlp(x)
        x = x.max(dim=-1).values  # max pool over neighbors: B, C, M
        new_features = x.transpose(1, 2)  # B, M, C
        return new_xyz, new_features


class PointNetPlusPlus(nn.Module):
    """PointNet++ for 3D point cloud classification."""
    def __init__(self, num_classes=40, in_dim=0):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_dim, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(1, 1e10, 128, 256, [256, 512, 1024])
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, xyz):
        xyz1, f1 = self.sa1(xyz)
        xyz2, f2 = self.sa2(xyz1, f1)
        _, f3 = self.sa3(xyz2, f2)
        return self.classifier(f3.squeeze(1))


class OccupancyNetwork(nn.Module):
    """OccNet: implicit 3D surface as occ = f(p, z) in R (Mescheder CVPR 2019).
    Conditional Batch Norm (CBN) for latent code conditioning.
    """
    def __init__(self, latent_dim=256, hidden_dim=256, num_layers=5):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = 3
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.cbn_betas = nn.ModuleList([nn.Linear(latent_dim, hidden_dim) for _ in range(num_layers)])
        self.cbn_gammas = nn.ModuleList([nn.Linear(latent_dim, hidden_dim) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, p, z):
        """p: (B, N, 3) query points, z: (B, latent_dim) shape code."""
        B, N, _ = p.shape
        h = p
        for i, (lin, beta_lin, gamma_lin) in enumerate(zip(self.layers, self.cbn_betas, self.cbn_gammas)):
            h = lin(h)
            beta = beta_lin(z).unsqueeze(1)
            gamma = gamma_lin(z).unsqueeze(1)
            h = gamma * h + beta
            h = F.relu(h)
        return torch.sigmoid(self.out(h)).squeeze(-1)


class NeuralSDF(nn.Module):
    """Neural SDF with eikonal regularization (Gropp et al., ICML 2020).
    |grad f(x)| = 1 enforces SDF property.
    """
    def __init__(self, hidden_dim=256, num_layers=8, skip_layers=(4,)):
        super().__init__()
        self.skip_layers = skip_layers
        self.layers = nn.ModuleList()
        in_dim = 3
        for i in range(num_layers):
            out_dim = hidden_dim
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = hidden_dim + 3 if (i + 1) in skip_layers else hidden_dim
        self.out = nn.Linear(hidden_dim, 1)
        # Sine activation for IGR/SIREN
        self.first_omega = 30.0; self.hidden_omega = 1.0
        for i, lin in enumerate(self.layers):
            if i == 0:
                nn.init.uniform_(lin.weight, -1 / 3, 1 / 3)
            else:
                nn.init.uniform_(lin.weight, -(6 / hidden_dim) ** 0.5, (6 / hidden_dim) ** 0.5)

    def forward(self, x):
        h = x
        for i, lin in enumerate(self.layers):
            h = torch.sin(self.first_omega * lin(h) if i == 0 else self.hidden_omega * lin(h))
            if (i + 1) in self.skip_layers:
                h = torch.cat([h, x], -1)
        return self.out(h)

    def eikonal_loss(self, x):
        """Eikonal loss: E[||grad SDF||_2 - 1]^2."""
        x = x.requires_grad_(True)
        sdf = self(x)
        grad = torch.autograd.grad(sdf.sum(), x, create_graph=True)[0]
        return ((grad.norm(dim=-1) - 1) ** 2).mean()

    def loss(self, x_surf, x_free, sdf_gt=None, lambda_eik=0.1):
        sdf_pred = self(x_surf)
        surf_loss = sdf_pred.abs().mean()  # zero on surface
        eik_loss = self.eikonal_loss(torch.cat([x_surf, x_free], 0))
        total = surf_loss + lambda_eik * eik_loss
        if sdf_gt is not None:
            total = total + F.l1_loss(self(x_free), sdf_gt)
        return total
