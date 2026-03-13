"""3D Gaussian Splatting: Real-Time Novel View Synthesis (Kerbl SIGGRAPH 2023).
Differentiable rasterization of 3D Gaussians projected to 2D screen space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianModel(nn.Module):
    """Learnable 3D Gaussian primitives."""
    def __init__(self, num_gaussians=100000):
        super().__init__()
        self.num_gaussians = num_gaussians
        # Position (xyz), opacity, scale (xyz), rotation (quaternion), SH coefficients
        self.xyz = nn.Parameter(torch.randn(num_gaussians, 3) * 0.1)
        self.opacity_logit = nn.Parameter(torch.zeros(num_gaussians))
        self.log_scale = nn.Parameter(torch.full((num_gaussians, 3), -4.0))
        self.rotation = nn.Parameter(torch.cat([
            torch.ones(num_gaussians, 1), torch.zeros(num_gaussians, 3)
        ], dim=1))  # quaternion (w, x, y, z)
        # Spherical harmonics coefficients (degree 3 = 16 coeffs per channel)
        self.sh_coeffs = nn.Parameter(torch.zeros(num_gaussians, 3, 16))

    @property
    def opacity(self):
        return torch.sigmoid(self.opacity_logit)

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def get_rotation_matrix(self):
        """Quaternion to rotation matrix."""
        q = F.normalize(self.rotation, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y),
            2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x),
            2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)
        ], dim=-1).view(-1, 3, 3)
        return R

    def get_covariance_3d(self):
        """Compute 3D covariance: Sigma = R * S * S^T * R^T."""
        R = self.get_rotation_matrix()
        S = torch.diag_embed(self.scale)
        RS = torch.bmm(R, S)
        return torch.bmm(RS, RS.transpose(-2, -1))

    def project_to_2d(self, camera_transform, intrinsics):
        """Project 3D Gaussians to 2D screen-space covariances."""
        R_cam, t_cam = camera_transform
        # Transform means to camera space
        xyz_cam = (self.xyz @ R_cam.T) + t_cam
        depth = xyz_cam[:, 2].clamp(min=0.01)

        # Project to screen
        fx, fy, cx, cy = intrinsics
        x_screen = (xyz_cam[:, 0] * fx) / depth + cx
        y_screen = (xyz_cam[:, 1] * fy) / depth + cy
        means_2d = torch.stack([x_screen, y_screen], dim=-1)

        # 2D covariance via Jacobian of projection
        Sigma3d = self.get_covariance_3d()
        J = torch.zeros(*self.xyz.shape[:-1], 2, 3, device=self.xyz.device)
        J[:, 0, 0] = fx / depth; J[:, 0, 2] = -fx * xyz_cam[:, 0] / depth**2
        J[:, 1, 1] = fy / depth; J[:, 1, 2] = -fy * xyz_cam[:, 1] / depth**2
        W = R_cam[:2, :3].unsqueeze(0)
        JW = J @ W.expand_as(J)
        Sigma2d = JW @ Sigma3d @ JW.transpose(-2, -1)
        Sigma2d = Sigma2d + 0.3 * torch.eye(2, device=Sigma2d.device)

        return means_2d, Sigma2d, depth

    def eval_sh(self, dirs):
        """Evaluate degree-1 spherical harmonics for color from viewing direction."""
        C0 = 0.28209479177387814
        C1 = 0.4886025119029199
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        # Degree 0 + 1
        basis = torch.stack([
            torch.ones_like(x) * C0,
            -C1 * y, C1 * z, -C1 * x
        ], dim=-1)  # N, 4
        colors = (self.sh_coeffs[:, :, :4] * basis.unsqueeze(1)).sum(-1) + 0.5
        return torch.clamp(colors, 0.0, 1.0)


class GaussianRasterizer(nn.Module):
    """Differentiable 2D Gaussian splatting rasterizer."""
    def __init__(self, img_h=800, img_w=800):
        super().__init__()
        self.H = img_h; self.W = img_w

    def render_tile(self, means_2d, Sigma2d, colors, opacity, depth_order):
        """Render sorted Gaussians using alpha compositing."""
        H, W = self.H, self.W
        # Sort by depth (front to back)
        sorted_idx = depth_order.argsort()
        img = torch.zeros(H, W, 3, device=means_2d.device)
        alpha_acc = torch.zeros(H, W, device=means_2d.device)

        # Precompute inverse covariances
        det = Sigma2d[:, 0, 0] * Sigma2d[:, 1, 1] - Sigma2d[:, 0, 1] ** 2
        inv_det = 1.0 / (det + 1e-6)
        Sigma_inv = torch.stack([
            Sigma2d[:, 1, 1] * inv_det, -Sigma2d[:, 0, 1] * inv_det,
            -Sigma2d[:, 0, 1] * inv_det, Sigma2d[:, 0, 0] * inv_det
        ], dim=-1).view(-1, 2, 2)

        # Pixel coordinates
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=means_2d.device, dtype=torch.float32),
            torch.arange(W, device=means_2d.device, dtype=torch.float32),
            indexing='ij'
        )
        pixels = torch.stack([x_grid, y_grid], dim=-1)  # H, W, 2

        for idx in sorted_idx[:min(len(sorted_idx), 50000)]:
            if alpha_acc.min() > 0.9999:
                break
            mu = means_2d[idx]  # (2,)
            diff = pixels - mu.view(1, 1, 2)  # H, W, 2
            Sinv = Sigma_inv[idx]  # 2, 2
            exponent = -0.5 * (diff @ Sinv * diff).sum(-1)
            gaussian = torch.exp(exponent.clamp(min=-20))
            alpha = opacity[idx] * gaussian
            T = 1.0 - alpha_acc
            img = img + T.unsqueeze(-1) * alpha.unsqueeze(-1) * colors[idx].view(1, 1, 3)
            alpha_acc = alpha_acc + T * alpha

        return img.permute(2, 0, 1)  # CHW

    def forward(self, gaussians: GaussianModel, camera_transform, intrinsics, dirs=None):
        if dirs is None:
            cam_pos = -camera_transform[0].T @ camera_transform[1]
            dirs = F.normalize(gaussians.xyz - cam_pos, dim=-1)
        colors = gaussians.eval_sh(dirs)
        means_2d, Sigma2d, depth = gaussians.project_to_2d(camera_transform, intrinsics)
        visible = (depth > 0.1) & (means_2d[:, 0] > 0) & (means_2d[:, 0] < self.W) & \
                  (means_2d[:, 1] > 0) & (means_2d[:, 1] < self.H)
        img = self.render_tile(means_2d[visible], Sigma2d[visible],
                               colors[visible], gaussians.opacity[visible], depth[visible])
        return img
