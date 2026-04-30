import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from core.device import DEVICE


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network Layer in PyTorch.
    Replaces y = Sigma(Wx + b) with y_j = sum_i phi_{i,j}(x_i).

    Reference: Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks"
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3, use_factorization: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.use_factorization = use_factorization

        # Base weights (Section 3.1.2: Weight Factorization W = diag(s) * V)
        scale_base = 1.0 / math.sqrt(in_features)
        if use_factorization:
            self.base_s = nn.Parameter(torch.ones(out_features, 1))
            self.base_v = nn.Parameter(torch.randn(out_features, in_features) * scale_base)
        else:
            self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * scale_base)

        # Spline weights: [out, in, grid_size + spline_order]
        n_spline = grid_size + spline_order
        scale_spline = 1.0 / math.sqrt(in_features * grid_size)
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, n_spline) * scale_spline)

        # Grid points for B-splines
        grid = torch.linspace(-1.0, 1.0, n_spline).reshape(1, 1, n_spline)
        self.register_buffer('grid', grid)

    def update_grid(self, new_grid_size: int):
        """Adaptive Grid Extension: Interpolate existing weights to a finer grid."""
        old_n = self.grid_size + self.spline_order
        new_n = new_grid_size + self.spline_order
        
        # Linear interpolation of spline weights from old_n to new_n
        x_old = torch.linspace(0, 1, old_n, device=self.spline_weight.device)
        x_new = torch.linspace(0, 1, new_n, device=self.spline_weight.device)
        
        # Simple linear interpolation logic
        idx = (x_new * (old_n - 1)).long()
        idx_next = torch.clamp(idx + 1, max=old_n - 1)
        alpha = (x_new * (old_n - 1)) - idx
        alpha = alpha.reshape(1, 1, new_n) # [1, 1, new_n]
        
        w_low = self.spline_weight[:, :, idx]
        w_high = self.spline_weight[:, :, idx_next]
        
        # New interpolated weights
        with torch.no_grad():
            new_spline_weight = w_low + alpha * (w_high - w_low)
            self.spline_weight = nn.Parameter(new_spline_weight)
            self.grid_size = new_grid_size
            grid = torch.linspace(-1.0, 1.0, new_n, device=self.spline_weight.device).reshape(1, 1, new_n)
            self.register_buffer('grid', grid)
        print(f"[KAN] Grid refined: {old_n-self.spline_order} -> {new_grid_size}")

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline basis functions for input x.
        x: [B, N, in_features]
        Returns: [B, N, in_features, n_spline]
        """
        x = x.unsqueeze(-1)  # [..., in, 1]
        grid = self.grid  # [1, 1, G]

        dist = torch.abs(x - grid)
        h = 2.0 / self.grid_size
        basis = torch.clamp(1.0 - (dist / h), min=0.0)  # Triangular basis
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, ..., in_features]
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # Base path: SiLU(x) * W
        if self.use_factorization:
            # W = s * V
            base_weight = self.base_s * self.base_v
        else:
            base_weight = self.base_weight
        base_output = F.silu(x) @ base_weight.t()

        # Spline path
        basis = self.b_splines(x)

        # Efficient contraction: sum_{in} basis[b, i, k] * spline_weight[out, i, k]
        # output: [batch_flat, out_features]
        spline_output = torch.einsum("bik,oik->bo", basis, self.spline_weight)

        out = base_output + spline_output

        # Restore shape
        new_shape = list(original_shape[:-1]) + [self.out_features]
        return out.reshape(new_shape)


class KANBlock1d(nn.Module):
    """1D KAN layer: spectral conv + pointwise KAN."""

    def __init__(self, channels: int, n_modes: int, grid_size: int = 5):
        super().__init__()
        from models.fno import SpectralConv1d

        self.spec = SpectralConv1d(channels, channels, n_modes)
        self.kan = KANLinear(channels, channels, grid_size=grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # KAN replaces the classic Linear + GELU activation
        return self.spec(x) + self.kan(x)


class KAN_FNO(nn.Module):
    """Kolmogorov-Arnold Spectral Operator (KAN-FNO) for 1-D PDEs."""

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, in_ch: int = 2, grid_size: int = 5):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([KANBlock1d(hidden_dim, n_modes, grid_size) for _ in range(n_layers)])
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        # u0: [B, N] or [B, N, C]
        if u0.ndim == 2:
            B, N = u0.shape
            grid = torch.linspace(0.0, 1.0, N, device=u0.device).reshape(1, N).expand(B, N)
            x = torch.stack([u0, grid], dim=-1)
        else:
            B, N, _C = u0.shape
            grid = torch.linspace(0.0, 1.0, N, device=u0.device).reshape(1, N, 1).expand(B, N, 1)
            x = torch.cat([u0, grid], dim=-1)

        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.proj1(x))
        return self.proj2(x)[..., 0]

class ModifiedKAN_FNO(nn.Module):
    """
    Modified KAN-FNO incorporating input-projection sub-networks.
    Reference: Section 3.1.2 "Architectures" (Wang et al. modified MLP).
    Uses two encoders to project input into high-dim space, then gates each layer.
    """
    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, in_ch: int = 2, grid_size: int = 5):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        
        # Encoders for gating
        self.encoder_u = nn.Linear(in_ch, hidden_dim)
        self.encoder_v = nn.Linear(in_ch, hidden_dim)
        
        self.blocks = nn.ModuleList([KANBlock1d(hidden_dim, n_modes, grid_size) for _ in range(n_layers)])
        
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        if u0.ndim == 2:
            B, N = u0.shape
            grid = torch.linspace(0.0, 1.0, N, device=u0.device).reshape(1, N).expand(B, N)
            raw_x = torch.stack([u0, grid], dim=-1)
        else:
            B, N, _C = u0.shape
            grid = torch.linspace(0.0, 1.0, N, device=u0.device).reshape(1, N, 1).expand(B, N, 1)
            raw_x = torch.cat([u0, grid], dim=-1)
            
        # Global encoders
        u = F.silu(self.encoder_u(raw_x))
        v = F.silu(self.encoder_v(raw_x))
        
        x = self.lift(raw_x)
        for blk in self.blocks:
            # Modified MLP style: Gate layer with projection
            h = blk(x)
            x = (1.0 - h) * u + h * v
            
        x = F.gelu(self.proj1(x))
        return self.proj2(x)[..., 0]
