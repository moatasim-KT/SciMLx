import mlx.core as mx
import mlx.nn as nn
from models.fno import SpectralConv2d

class FEDONet2d(nn.Module):
    """Fourier-Enhanced DeepONet (Spectral conv within DeepONet branches)."""
    
    def __init__(self, n_modes=8, hidden_dim=32, n_layers=4, in_channels=1, out_channels=1):
        super().__init__()
        # Lift: data channels + 2 spatial grids
        self.p = nn.Linear(in_channels + 2, hidden_dim)
        self.branch_spectral = SpectralConv2d(in_ch=hidden_dim, out_ch=hidden_dim, n_modes1=n_modes, n_modes2=n_modes)
        self.branch_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj = nn.Linear(hidden_dim, out_channels)

    def __call__(self, x):
        # x : [B, N1, N2] or [B, N1, N2, C]
        if x.ndim == 3:
            B, N1, N2 = x.shape
            x = x[..., None]
        else:
            B, N1, N2, _ = x.shape
            
        grid1 = mx.broadcast_to(mx.linspace(0.0, 1.0, N1).reshape(1, N1, 1, 1), (B, N1, N2, 1))
        grid2 = mx.broadcast_to(mx.linspace(0.0, 1.0, N2).reshape(1, 1, N2, 1), (B, N1, N2, 1))
        x     = mx.concatenate([x, grid1, grid2], axis=-1)  # [B, N1, N2, C+2]
        
        # Branch
        x_proj = self.p(x)
        b_spec = self.branch_spectral(x_proj)
        b_mlp = self.branch_mlp(x_proj)
        b_out = nn.gelu(b_spec + b_mlp)
        
        # Trunk (uses the same grid1/grid2 logic for coords)
        coords = mx.concatenate([grid1[0], grid2[0]], axis=-1) # [N1, N2, 2]
        t_out = self.trunk(coords)
        
        out = self.proj(b_out * t_out[None, ...])
        if out.shape[-1] == 1:
            return out[:, :, :, 0]
        return out
