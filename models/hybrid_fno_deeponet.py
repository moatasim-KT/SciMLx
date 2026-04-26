import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fno import FNO2d, FNOBlock2d
from core.device import DEVICE

class HybridFNODeepONet2d(nn.Module):
    """Hybrid FNO-DeepONet (FNO branch + DeepONet trunk)."""
    
    def __init__(self, n_modes=8, hidden_dim=32, n_layers=4, in_channels=1, out_channels=1):
        super().__init__()
        # Lift: data channels + 2 spatial grids
        self.branch_lift = nn.Linear(in_channels + 2, hidden_dim)
        self.branch_blocks = nn.ModuleList([FNOBlock2d(hidden_dim, n_modes, n_modes) for _ in range(n_layers)])
        
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj = nn.Linear(hidden_dim, out_channels)
        
    def forward(self, x):
        # x : [B, N1, N2] or [B, N1, N2, C]
        if x.ndim == 3:
            B, N1, N2 = x.shape
            x = x[..., None]
        else:
            B, N1, N2, _ = x.shape
            
        grid1 = torch.linspace(0.0, 1.0, N1, device=x.device).reshape(1, N1, 1, 1).expand(B, N1, N2, 1)
        grid2 = torch.linspace(0.0, 1.0, N2, device=x.device).reshape(1, 1, N2, 1).expand(B, N1, N2, 1)
        x     = torch.cat([x, grid1, grid2], dim=-1)  # [B, N1, N2, C+2]

        b_x = self.branch_lift(x)
        for blk in self.branch_blocks:
            b_x = blk(b_x)
        b_out = b_x  # [B, N1, N2, hidden]
        
        # Trunk embeddings (uses unique coordinates)
        coords = torch.cat([grid1[0], grid2[0]], dim=-1) # [N1, N2, 2]
        t_out = self.trunk(coords)
        
        # Merge by standard dot product approximation
        out = self.proj(b_out * t_out.unsqueeze(0))
        if out.shape[-1] == 1:
            return out[:, :, :, 0]
        return out
