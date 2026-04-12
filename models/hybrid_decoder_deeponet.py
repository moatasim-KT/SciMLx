import mlx.core as mx
import mlx.nn as nn
from models.fno import FNO2d, FNOBlock2d, SpectralConv2d

class HybridDecoderDeepONet2d(nn.Module):
    """Hybrid Decoder-DeepONet (FNO spatial encoder + DeepONet trunk)."""

    def __init__(self, n_modes=8, hidden_dim=32, n_layers=4, in_channels=1, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Branch network (FNO encoder) - Lift data channels + 2 grids
        self.branch_lift = nn.Linear(in_channels + 2, hidden_dim)
        self.branch_blocks = [FNOBlock2d(hidden_dim, n_modes, n_modes) for _ in range(n_layers)]
        
        # Trunk network (Standard MLP for coordinate embeddings)
        self.trunk_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.projection = nn.Linear(hidden_dim, out_channels)

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
        
        # Compute branch embeddings
        b_x = self.branch_lift(x)
        for blk in self.branch_blocks:
            b_x = blk(b_x)
        branch_emb = b_x
        
        # Trunk embeddings (uses unique coordinates)
        coords = mx.concatenate([grid1[0], grid2[0]], axis=-1) # [N1, N2, 2]
        trunk_emb = self.trunk_mlp(coords)
        
        # Dot product element-wise (branch * trunk)
        combined = branch_emb * trunk_emb[None, ...]
        
        out = self.projection(combined)
        if out.shape[-1] == 1:
            return out[:, :, :, 0]
        return out
