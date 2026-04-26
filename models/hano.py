import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fno import SpectralConv2d
from models.axial_attention import AxialAttention2d
from core.device import DEVICE

class HANOLayer(nn.Module):
    def __init__(self, hidden_dim, n_modes):
        super().__init__()
        self.spec = SpectralConv2d(in_ch=hidden_dim, out_ch=hidden_dim, n_modes1=n_modes, n_modes2=n_modes)
        self.attn = AxialAttention2d(hidden_dim, num_heads=4)
        self.mlp = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x_spec = self.spec(x)
        attn_out = self.attn(x)
        x_mlp = self.mlp(x)
        return F.gelu(x_spec + x_mlp + attn_out)

class HANO2d(nn.Module):
    """Hierarchical Attention Neural Operator (Multi-scale self-attention + spectral conv)."""
    
    def __init__(self, n_modes=8, hidden_dim=32, n_layers=4, in_channels=1, out_channels=1):
        super().__init__()
        # Lift: data channels + 2 spatial grids
        self.p = nn.Linear(in_channels + 2, hidden_dim)
        self.layers = nn.ModuleList([HANOLayer(hidden_dim, n_modes) for _ in range(n_layers)])
        self.q = nn.Linear(hidden_dim, out_channels)

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

        x = self.p(x)
        
        for layer in self.layers:
            x = layer(x)
            
        out = self.q(x)
        if out.shape[-1] == 1:
            return out[:, :, :, 0]
        return out
