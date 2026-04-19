import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fno import SpectralConv2d
from models.axial_attention import AxialAttention2d

class AttentionEnhancedFNO2d(nn.Module):
    """Attention-Enhanced FNO: FNO + axial self-attention for global spatial awareness."""

    def __init__(self, n_modes=8, hidden_dim=32, n_layers=4, in_channels=1, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Lift: data channels + 2 spatial grids
        self.p = nn.Linear(in_channels + 2, hidden_dim)

        self.spectral_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv2d(in_ch=hidden_dim, out_ch=hidden_dim, n_modes1=n_modes, n_modes2=n_modes))
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ))
            # Axial attention over spatial grid
            self.attentions.append(AxialAttention2d(hidden_dim, num_heads=4))

        self.q = nn.Linear(hidden_dim, out_channels)

    def forward(self, x):
        # x : [B, N1, N2] or [B, N1, N2, C]
        if x.ndim == 3:
            B, N1, N2 = x.shape
            x = x[..., None]
        else:
            B, N1, N2, _ = x.shape

        grid1 = torch.linspace(0.0, 1.0, N1, device=x.device, dtype=x.dtype).reshape(1, N1, 1, 1).expand(B, -1, N2, 1)
        grid2 = torch.linspace(0.0, 1.0, N2, device=x.device, dtype=x.dtype).reshape(1, 1, N2, 1).expand(B, N1, -1, 1)
        x     = torch.cat([x, grid1, grid2], dim=-1)  # [B, N1, N2, C+2]

        x = self.p(x)

        for spec, mlp, attn in zip(self.spectral_layers, self.mlp_layers, self.attentions):
            x1 = spec(x)
            x_attn = attn(x)
            x2 = mlp(x)
            x = x1 + x2 + x_attn
            x = F.gelu(x)

        out = self.q(x)
        if out.shape[-1] == 1:
            return out[:, :, :, 0]
        return out
