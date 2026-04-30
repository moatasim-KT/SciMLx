import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fno import SpectralConv2d

class VSMNO2d(nn.Module):
    """Variational Spectral Mixture Neural Operator (Mixture of spectral patterns)."""
    
    def __init__(self, n_modes=8, hidden_dim=32, n_layers=4, in_channels=1, out_channels=1):
        super().__init__()
        self.p = nn.Linear(in_channels + 2, hidden_dim)
        
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "spec_1": SpectralConv2d(in_ch=hidden_dim, out_ch=hidden_dim, n_modes1=n_modes, n_modes2=n_modes),
                "spec_2": SpectralConv2d(in_ch=hidden_dim, out_ch=hidden_dim, n_modes1=max(2, n_modes//2), n_modes2=max(2, n_modes//2)),
                "mlp": nn.Linear(hidden_dim, hidden_dim)
            }))
            
        self.w_mix = nn.Parameter(torch.ones(2))
        self.q = nn.Linear(hidden_dim, out_channels)

    def forward(self, x):
        if x.ndim == 3:
            B, N1, N2 = x.shape
            x = x.unsqueeze(-1)
        else:
            B, N1, N2, _ = x.shape
            
        grid1 = torch.linspace(0.0, 1.0, N1, device=x.device).view(1, N1, 1, 1).expand(B, N1, N2, 1)
        grid2 = torch.linspace(0.0, 1.0, N2, device=x.device).view(1, 1, N2, 1).expand(B, N1, N2, 1)
        x = torch.cat([x, grid1, grid2], dim=-1)

        x = self.p(x)
        weights = F.softmax(self.w_mix, dim=0)
        
        for block in self.blocks:
            s1 = block["spec_1"](x)
            s2 = block["spec_2"](x)
            m = block["mlp"](x)
            x = F.gelu(weights[0] * s1 + weights[1] * s2 + m)
            
        out = self.q(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
