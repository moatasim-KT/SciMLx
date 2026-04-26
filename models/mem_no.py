"""Memory Neural Operator (MemNO) for SciML.

Integrates structured state-space models (S4) into FNO architectures to 
provide explicit temporal memory for modeling time-dependent PDEs.

Reference: Buitrago Ruiz et al. (2025) "On the Benefits of Memory for 
Modeling Time-Dependent PDEs" (ICLR)
"""

import math
import torch
import torch.nn as nn
from models.s4d import S4DLayer
from core.device import DEVICE

class MemNOBlock1d(nn.Module):
    """Hybrid FNO block with integrated S4 state-space memory."""
    def __init__(self, hidden_dim, n_modes, d_state=64):
        super().__init__()
        # 1. Fourier branch
        self.w = nn.Parameter(torch.randn(n_modes, hidden_dim, hidden_dim, dtype=torch.complex64) * (hidden_dim ** -0.5))
        self.n_modes = n_modes
        
        # 2. Memory (SSM) branch
        self.memory = S4DLayer(hidden_dim, d_state=d_state)
        
        # 3. Local bypass
        self.w_local = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.to(DEVICE)

    def _spectral_conv(self, x):
        """Standard 1D spectral convolution."""
        B, N, C = x.shape
        x_ft = torch.fft.rfft(x, dim=1)
        nm = min(self.n_modes, x_ft.shape[1])
        
        # Complex multiply
        out_ft = torch.zeros_like(x_ft, dtype=torch.complex64, device=DEVICE)
        out_ft[:, :nm] = torch.einsum("bnc,nco->bno", x_ft[:, :nm], self.w[:nm])
            
        return torch.fft.irfft(out_ft, n=N, dim=1)

    def forward(self, x):
        # x: [B, N, H]
        x_norm = self.norm(x)
        
        # Parallel branches
        fourier_out = self._spectral_conv(x_norm)
        memory_out = self.memory(x_norm) # Memory sees full sequence
        local_out = self.w_local(x_norm)
        
        # Fuse with activation
        return x + nn.functional.gelu(fourier_out + memory_out + local_out)

class MemNO1d(nn.Module):
    """Memory Neural Operator architecture."""
    def __init__(self, hidden_dim=64, n_layers=4, n_modes=16, d_state=64, in_ch=2):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        
        self.blocks = nn.ModuleList([
            MemNOBlock1d(hidden_dim, n_modes, d_state) for _ in range(n_layers)
        ])
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.to(DEVICE)

    def forward(self, u0):
        # Handle multi-dimensional inputs (e.g., 2D Darcy/NS) by flattening
        B = u0.shape[0]
        if u0.ndim > 2:
            spatial_shape = u0.shape[1:]
            N = 1
            for s in spatial_shape: N *= s
            u0_flat = u0.reshape(B, N)
        else:
            N = u0.shape[1]
            u0_flat = u0

        grid = torch.linspace(0.0, 1.0, N, device=DEVICE).view(1, N, 1).repeat(B, 1, 1)
        x = torch.cat([u0_flat.unsqueeze(-1), grid], dim=-1)
        
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
            
        return self.proj(x)[:, :, 0]

if __name__ == "__main__":
    from core.device import DEVICE
    model = MemNO1d().to(DEVICE)
    u0 = torch.randn(2, 64).to(DEVICE)
    y = model(u0)
    print(f"MemNO output shape: {y.shape}")
