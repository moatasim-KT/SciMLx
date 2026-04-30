"""Alias-Free Mamba Neural Operator (MambaNO) for SciML.

Reference: Zheng et al. (2024) "Alias-Free Mamba Neural Operator" (NeurIPS)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.device import DEVICE

def selective_scan_torch(u, delta, A, B, C, D=None):
    """
    Simplified selective scan in PyTorch.
    u: [B, L, D]
    delta: [B, L, D]
    A: [D, N]
    B: [B, L, N]
    C: [B, L, N]
    D: [D] (optional)
    """
    B_batch, L, D_dim = u.shape
    N = A.shape[1]

    h = torch.zeros(B_batch, D_dim, N, device=u.device)
    outputs = []
    
    for t in range(L):
        u_t = u[:, t, :]      # [B, D]
        d_t = delta[:, t, :]  # [B, D]
        b_t = B[:, t, :]      # [B, N]
        c_t = C[:, t, :]      # [B, N]
        
        # Discretize A and B (Zero-Order Hold)
        a_bar = torch.exp(d_t.unsqueeze(-1) * A.unsqueeze(0))   # [B, D, N]
        b_bar = d_t.unsqueeze(-1) * b_t.unsqueeze(1)        # [B, D, N]
        
        h = a_bar * h + b_bar * u_t.unsqueeze(-1)          # [B, D, N]
        y_t = torch.sum(h * c_t.unsqueeze(1), dim=-1)       # [B, D]
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1) # [B, L, D]
    
    if D is not None:
        y = y + u * D
        
    return y

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1
        )
        
        self.x_proj = nn.Linear(self.d_inner, 1 + self.d_state * 2)
        self.dt_proj = nn.Linear(1, self.d_inner)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, x):
        """x: [B, L, D]"""
        B, L, D = x.shape
        
        xz = self.in_proj(x) # [B, L, 2*D_inner]
        x, z = torch.chunk(xz, 2, dim=-1)
        
        # Conv path: PyTorch Conv1d expects [B, C, L]
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L] 
        x = x.transpose(1, 2) # [B, L, D_inner]
        x = F.silu(x)
        
        # SSM path
        x_dbl = self.x_proj(x) # [B, L, 1 + 2*d_state]
        dt, B_ssm, C_ssm = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt)) # [B, L, D_inner]
        A = -torch.exp(self.A_log) # [D_inner, d_state]
        
        y = selective_scan_torch(x, dt, A, B_ssm, C_ssm, self.D)
        
        # Output gating
        y = y * F.silu(z)
        return self.out_proj(y)

class SpectralFilter(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.modes = m
        scale = d ** -0.5
        self.w = nn.Parameter(scale * torch.randn(m, d, d, dtype=torch.complex64))
        
    def forward(self, x):
        B, N, C = x.shape
        x_ft = torch.fft.rfft(x, dim=1)
        m = min(self.modes, x_ft.shape[1])
        
        # Filter high frequencies (alias-free)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :m, :] = torch.einsum("bnc,nco->bno", x_ft[:, :m, :], self.w[:m])
        
        return torch.fft.irfft(out_ft, n=N, dim=1)

class MambaNO1d(nn.Module):
    """Alias-Free Mamba Neural Operator implementation."""
    def __init__(self, hidden_dim=64, n_layers=4, n_modes=16, in_ch=2):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        
        layers = []
        for _ in range(n_layers):
            layers.append(MambaBlock(hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(SpectralFilter(hidden_dim, n_modes))
        self.layers = nn.ModuleList(layers)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, u0):
        B = u0.shape[0]
        if u0.ndim > 2:
            spatial_shape = u0.shape[1:]
            N = 1
            for s in spatial_shape: N *= s
            u0_flat = u0.reshape(B, N)
        else:
            N = u0.shape[1]
            u0_flat = u0

        grid = torch.linspace(0.0, 1.0, N, device=u0.device).reshape(1, N, 1).expand(B, N, 1)
        x = torch.cat([u0_flat[..., None], grid], dim=-1)
        
        x = self.lift(x)
        for layer in self.layers:
            x = layer(x)
            
        return self.proj(x)[:, :, 0]
