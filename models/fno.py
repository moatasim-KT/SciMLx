"""Fourier Neural Operator (FNO) implementations (PyTorch/CUDA)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── 1D Spectral Components ────────────────────────────────────────────────────

class SpectralConv1d(nn.Module):
    """1-D Fourier spectral convolution."""

    def __init__(self, in_ch: int, out_ch: int, n_modes: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_modes = n_modes
        scale = (in_ch * out_ch) ** -0.5
        # PyTorch uses complex64 for complex weights
        self.weights = nn.Parameter(
            scale * torch.randn(n_modes, in_ch, out_ch, dtype=torch.complex64)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # x: [B, N, C]
        x_ft = torch.fft.rfft(x, dim=1)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B, N // 2 + 1, self.out_ch, device=x.device, dtype=torch.complex64)
        out_ft[:, :self.n_modes, :] = torch.einsum("bmi,mio->bmo", x_ft[:, :self.n_modes, :], self.weights)
        
        return torch.fft.irfft(out_ft, n=N, dim=1)


class FNOBlock1d(nn.Module):
    """1D FNO layer: spectral conv + pointwise linear + GELU."""
    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.spec = SpectralConv1d(channels, channels, n_modes)
        self.w = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.w(x))


class FNO1d(nn.Module):
    """Fourier Neural Operator for 1-D operator learning."""
    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, in_ch: int = 2):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([FNOBlock1d(hidden_dim, n_modes) for _ in range(n_layers)])
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        if u0.ndim == 1:
            u0 = u0.unsqueeze(0)
        B, N = u0.shape
        grid = torch.linspace(0.0, 1.0, N, device=u0.device).view(1, N).expand(B, N)
        x = torch.stack([u0, grid], dim=-1)
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.proj1(x))
        return self.proj2(x).squeeze(-1)


# ── Residual FNO (Pre-LN) ─────────────────────────────────────────────────────

class FNOBlockResidual1d(nn.Module):
    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.spec = SpectralConv1d(channels, channels, n_modes)
        self.w = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + F.gelu(self.spec(h) + self.w(h))


class RFNO1d(nn.Module):
    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, in_ch: int = 2):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([FNOBlockResidual1d(hidden_dim, n_modes) for _ in range(n_layers)])
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        if u0.ndim == 1:
            u0 = u0.unsqueeze(0)
        B, N = u0.shape
        grid = torch.linspace(0.0, 1.0, N, device=u0.device).view(1, N).expand(B, N)
        x = torch.stack([u0, grid], dim=-1)
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.proj1(x))
        return self.proj2(x).squeeze(-1)

# TODO: Add 2D FNO implementations
