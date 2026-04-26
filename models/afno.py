"""Adaptive Fourier Neural Operator (AFNO) for 1-D problems (PyTorch/CUDA).

Implements the block-diagonal MLP in Fourier space with softshrink sparsity
from Guibas et al. (2022) "Adaptive Fourier Neural Operators: Efficient Token
Mixers for Transformers" (ICLR 2022).

Reference: arXiv:2111.13587
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockDiagMLP(nn.Module):
    """Block-diagonal 2-layer MLP applied independently to channel groups."""

    def __init__(self, channels: int, block_size: int = None):
        super().__init__()
        if block_size is None:
            block_size = channels

        while channels % block_size != 0 and block_size > 1:
            block_size //= 2
        assert channels % block_size == 0
        self.n_blocks   = channels // block_size
        self.block_size = block_size
        
        bound = 1.0 / math.sqrt(block_size)
        self.W1 = nn.Parameter(torch.empty(self.n_blocks, block_size, block_size).uniform_(-bound, bound))
        self.W2 = nn.Parameter(torch.empty(self.n_blocks, block_size, block_size).uniform_(-bound, bound))
        self.b1 = nn.Parameter(torch.empty(self.n_blocks, block_size).uniform_(-bound, bound))
        self.b2 = nn.Parameter(torch.empty(self.n_blocks, block_size).uniform_(-bound, bound))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N_flat, channels] → [N_flat, channels]"""
        N_flat = x.shape[0]
        nb     = self.n_blocks
        K      = self.block_size
        xb = x.reshape(N_flat, nb, K)
        h  = F.gelu(torch.einsum("bnk,nkj->bnj", xb, self.W1) + self.b1.unsqueeze(0))
        return (torch.einsum("bnk,nkj->bnj", h, self.W2) + self.b2.unsqueeze(0)).reshape(N_flat, -1)


def _softshrink(x: torch.Tensor, lam: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)


class AdaptiveSpectralMixer1d(nn.Module):
    """AFNO spectral mixing layer."""

    def __init__(self, channels: int, n_modes: int,
                 block_size: int = None, sparsity: float = 0.0):
        super().__init__()
        self.n_modes  = n_modes
        self.channels = channels
        self.sparsity = sparsity

        if block_size is None:
            block_size = 32

        self.mixer = BlockDiagMLP(2 * channels, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        m       = self.n_modes
        x_ft    = torch.fft.rfft(x, dim=1)

        xr = x_ft[:, :m, :].real
        xi = x_ft[:, :m, :].imag

        ri_flat = torch.cat([xr, xi], dim=-1).reshape(B * m, 2 * C)

        out     = self.mixer(ri_flat)
        out     = out.reshape(B, m, 2 * C)
        out_r   = out[:, :, :C]
        out_i   = out[:, :, C:]

        if self.sparsity > 0:
            out_r = _softshrink(out_r, self.sparsity)
            out_i = _softshrink(out_i, self.sparsity)

        out_modes = torch.complex(out_r, out_i)
        n_rfft    = N // 2 + 1
        if n_rfft > m:
            pad    = torch.zeros(B, n_rfft - m, C, device=x.device, dtype=torch.complex64)
            out_ft = torch.cat([out_modes, pad], dim=1)
        else:
            out_ft = out_modes

        return torch.fft.irfft(out_ft, n=N, dim=1)


class AFNOBlock1d(nn.Module):
    """Transformer-style two-stage residual AFNO block (Pre-LN)."""

    def __init__(self, channels: int, n_modes: int,
                 block_size: int = None, sparsity: float = 0.0, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.spec  = AdaptiveSpectralMixer1d(channels, n_modes, block_size, sparsity)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp   = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(channels * mlp_ratio, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.spec(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AFNO1d(nn.Module):
    """Adaptive Fourier Neural Operator (v2) for 1-D operator learning."""

    def __init__(self, n_modes: int = 32, hidden_dim: int = 128, n_layers: int = 8,
                 in_ch: int = 2, block_size: int = None, sparsity: float = 0.0):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([
            AFNOBlock1d(hidden_dim, n_modes, block_size, sparsity)
            for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(hidden_dim)
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        if u0.ndim == 1:
            u0 = u0.unsqueeze(0)
        B, N  = u0.shape
        grid  = torch.linspace(0.0, 1.0, N, device=u0.device).view(1, N).expand(B, N)
        x     = torch.stack([u0, grid], dim=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(self.norm(x)))
        return self.proj2(x)[:, :, 0]


class DiagSpectralConv1d(nn.Module):
    """Diagonal (factorized) spectral convolution."""

    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.n_modes  = n_modes
        scale = 0.02
        self.weights = nn.Parameter(torch.randn(n_modes, channels, dtype=torch.complex64) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        n_rfft  = N // 2 + 1
        m       = min(self.n_modes, n_rfft)
        x_ft    = torch.fft.rfft(x, dim=1)
        
        out_modes = x_ft[:, :m, :] * self.weights[:m].unsqueeze(0)

        if n_rfft > m:
            pad    = torch.zeros(B, n_rfft - m, C, device=x.device, dtype=torch.complex64)
            out_ft = torch.cat([out_modes, pad], dim=1)
        else:
            out_ft = out_modes
        return torch.fft.irfft(out_ft, n=N, dim=1)


class FFNOBlock1d(nn.Module):
    """Factorized FNO block."""

    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.spec = DiagSpectralConv1d(channels, n_modes)
        self.w    = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + F.gelu(self.spec(h) + self.w(h))


class FFNO1d(nn.Module):
    """Factorized Fourier Neural Operator for 1-D operator learning."""

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int,
                 in_ch: int = 2):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([FFNOBlock1d(hidden_dim, n_modes) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(hidden_dim)
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        if u0.ndim == 1:
            u0 = u0.unsqueeze(0)
        B, N  = u0.shape
        grid  = torch.linspace(0.0, 1.0, N, device=u0.device).view(1, N).expand(B, N)
        x     = torch.stack([u0, grid], dim=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(self.norm(x)))
        return self.proj2(x)[:, :, 0]
