"""Tucker-Factorized FNO (TFNO) and CP-Factorized FNO (CPFNO).

Inspired by PhysicsNeMo (NVIDIA Modulus) TFNO implementation and the paper:
  "Factorized Fourier Neural Operators" - Kossaifi et al., ICLR 2024
  arXiv: https://arxiv.org/abs/2111.13802
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from core.device import DEVICE


# ── Tucker spectral conv (1D) ─────────────────────────────────────────────────

class TuckerSpectralConv1d(nn.Module):
    """Tucker-factorized 1-D Fourier spectral convolution."""

    def __init__(self, in_ch: int, out_ch: int, n_modes: int,
                 rank_ratio: float = 0.5):
        super().__init__()
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.n_modes = n_modes
        r_m = max(1, int(n_modes    * rank_ratio))
        r_i = max(1, int(in_ch     * rank_ratio))
        r_o = max(1, int(out_ch    * rank_ratio))
        self.r_m, self.r_i, self.r_o = r_m, r_i, r_o
        scale = (in_ch * out_ch) ** -0.5
        # Tucker core (complex)
        self.G = nn.Parameter(torch.randn(r_m, r_i, r_o, dtype=torch.complex64) * scale)
        # Factor matrices (shared)
        self.Um = nn.Parameter(torch.randn(n_modes, r_m) * scale)
        self.Ui = nn.Parameter(torch.randn(in_ch,   r_i) * scale)
        self.Uo = nn.Parameter(torch.randn(out_ch,  r_o) * scale)
        self.to(DEVICE)

    def _reconstruct(self):
        """Reconstruct full [n_modes, in_ch, out_ch] weight from Tucker factors."""
        return torch.einsum("abc,ma,ib,oc->mio", self.G, self.Um, self.Ui, self.Uo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x_ft = torch.fft.rfft(x, dim=1)
        
        W = self._reconstruct()
        nm = min(self.n_modes, x_ft.shape[1])
        
        out_ft = torch.zeros_like(x_ft, dtype=torch.complex64, device=DEVICE)
        out_ft[:, :nm] = torch.einsum("bmi,mio->bmo", x_ft[:, :nm], W[:nm])
        
        return torch.fft.irfft(out_ft, n=N, dim=1)


# ── CP spectral conv (1D) ─────────────────────────────────────────────────────

class CPSpectralConv1d(nn.Module):
    """CP-factorized 1-D Fourier spectral convolution.

    W[m, i, o] ≈ Σ_r A[m,r] * B[i,r] * C[o,r]
    """

    def __init__(self, in_ch: int, out_ch: int, n_modes: int, rank: int = 8):
        super().__init__()
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.n_modes = n_modes
        self.rank    = rank
        scale = (in_ch * out_ch) ** -0.5
        # CP factors (A complex, B/C real)
        self.A = nn.Parameter(torch.randn(n_modes, rank, dtype=torch.complex64) * scale)
        self.B = nn.Parameter(torch.randn(in_ch,   rank) * scale)
        self.C = nn.Parameter(torch.randn(out_ch,  rank) * scale)
        self.to(DEVICE)

    def _reconstruct(self):
        return torch.einsum("mr,ir,or->mio", self.A, self.B, self.C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x_ft = torch.fft.rfft(x, dim=1)
        
        W = self._reconstruct()
        nm = min(self.n_modes, x_ft.shape[1])
        
        out_ft = torch.zeros_like(x_ft, dtype=torch.complex64, device=DEVICE)
        out_ft[:, :nm] = torch.einsum("bmi,mio->bmo", x_ft[:, :nm], W[:nm])
        
        return torch.fft.irfft(out_ft, n=N, dim=1)


# ── TFNO block (1D) ──────────────────────────────────────────────────────────

class TFNOBlock1d(nn.Module):
    """1D TFNO layer: Tucker-spectral conv + pointwise linear + GELU."""
    def __init__(self, channels: int, n_modes: int, rank_ratio: float = 0.5):
        super().__init__()
        self.spec = TuckerSpectralConv1d(channels, channels, n_modes, rank_ratio)
        self.w    = nn.Linear(channels, channels)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.w(x))


class TFNOBlockResidual1d(nn.Module):
    """Pre-LN residual TFNO block (Pre-LN + Tucker spectral conv)."""
    def __init__(self, channels: int, n_modes: int, rank_ratio: float = 0.5):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.spec = TuckerSpectralConv1d(channels, channels, n_modes, rank_ratio)
        self.w    = nn.Linear(channels, channels)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + F.gelu(self.spec(h) + self.w(h))


# ── TFNO1d ────────────────────────────────────────────────────────────────────

class TFNO1d(nn.Module):
    """Tucker-Factorized Fourier Neural Operator (1-D)."""

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int,
                 in_ch: int = 2, rank_ratio: float = 0.5):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([TFNOBlock1d(hidden_dim, n_modes, rank_ratio)
                       for _ in range(n_layers)])
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)
        self.to(DEVICE)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N  = u0.shape
        grid  = torch.linspace(0.0, 1.0, N, device=DEVICE).unsqueeze(0).expand(B, N)
        x     = torch.stack([u0, grid], dim=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(x))
        return self.proj2(x)[:, :, 0]


class RTFNO1d(nn.Module):
    """Residual Tucker-Factorized FNO (Pre-LN + Tucker spectral conv)."""

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int,
                 in_ch: int = 2, rank_ratio: float = 0.5):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([TFNOBlockResidual1d(hidden_dim, n_modes, rank_ratio)
                       for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(hidden_dim)
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)
        self.to(DEVICE)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N  = u0.shape
        grid  = torch.linspace(0.0, 1.0, N, device=DEVICE).unsqueeze(0).expand(B, N)
        x     = torch.stack([u0, grid], dim=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(self.norm(x)))
        return self.proj2(x)[:, :, 0]


# ── CPFNO1d ───────────────────────────────────────────────────────────────────

class CPFNOBlock1d(nn.Module):
    """1D CP-FNO layer."""
    def __init__(self, channels: int, n_modes: int, rank: int = 8):
        super().__init__()
        self.spec = CPSpectralConv1d(channels, channels, n_modes, rank)
        self.w    = nn.Linear(channels, channels)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.w(x))


class CPFNO1d(nn.Module):
    """CP-Factorized FNO (1-D). Most compact spectral operator."""

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int,
                 in_ch: int = 2, rank: int = 8):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([CPFNOBlock1d(hidden_dim, n_modes, rank)
                       for _ in range(n_layers)])
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)
        self.to(DEVICE)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N  = u0.shape
        grid  = torch.linspace(0.0, 1.0, N, device=DEVICE).unsqueeze(0).expand(B, N)
        x     = torch.stack([u0, grid], dim=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(x))
        return self.proj2(x)[:, :, 0]


# ── Tucker SpectralConv2d ─────────────────────────────────────────────────────

class TuckerSpectralConv2d(nn.Module):
    """Tucker-factorized 2-D Fourier spectral convolution."""

    def __init__(self, in_ch: int, out_ch: int,
                 n_modes1: int, n_modes2: int,
                 rank_ratio: float = 0.5):
        super().__init__()
        self.in_ch    = in_ch
        self.out_ch   = out_ch
        self.n_modes1 = n_modes1
        self.n_modes2 = n_modes2
        r1 = max(1, int(n_modes1 * rank_ratio))
        r2 = max(1, int(n_modes2 * rank_ratio))
        ri = max(1, int(in_ch    * rank_ratio))
        ro = max(1, int(out_ch   * rank_ratio))
        scale = (in_ch * out_ch) ** -0.5
        
        # Two Tucker cores (for two quadrants of rfft2)
        self.G1 = nn.Parameter(torch.randn(r1, r2, ri, ro, dtype=torch.complex64) * scale)
        self.G2 = nn.Parameter(torch.randn(r1, r2, ri, ro, dtype=torch.complex64) * scale)
        
        # Shared factor matrices
        self.Um1 = nn.Parameter(torch.randn(n_modes1, r1) * scale)
        self.Um2 = nn.Parameter(torch.randn(n_modes2, r2) * scale)
        self.Ui  = nn.Parameter(torch.randn(in_ch,    ri) * scale)
        self.Uo  = nn.Parameter(torch.randn(out_ch,   ro) * scale)
        self.to(DEVICE)

    def _reconstruct(self, G):
        return torch.einsum("abcd,ma,nb,ic,oc->mnio", G, self.Um1, self.Um2, self.Ui, self.Uo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N1, N2, _ = x.shape
        x_ft = torch.fft.rfft2(x, dim=(1, 2))
        out_ft = torch.zeros(B, N1, N2 // 2 + 1, self.out_ch, device=DEVICE, dtype=torch.complex64)

        W1 = self._reconstruct(self.G1)
        out_ft[:, :self.n_modes1, :self.n_modes2, :] = \
            torch.einsum("bmki,mkio->bmko", x_ft[:, :self.n_modes1, :self.n_modes2, :], W1)

        W2 = self._reconstruct(self.G2)
        out_ft[:, -self.n_modes1:, :self.n_modes2, :] = \
            torch.einsum("bmki,mkio->bmko", x_ft[:, -self.n_modes1:, :self.n_modes2, :], W2)

        return torch.fft.irfft2(out_ft, s=(N1, N2), dim=(1, 2))


class TFNOBlock2d(nn.Module):
    """2D TFNO layer."""
    def __init__(self, channels: int, n_modes1: int, n_modes2: int,
                 rank_ratio: float = 0.5):
        super().__init__()
        self.spec = TuckerSpectralConv2d(channels, channels, n_modes1, n_modes2, rank_ratio)
        self.w    = nn.Linear(channels, channels)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.w(x))


class TFNO2d(nn.Module):
    """Tucker-Factorized FNO for 2-D operator learning."""

    def __init__(self, n_modes1: int, n_modes2: int, hidden_dim: int, n_layers: int,
                 in_ch: int = 3, rank_ratio: float = 0.5):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([TFNOBlock2d(hidden_dim, n_modes1, n_modes2, rank_ratio)
                       for _ in range(n_layers)])
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)
        self.to(DEVICE)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N1, N2 = u0.shape
        grid1 = torch.linspace(0.0, 1.0, N1, device=DEVICE).view(1, N1, 1).expand(B, N1, N2)
        grid2 = torch.linspace(0.0, 1.0, N2, device=DEVICE).view(1, 1, N2).expand(B, N1, N2)
        x     = torch.stack([u0, grid1, grid2], dim=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(x))
        return self.proj2(x)[:, :, :, 0]
