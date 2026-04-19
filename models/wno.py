"""Wavelet Neural Operator (WNO) for 1-D problems.

Unlike FNO (which uses global Fourier modes), WNO applies multi-resolution
Haar wavelet decomposition.  This is better suited to:
  - Non-periodic boundary conditions (e.g., Darcy flow)
  - Problems with localised shocks or sharp features (e.g., Burgers at low nu)
  - Multi-scale phenomena where both global and local features matter

Reference:
  Tripura & Chakraborty (2022) "Wavelet Neural Operator for solving parametric
  partial differential equations in computational mechanics problems"
  arXiv:2205.02191
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Haar Wavelet Transform ────────────────────────────────────────────────────

def _haar_forward(x: torch.Tensor, levels: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Multi-level Haar forward transform.

    Args:
        x      : [B, N, C] - spatial signal
        levels : number of decomposition levels

    Returns:
        details : list of length `levels`, details[0] is finest scale
                  each entry is [B, N // 2^(k+1), C]
        approx  : [B, N // 2^levels, C] - final low-pass approximation
    """
    details: list[torch.Tensor] = []
    approx = x
    sq2 = math.sqrt(2)
    for _ in range(levels):
        B, N, C = approx.shape
        # Reshape into pairs first — avoids stride-2 slicing which can
        # produce weak/zero gradients through PyTorch's autograd engine.
        pairs = approx.reshape(B, N // 2, 2, C)
        lo = (pairs[:, :, 0, :] + pairs[:, :, 1, :]) / sq2
        hi = (pairs[:, :, 0, :] - pairs[:, :, 1, :]) / sq2
        details.append(hi)
        approx = lo
    return details, approx


def _haar_inverse(details: list[torch.Tensor], approx: torch.Tensor) -> torch.Tensor:
    """Multi-level Haar inverse transform.

    Args:
        details : list of detail coefficient arrays (finest last → coarsest
                  first in reversed order)
        approx  : coarsest approximation

    Returns:
        [B, N, C] - reconstructed signal
    """
    x  = approx
    sq2 = math.sqrt(2)
    for hi in reversed(details):
        lo   = x
        even = (lo + hi) / sq2   # reconstructed even-index samples
        odd  = (lo - hi) / sq2   # reconstructed odd-index samples
        B, N2, C = lo.shape
        # Interleave: stack on new axis then reshape → correct interleaving
        x = torch.stack([even, odd], dim=2).reshape(B, N2 * 2, C)
    return x


# ── WNO Layers ────────────────────────────────────────────────────────────────

class WaveletConv1d(nn.Module):
    """Multi-level Haar wavelet convolution layer.

    Decomposes the input, applies independent learned linear projections at
    each scale, then reconstructs.  Analogous to SpectralConv1d but in wavelet
    space.
    """

    def __init__(self, in_ch: int, out_ch: int, n_levels: int = 3):
        super().__init__()
        self.n_levels = n_levels
        # One linear per detail level + one for the approximation
        self.W = nn.ModuleList([nn.Linear(in_ch, out_ch) for _ in range(n_levels + 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        details, approx = _haar_forward(x, self.n_levels)
        new_details = [self.W[k](d) for k, d in enumerate(details)]
        new_approx  = self.W[-1](approx)
        return _haar_inverse(new_details, new_approx)


class WNOBlock1d(nn.Module):
    """WNO layer: wavelet conv + pointwise linear + GELU residual."""

    def __init__(self, channels: int, n_levels: int = 3):
        super().__init__()
        self.wav = WaveletConv1d(channels, channels, n_levels)
        self.w   = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.wav(x) + self.w(x))


# ── Full WNO Model ────────────────────────────────────────────────────────────

class WNO1d(nn.Module):
    """Wavelet Neural Operator for 1-D operator learning.

    Drop-in replacement for FNO1d.  Uses N_LEVELS instead of N_MODES.

    Hyperparameter guide:
        n_levels = 3  : good default for N=64 (decompose to N//8=8 points)
        hidden_dim    : same as FNO (32-128)
        n_layers      : same as FNO (4-6)

    Works well on:
        - burgers_1d   (localised shock, benefits from multi-scale)
        - Non-periodic domains (Haar wavelets have compact support)
    """

    def __init__(self, n_levels: int = 3, hidden_dim: int = 32,
                 n_layers: int = 4, in_ch: int = 2):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([WNOBlock1d(hidden_dim, n_levels) for _ in range(n_layers)])
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N = u0.shape
        grid = torch.linspace(0.0, 1.0, N, device=u0.device, dtype=u0.dtype).unsqueeze(0).expand(B, -1)
        x    = torch.stack([u0, grid], dim=-1)
        x    = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x    = F.gelu(self.proj1(x))
        return self.proj2(x)[:, :, 0]

class WNO_GNOT_Block(nn.Module):
    """Hybrid Wavelet-Attention block for multi-scale discontinuity capture.

    Combines Haar wavelet localization with Transformer spatial coordination.
    """
    def __init__(self, dims: int, n_levels: int, n_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        # Import attention from gnot to reuse implementation
        from models.gnot import MultiHeadAttention
        self.ln1 = nn.LayerNorm(dims)
        self.attn = MultiHeadAttention(dims, n_heads)
        self.wav = WaveletConv1d(dims, dims, n_levels)

        # Learnable gate for wavelet vs spatial weighting
        self.gate = nn.Parameter(torch.zeros(1, 1, dims))

        self.ln2 = nn.LayerNorm(dims)
        self.mlp = nn.Sequential(
            nn.Linear(dims, mlp_ratio * dims),
            nn.GELU(),
            nn.Linear(mlp_ratio * dims, dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        # Spatial Path (Transformer)
        x_attn = self.attn(h, h, h)
        # Multi-scale Path (Wavelet)
        x_wav = self.wav(h)

        # Gated fusion
        g = torch.sigmoid(self.gate)
        x = x + g * x_wav + (1 - g) * x_attn

        x = x + self.mlp(self.ln2(x))
        return x

class WNO_GNOT(nn.Module):
    """Wavelet-Transformer Hybrid Model (Phase 9 Breakthrough)."""
    def __init__(self, hidden_dim: int, n_layers: int, n_levels: int = 3, n_heads: int = 4, in_channels: int = 1):
        super().__init__()
        self.lift = nn.Linear(in_channels + 1, hidden_dim)
        self.blocks = nn.ModuleList([WNO_GNOT_Block(hidden_dim, n_levels, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            B, N = x.shape
            x = x[..., None]
        else:
            B, N, _ = x.shape

        grid = torch.linspace(0.0, 1.0, N, device=x.device, dtype=x.dtype).reshape(1, N, 1).expand(B, -1, -1)
        x = torch.cat([x, grid], dim=-1)

        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)

        x = F.gelu(self.proj1(self.norm(x)))
        out = self.proj2(x)
        if out.shape[-1] == 1:
            return out[:, :, 0]
        return out

# ── 2D Wavelet Components ───────────────────────────────────────────────────

def _haar2d_forward(x: torch.Tensor, levels: int):
    """2D Multi-level Haar forward transform."""
    details = []
    approx = x
    sq2 = math.sqrt(2)
    for _ in range(levels):
        B, H, W, C = approx.shape
        # Decompose rows
        rows = approx.reshape(B, H // 2, 2, W, C)
        L = (rows[:, :, 0, :, :] + rows[:, :, 1, :, :]) / sq2
        H_sub = (rows[:, :, 0, :, :] - rows[:, :, 1, :, :]) / sq2

        # Decompose columns
        cols_L = L.reshape(B, H // 2, W // 2, 2, C)
        LL = (cols_L[:, :, :, 0, :] + cols_L[:, :, :, 1, :]) / 2.0
        LH = (cols_L[:, :, :, 0, :] - cols_L[:, :, :, 1, :]) / 2.0

        cols_H = H_sub.reshape(B, H // 2, W // 2, 2, C)
        HL = (cols_H[:, :, :, 0, :] + cols_H[:, :, :, 1, :]) / 2.0
        HH = (cols_H[:, :, :, 0, :] - cols_H[:, :, :, 1, :]) / 2.0

        details.append((LH, HL, HH))
        approx = LL
    return details, approx

def _haar2d_inverse(details, approx):
    """2D Multi-level Haar inverse transform."""
    x = approx
    for (LH, HL, HH) in reversed(details):
        # Reconstruct L and H_sub
        L_even = (x + LH) # * sq2 / 2
        L_odd  = (x - LH)
        L = torch.stack([L_even, L_odd], dim=3).reshape(x.shape[0], x.shape[1], -1, x.shape[3])

        H_even = (HL + HH)
        H_odd  = (HL - HH)
        H_sub = torch.stack([H_even, H_odd], dim=3).reshape(x.shape[0], x.shape[1], -1, x.shape[3])

        # Reconstruct approx
        approx_even = (L + H_sub)
        approx_odd  = (L - H_sub)
        x = torch.stack([approx_even, approx_odd], dim=2).reshape(x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3])
    return x

class WaveletConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_levels: int = 3):
        super().__init__()
        self.n_levels = n_levels
        # 3 detail sub-bands per level + 1 approx
        self.W = nn.ModuleList([nn.Linear(in_ch, out_ch) for _ in range(3 * n_levels + 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        details, approx = _haar2d_forward(x, self.n_levels)
        new_details = []
        idx = 0
        for LH, HL, HH in details:
            new_details.append((self.W[idx](LH), self.W[idx+1](HL), self.W[idx+2](HH)))
            idx += 3
        new_approx = self.W[-1](approx)
        return _haar2d_inverse(new_details, new_approx)

class WNO2d(nn.Module):
    """Wavelet Neural Operator for 2D benchmarks."""
    def __init__(self, n_levels: int = 3, hidden_dim: int = 32, n_layers: int = 4, in_channels: int = 1):
        super().__init__()
        self.lift = nn.Linear(in_channels + 2, hidden_dim)
        self.blocks = nn.ModuleList([nn.Sequential(WaveletConv2d(hidden_dim, hidden_dim, n_levels), nn.GELU())
                        for _ in range(n_layers)])
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            B, N1, N2 = x.shape
            x = x[..., None]
        else:
            B, N1, N2, _ = x.shape

        grid1 = torch.linspace(0.0, 1.0, N1, device=x.device, dtype=x.dtype).reshape(1, N1, 1, 1).expand(B, -1, N2, -1)
        grid2 = torch.linspace(0.0, 1.0, N2, device=x.device, dtype=x.dtype).reshape(1, 1, N2, 1).expand(B, N1, -1, -1)
        x     = torch.cat([x, grid1, grid2], dim=-1)

        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.proj1(x))
        out = self.proj2(x)

        if out.shape[-1] == 1:
            return out[:, :, :, 0]
        return out
