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
import mlx.core as mx
import mlx.nn as nn


# ── Haar Wavelet Transform ────────────────────────────────────────────────────

def _haar_forward(x: mx.array, levels: int) -> tuple[list[mx.array], mx.array]:
    """Multi-level Haar forward transform.

    Args:
        x      : [B, N, C] - spatial signal
        levels : number of decomposition levels

    Returns:
        details : list of length `levels`, details[0] is finest scale
                  each entry is [B, N // 2^(k+1), C]
        approx  : [B, N // 2^levels, C] - final low-pass approximation
    """
    details: list[mx.array] = []
    approx = x
    sq2 = math.sqrt(2)
    for _ in range(levels):
        B, N, C = approx.shape
        # Reshape into pairs first — avoids stride-2 slicing which can
        # produce weak/zero gradients through MLX's autograd engine.
        pairs = approx.reshape(B, N // 2, 2, C)
        lo = (pairs[:, :, 0, :] + pairs[:, :, 1, :]) / sq2
        hi = (pairs[:, :, 0, :] - pairs[:, :, 1, :]) / sq2
        details.append(hi)
        approx = lo
    return details, approx


def _haar_inverse(details: list[mx.array], approx: mx.array) -> mx.array:
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
        x = mx.stack([even, odd], axis=2).reshape(B, N2 * 2, C)
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
        self.W = [nn.Linear(in_ch, out_ch) for _ in range(n_levels + 1)]

    def __call__(self, x: mx.array) -> mx.array:
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

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.wav(x) + self.w(x))


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
        self.blocks = [WNOBlock1d(hidden_dim, n_levels) for _ in range(n_layers)]
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, u0: mx.array) -> mx.array:
        B, N = u0.shape
        grid = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N), (B, N))
        x    = mx.stack([u0, grid], axis=-1)
        x    = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x    = nn.gelu(self.proj1(x))
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
        self.gate = mx.zeros([1, 1, dims]) 
        
        self.ln2 = nn.LayerNorm(dims)
        self.mlp = nn.Sequential(
            nn.Linear(dims, mlp_ratio * dims),
            nn.GELU(),
            nn.Linear(mlp_ratio * dims, dims),
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.ln1(x)
        # Spatial Path (Transformer)
        x_attn = self.attn(h, h, h)
        # Multi-scale Path (Wavelet)
        x_wav = self.wav(h)
        
        # Gated fusion
        g = mx.sigmoid(self.gate)
        x = x + g * x_wav + (1 - g) * x_attn
        
        x = x + self.mlp(self.ln2(x))
        return x

class WNO_GNOT(nn.Module):
    """Wavelet-Transformer Hybrid Model (Phase 9 Breakthrough)."""
    def __init__(self, hidden_dim: int, n_layers: int, n_levels: int = 3, n_heads: int = 4, in_channels: int = 1):
        super().__init__()
        self.lift = nn.Linear(in_channels + 1, hidden_dim)
        self.blocks = [WNO_GNOT_Block(hidden_dim, n_levels, n_heads) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, in_channels)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 2:
            B, N = x.shape
            x = x[..., None]
        else:
            B, N, _ = x.shape
            
        grid = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N, 1), (B, N, 1))
        x = mx.concatenate([x, grid], axis=-1)
        
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        
        x = nn.gelu(self.proj1(self.norm(x)))
        out = self.proj2(x)
        if out.shape[-1] == 1:
            return out[:, :, 0]
        return out
