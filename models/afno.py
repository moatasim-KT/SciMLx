"""Adaptive Fourier Neural Operator (AFNO) for 1-D problems.

Implements the block-diagonal MLP in Fourier space with softshrink sparsity
from Guibas et al. (2022) "Adaptive Fourier Neural Operators: Efficient Token
Mixers for Transformers" (ICLR 2022).

Key differences vs FNO:
  - FNO: learned linear map W[m, in_c, out_c] per Fourier mode
  - AFNO: shared block-diagonal 2-layer MLP across all modes + softshrink

Advantages:
  - Non-linear Fourier mode mixing (vs FNO's linear)
  - Learned sparsity via softshrink — focuses on shock-relevant frequencies
  - Same parameter budget as FNO; scales better in channel width

Reference: arXiv:2111.13587
"""

import math
import mlx.core as mx
import mlx.nn as nn


class BlockDiagMLP(nn.Module):
    """Block-diagonal 2-layer MLP applied independently to channel groups.

    Divides the channel dimension into n_blocks = channels // block_size groups.
    Each group is processed by the same small MLP (weight sharing).
    This is the "adaptive" mixer in AFNO.

    Args:
        channels:   total channel width (must be divisible by block_size)
        block_size: channels per block (default 16; try 8 or 32)
    """

    def __init__(self, channels: int, block_size: int = 16):
        super().__init__()
        assert channels % block_size == 0, (
            f"channels ({channels}) must be divisible by block_size ({block_size})"
        )
        self.n_blocks  = channels // block_size
        self.block_size = block_size
        scale = block_size ** -0.5
        # Shared across all modes — weight matrices for each block
        self.W1 = mx.random.normal([self.n_blocks, block_size, block_size]) * scale
        self.W2 = mx.random.normal([self.n_blocks, block_size, block_size]) * scale
        self.b1 = mx.zeros([self.n_blocks, block_size])
        self.b2 = mx.zeros([self.n_blocks, block_size])

    def __call__(self, x: mx.array) -> mx.array:
        """x: [..., channels] → [..., channels]"""
        shape = x.shape
        C     = shape[-1]
        B_dim = math.prod(shape[:-1])
        nb    = self.n_blocks
        K     = self.block_size

        # Reshape to [B_flat, n_blocks, block_size]
        xb = x.reshape(B_dim, nb, K)

        # Layer 1: [B_flat, nb, K] x [nb, K, K] → [B_flat, nb, K]
        h = mx.einsum("bnk,nkj->bnj", xb, self.W1) + self.b1[None]
        h = nn.gelu(h)

        # Layer 2
        out = mx.einsum("bnk,nkj->bnj", h, self.W2) + self.b2[None]

        return out.reshape(*shape)


def _softshrink(x: mx.array, lam: float) -> mx.array:
    """SoftShrink: zeros out values with |x| < lam, shrinks others by lam."""
    return mx.sign(x) * mx.maximum(mx.abs(x) - lam, 0.0)


class AdaptiveSpectralMixer1d(nn.Module):
    """AFNO spectral mixing layer for 1-D signals.

    1. rfft  → Fourier domain [B, n_rfft, C]
    2. Keep lowest n_modes; apply BlockDiagMLP to real & imag independently
    3. SoftShrink for learned sparsity
    4. irfft → physical domain

    Because the MLP is shared across all modes (not per-mode weights),
    this has identical parameter count to SpectralConv1d when block_size=C.
    With block_size < C it uses fewer params — allowing wider models.
    """

    def __init__(self, channels: int, n_modes: int,
                 block_size: int = 16, sparsity: float = 0.01):
        super().__init__()
        if channels % block_size != 0:
            # Fall back to largest valid block_size
            block_size = max(b for b in [1, 2, 4, 8, 16, 32]
                             if channels % b == 0 and b <= block_size)
        self.n_modes  = n_modes
        self.sparsity = sparsity
        self.mixer    = BlockDiagMLP(channels, block_size)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        x_ft    = mx.fft.rfft(x, axis=1)           # [B, N//2+1, C] complex

        # Select lowest n_modes
        xr = x_ft[:, :self.n_modes, :].real        # [B, m, C]
        xi = x_ft[:, :self.n_modes, :].imag

        # Block-diagonal MLP (shared weights across all modes)
        # Flatten mode × batch dimension so BlockDiagMLP processes [B*m, nb, K]
        B_, m = B, self.n_modes
        xr_flat = xr.reshape(B_ * m, C)
        xi_flat = xi.reshape(B_ * m, C)

        out_r = self.mixer(xr_flat).reshape(B_, m, C)
        out_i = self.mixer(xi_flat).reshape(B_, m, C)

        # Learned sparsity: suppress near-zero activations
        out_r = _softshrink(out_r, self.sparsity)
        out_i = _softshrink(out_i, self.sparsity)

        # Pad back to full rfft size and inverse transform
        out_modes = out_r + 1j * out_i
        n_rfft    = N // 2 + 1
        if n_rfft > self.n_modes:
            pad    = mx.zeros([B, n_rfft - self.n_modes, C], dtype=mx.complex64)
            out_ft = mx.concatenate([out_modes, pad], axis=1)
        else:
            out_ft = out_modes

        return mx.fft.irfft(out_ft, n=N, axis=1)


class AFNOBlock1d(nn.Module):
    """AFNO layer with Pre-LN residual: x = x + AFNO(LN(x)) + w(LN(x)).

    Uses the same Pre-LN residual pattern as RFNO1d for stability.
    """

    def __init__(self, channels: int, n_modes: int,
                 block_size: int = 16, sparsity: float = 0.01):
        super().__init__()
        self.norm    = nn.LayerNorm(channels)
        self.spec    = AdaptiveSpectralMixer1d(channels, n_modes, block_size, sparsity)
        self.w       = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)
        return x + nn.gelu(self.spec(h) + self.w(h))


class AFNO1d(nn.Module):
    """Adaptive Fourier Neural Operator for 1-D operator learning.

    Drop-in replacement for FNO1d / RFNO1d.  Uses non-linear (MLP) Fourier
    mode mixing with learned sparsity instead of learned linear maps.

    Hyperparameter guide:
        n_modes    = 24    : same as best FNO config
        hidden_dim = 128   : same width as best FNO config
        n_layers   = 8–12  : Pre-LN residuals allow deeper stacks
        block_size = 16    : channel block size for block-diagonal MLP
        sparsity   = 0.01  : softshrink threshold (0 = disabled)

    Expected improvement over FNO: 5–20% on Burgers (non-linear mixing
    captures more complex shock dynamics).
    """

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int,
                 in_ch: int = 2, block_size: int = 16, sparsity: float = 0.01):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = [
            AFNOBlock1d(hidden_dim, n_modes, block_size, sparsity)
            for _ in range(n_layers)
        ]
        self.norm   = nn.LayerNorm(hidden_dim)
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, u0: mx.array) -> mx.array:
        B, N  = u0.shape
        grid  = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N), (B, N))
        x     = mx.stack([u0, grid], axis=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = nn.gelu(self.proj1(self.norm(x)))
        return self.proj2(x)[:, :, 0]
