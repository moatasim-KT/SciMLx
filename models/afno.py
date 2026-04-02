"""Adaptive Fourier Neural Operator (AFNO) for 1-D problems.

Implements the block-diagonal MLP in Fourier space with softshrink sparsity
from Guibas et al. (2022) "Adaptive Fourier Neural Operators: Efficient Token
Mixers for Transformers" (ICLR 2022).

Key differences vs FNO:
  - FNO: per-mode learned linear map W[m, in_c, out_c]
  - AFNO: shared block-diagonal 2-layer MLP across all modes + softshrink

Critical implementation note (v2 fix):
  The MLP must receive [real; imag] concatenated (2C features) so it can
  learn proper complex interactions.  Processing real and imaginary parts
  independently breaks the spectral convolution theorem — out_r must depend
  on both xr and xi (just like FNO: out_r = xr*wr - xi*wi).

Reference: arXiv:2111.13587
"""

import math
import mlx.core as mx
import mlx.nn as nn


class BlockDiagMLP(nn.Module):
    """Block-diagonal 2-layer MLP applied independently to channel groups.

    Divides the channel dimension C into n_blocks = C // block_size groups.
    Each group is processed by the SAME small MLP (weight sharing across blocks).
    Input dim = C, output dim = C.

    For AFNO usage, call with C = 2 * hidden_dim so that real and imaginary
    parts of the Fourier coefficients are processed jointly.
    """

    def __init__(self, channels: int, block_size: int = 32):
        super().__init__()
        # Auto-adjust block_size if channels not divisible
        while channels % block_size != 0 and block_size > 1:
            block_size //= 2
        assert channels % block_size == 0
        self.n_blocks   = channels // block_size
        self.block_size = block_size
        scale = block_size ** -0.5
        self.W1 = mx.random.normal([self.n_blocks, block_size, block_size]) * scale
        self.W2 = mx.random.normal([self.n_blocks, block_size, block_size]) * scale
        self.b1 = mx.zeros([self.n_blocks, block_size])
        self.b2 = mx.zeros([self.n_blocks, block_size])

    def __call__(self, x: mx.array) -> mx.array:
        """x: [N_flat, channels] → [N_flat, channels]"""
        N_flat = x.shape[0]
        nb     = self.n_blocks
        K      = self.block_size
        xb = x.reshape(N_flat, nb, K)
        h  = nn.gelu(mx.einsum("bnk,nkj->bnj", xb, self.W1) + self.b1[None])
        return (mx.einsum("bnk,nkj->bnj", h, self.W2) + self.b2[None]).reshape(N_flat, -1)


def _softshrink(x: mx.array, lam: float) -> mx.array:
    return mx.sign(x) * mx.maximum(mx.abs(x) - lam, 0.0)


class AdaptiveSpectralMixer1d(nn.Module):
    """AFNO spectral mixing layer (v2 — correct complex coupling).

    The MLP receives [real; imag] as a 2*channels vector so it can learn
    the full complex transformation (not just independent real/imag maps).

    Pipeline:
        x [B,N,C] → rfft → concat [xr;xi] [B*m, 2C]
                  → BlockDiagMLP(2C) → split → softshrink
                  → irfft → [B,N,C]
    """

    def __init__(self, channels: int, n_modes: int,
                 block_size: int = 32, sparsity: float = 0.01):
        super().__init__()
        self.n_modes  = n_modes
        self.channels = channels
        self.sparsity = sparsity
        # MLP operates on 2*channels (real + imag stacked)
        self.mixer = BlockDiagMLP(2 * channels, block_size)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        m       = self.n_modes
        x_ft    = mx.fft.rfft(x, axis=1)           # [B, N//2+1, C] complex

        xr = x_ft[:, :m, :].real                   # [B, m, C]
        xi = x_ft[:, :m, :].imag

        # Stack real & imag → [B*m, 2C] so MLP learns complex interactions
        ri_flat = mx.concatenate(
            [xr.reshape(B * m, C), xi.reshape(B * m, C)], axis=-1
        )                                           # [B*m, 2C]

        out     = self.mixer(ri_flat)               # [B*m, 2C]
        out_r   = out[:, :C].reshape(B, m, C)
        out_i   = out[:, C:].reshape(B, m, C)

        # Learned sparsity
        if self.sparsity > 0:
            out_r = _softshrink(out_r, self.sparsity)
            out_i = _softshrink(out_i, self.sparsity)

        # Pad to full rfft size and inverse transform
        out_modes = out_r + 1j * out_i
        n_rfft    = N // 2 + 1
        if n_rfft > m:
            pad    = mx.zeros([B, n_rfft - m, C], dtype=mx.complex64)
            out_ft = mx.concatenate([out_modes, pad], axis=1)
        else:
            out_ft = out_modes

        return mx.fft.irfft(out_ft, n=N, axis=1)


class AFNOBlock1d(nn.Module):
    """Pre-LN residual AFNO block: x = x + GELU(AFNO(LN(x)) + w(LN(x)))."""

    def __init__(self, channels: int, n_modes: int,
                 block_size: int = 32, sparsity: float = 0.01):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.spec = AdaptiveSpectralMixer1d(channels, n_modes, block_size, sparsity)
        self.w    = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)
        return x + nn.gelu(self.spec(h) + self.w(h))


class AFNO1d(nn.Module):
    """Adaptive Fourier Neural Operator (v2) for 1-D operator learning.

    Uses shared block-diagonal MLP with joint real+imag processing for
    correct complex-valued Fourier mode mixing.

    Hyperparameter guide:
        n_modes    = 24    : same sweet spot as FNO
        hidden_dim = 128   : same as best FNO
        n_layers   = 8     : start here; Pre-LN residuals allow going deeper
        block_size = 32    : channel block size (must divide 2*hidden_dim)
        sparsity   = 0.01  : softshrink threshold
    """

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int,
                 in_ch: int = 2, block_size: int = 32, sparsity: float = 0.01):
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


# ── Factorized / Diagonal FNO ─────────────────────────────────────────────────

class DiagSpectralConv1d(nn.Module):
    """Diagonal (factorized) spectral convolution.

    Instead of a full W[m, in_c, out_c] matrix per mode (O(m*C²) params),
    uses a per-mode per-channel diagonal scale (O(m*C) params — 128× cheaper
    for C=128, m=24).

    Combined with the pointwise Linear 'w' in FNOBlock (which provides full
    channel mixing), this achieves similar expressivity to full SpectralConv
    with far fewer parameters — enabling much wider channels in the same budget.

    Reference: inspired by Tran et al. (2023) Factorized FNO (arXiv:2111.13587)
    """

    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.n_modes  = n_modes
        scale = 0.02
        # Per-mode per-channel complex weights (diagonal in channel dim)
        self.wr = mx.random.normal([n_modes, channels]) * scale
        self.wi = mx.random.normal([n_modes, channels]) * scale

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        x_ft    = mx.fft.rfft(x, axis=1)
        xr      = x_ft[:, :self.n_modes, :].real   # [B, m, C]
        xi      = x_ft[:, :self.n_modes, :].imag

        # Diagonal complex multiply: elementwise per (mode, channel)
        out_r = xr * self.wr[None] - xi * self.wi[None]
        out_i = xr * self.wi[None] + xi * self.wr[None]

        out_modes = out_r + 1j * out_i
        n_rfft    = N // 2 + 1
        if n_rfft > self.n_modes:
            pad    = mx.zeros([B, n_rfft - self.n_modes, C], dtype=mx.complex64)
            out_ft = mx.concatenate([out_modes, pad], axis=1)
        else:
            out_ft = out_modes
        return mx.fft.irfft(out_ft, n=N, axis=1)


class FFNOBlock1d(nn.Module):
    """Factorized FNO block: diagonal spectral conv + pointwise linear + GELU.

    Same structure as FNOBlock1d but with DiagSpectralConv1d (much cheaper).
    The pointwise linear 'w' provides the full channel mixing that the
    diagonal spectral conv lacks.
    """

    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.spec = DiagSpectralConv1d(channels, n_modes)
        self.w    = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.spec(x) + self.w(x))


class FFNO1d(nn.Module):
    """Factorized Fourier Neural Operator for 1-D operator learning.

    Uses diagonal (per-mode per-channel) spectral weights instead of
    FNO's full (per-mode per-channel-pair) weight matrices.  128× fewer
    spectral parameters → can afford much wider channels (C=256/512) or
    more modes (m=32/48) in the same 5-minute compute budget.

    Key insight: the per-mode expressiveness comes from the channel-mixing
    Linear 'w' in each FFNOBlock, not the spectral conv.  The diagonal
    spectral conv provides per-mode frequency gating only.

    Hyperparameter guide:
        n_modes    = 32    : can afford more modes (cheap diagonal weights)
        hidden_dim = 256   : can afford wider channels (few spectral params)
        n_layers   = 8     : same depth as FNO sweet spot
    """

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int,
                 in_ch: int = 2):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = [FFNOBlock1d(hidden_dim, n_modes) for _ in range(n_layers)]
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, u0: mx.array) -> mx.array:
        B, N  = u0.shape
        grid  = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N), (B, N))
        x     = mx.stack([u0, grid], axis=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = nn.gelu(self.proj1(x))
        return self.proj2(x)[:, :, 0]
