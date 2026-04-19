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
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockDiagMLP(nn.Module):
    """Block-diagonal 2-layer MLP applied independently to channel groups.

    Divides the channel dimension C into n_blocks = C // block_size groups.
    Each group is processed by the SAME small MLP (weight sharing across blocks).
    Input dim = C, output dim = C.

    For AFNO usage, call with C = 2 * hidden_dim so that real and imaginary
    parts of the Fourier coefficients are processed jointly.
    """

    def __init__(self, channels: int, block_size: int = None):
        super().__init__()
        # Default to full MLP (block_size = channels)
        if block_size is None:
            block_size = channels

        # Auto-adjust block_size if channels not divisible
        while channels % block_size != 0 and block_size > 1:
            block_size //= 2
        assert channels % block_size == 0
        self.n_blocks   = channels // block_size
        self.block_size = block_size

        # Improved initialization with proper bounds
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
        h  = F.gelu(torch.einsum("bnk,nkj->bnj", xb, self.W1) + self.b1[None])
        return (torch.einsum("bnk,nkj->bnj", h, self.W2) + self.b2[None]).reshape(N_flat, -1)


def _softshrink(x: torch.Tensor, lam: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)


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
                 block_size: int = None, sparsity: float = 0.0):
        super().__init__()
        self.n_modes  = n_modes
        self.channels = channels
        self.sparsity = sparsity

        # Mixer expressivity: default to 16 or 32 (whichever is larger but divisible)
        if block_size is None:
            # For hidden_dim=128, 2*channels=256. 256 % 32 == 0.
            # BlockDiagMLP will auto-reduce if not divisible.
            block_size = 32

        # MLP operates on 2*channels (real + imag stacked)
        self.mixer = BlockDiagMLP(2 * channels, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        m       = self.n_modes
        x_ft    = torch.fft.rfft(x, dim=1)         # [B, N//2+1, C] complex

        xr = x_ft[:, :m, :].real                   # [B, m, C]
        xi = x_ft[:, :m, :].imag

        # Simplified complex coupling: concatenate real and imag
        # This allows the MLP to learn the full xr*wr - xi*wi complex interaction.
        # [B, m, C] + [B, m, C] -> [B, m, 2C] -> [B*m, 2C]
        ri_flat = torch.cat([xr, xi], dim=-1).reshape(B * m, 2 * C)

        out     = self.mixer(ri_flat)               # [B*m, 2C]
        out     = out.reshape(B, m, 2 * C)
        out_r   = out[:, :, :C]
        out_i   = out[:, :, C:]

        # Learned sparsity
        if self.sparsity > 0:
            out_r = _softshrink(out_r, self.sparsity)
            out_i = _softshrink(out_i, self.sparsity)

        # Pad to full rfft size and inverse transform
        out_modes = out_r + 1j * out_i
        n_rfft    = N // 2 + 1
        if n_rfft > m:
            pad    = torch.zeros(B, n_rfft - m, C, dtype=x_ft.dtype, device=x.device)
            out_ft = torch.cat([out_modes, pad], dim=1)
        else:
            out_ft = out_modes

        return torch.fft.irfft(out_ft, n=N, dim=1)


class AFNOBlock1d(nn.Module):
    """Transformer-style two-stage residual AFNO block (Pre-LN).

    Phase 1: Spectral Mixing (Token mixing)
    Phase 2: Channel Mixing (Feed-forward)
    """

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
        # Phase 1: Spectral Mixing
        x = x + self.spec(self.norm1(x))
        # Phase 2: Channel Mixing (Feed-Forward)
        x = x + self.mlp(self.norm2(x))
        return x


class AFNO1d(nn.Module):
    """Adaptive Fourier Neural Operator (v2) for 1-D operator learning.

    Uses shared block-diagonal MLP with joint real+imag processing for
    correct complex-valued Fourier mode mixing.

    Hyperparameter guide:
        n_modes    = 32    : more modes due to parameter efficiency
        hidden_dim = 128   : same as best FNO
        n_layers   = 8     : start here; Pre-LN residuals allow going deeper
        block_size = None  : default to full MLP (no block-diagonal)
        sparsity   = 0.0   : softshrink threshold (default 0.0 for stability)
    """

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
        B, N  = u0.shape
        grid  = torch.linspace(0.0, 1.0, N, device=u0.device, dtype=u0.dtype).unsqueeze(0).expand(B, -1)
        x     = torch.stack([u0, grid], dim=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(self.norm(x)))
        return self.proj2(x)[:, :, 0]



# ── Factorized / Diagonal FNO ─────────────────────────────────────────────────

class DiagSpectralConv1d(nn.Module):
    """Diagonal (factorized) spectral convolution.

    Instead of a full W[m, in_c, out_c] matrix per mode (O(m*C²) params),
    uses a per-mode per-channel diagonal scale (O(m*C) params - 128x cheaper
    for C=128, m=24).

    Combined with the pointwise Linear 'w' in FNOBlock (which provides full
    channel mixing), this achieves similar expressivity to full SpectralConv
    with far fewer parameters - enabling much wider channels in the same budget.

    Reference: inspired by Tran et al. (2023) Factorized FNO (arXiv:2111.13587)
    """

    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.n_modes  = n_modes
        scale = 0.02
        # Per-mode per-channel complex weights (diagonal in channel dim)
        self.wr = nn.Parameter(torch.randn(n_modes, channels) * scale)
        self.wi = nn.Parameter(torch.randn(n_modes, channels) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        n_rfft  = N // 2 + 1
        m       = min(self.n_modes, n_rfft)     # clamp to available rfft modes
        x_ft    = torch.fft.rfft(x, dim=1)
        xr      = x_ft[:, :m, :].real          # [B, m, C]
        xi      = x_ft[:, :m, :].imag

        # Diagonal complex multiply: elementwise per (mode, channel)
        out_r = xr * self.wr[:m].unsqueeze(0) - xi * self.wi[:m].unsqueeze(0)
        out_i = xr * self.wi[:m].unsqueeze(0) + xi * self.wr[:m].unsqueeze(0)

        out_modes = out_r + 1j * out_i
        if n_rfft > m:
            pad    = torch.zeros(B, n_rfft - m, C, dtype=x_ft.dtype, device=x.device)
            out_ft = torch.cat([out_modes, pad], dim=1)
        else:
            out_ft = out_modes
        return torch.fft.irfft(out_ft, n=N, dim=1)


class FFNOBlock1d(nn.Module):
    """Factorized FNO block: diagonal spectral conv + pointwise linear + GELU.
    Added LayerNorm and residual connection for depth stability.
    """

    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.spec = DiagSpectralConv1d(channels, n_modes)
        self.w    = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + F.gelu(self.spec(h) + self.w(h))


class FFNO1d(nn.Module):
    """Factorized Fourier Neural Operator for 1-D operator learning.

    Uses diagonal (per-mode per-channel) spectral weights instead of
    FNO's full (per-mode per-channel-pair) weight matrices.  128x fewer
    spectral parameters - can afford much wider channels (C=256/512) or
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
        self.blocks = nn.ModuleList([FFNOBlock1d(hidden_dim, n_modes) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(hidden_dim)
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N  = u0.shape
        grid  = torch.linspace(0.0, 1.0, N, device=u0.device, dtype=u0.dtype).unsqueeze(0).expand(B, -1)
        x     = torch.stack([u0, grid], dim=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(self.norm(x)))
        return self.proj2(x)[:, :, 0]
