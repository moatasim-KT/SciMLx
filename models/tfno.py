"""Tucker-Factorized FNO (TFNO) and CP-Factorized FNO (CPFNO).

Inspired by PhysicsNeMo (NVIDIA Modulus) TFNO implementation and the paper:
  "Factorized Fourier Neural Operators" - Kossaifi et al., ICLR 2024
  arXiv: https://arxiv.org/abs/2111.13802

Key idea: factorize the spectral weight tensor W[n_modes, in_ch, out_ch] using
Tucker or CP decompositions.

Tucker: W = G x1 Um x2 Ui x3 Uo
  - core G[r_m, r_i, r_o] + factor matrices Um[n_modes, r_m], Ui[in_ch, r_i], Uo[out_ch, r_o]
  - fewer params, implicit low-rank regularization, generalize better

CP: W[m,i,o] ≈ Σ_r A[m,r] * B[i,r] * C[o,r]
  - even more compact; R*(n_modes + in_ch + out_ch) params vs n_modes*in_ch*out_ch

Benefits over full FNO:
  - ~2-4x fewer spectral params -> can increase hidden_dim or n_layers at same memory
  - Low-rank inductive bias reduces overfitting on small datasets (e.g. darcy_2d)
  - PhysicsNeMo shows TFNO matches or beats FNO on Darcy, NS, MHD benchmarks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import FNOBlock1d, FNOBlock2d, SpectralConv2d


# ── Tucker spectral conv (1D) ─────────────────────────────────────────────────

class TuckerSpectralConv1d(nn.Module):
    """Tucker-factorized 1-D Fourier spectral convolution.

    W[m, i, o] = Σ_{a,b,c} G[a,b,c] * Um[m,a] * Ui[i,b] * Uo[o,c]

    Parameters
    ----------
    in_ch, out_ch : int
        Channel counts.
    n_modes : int
        Number of Fourier modes kept.
    rank_ratio : float
        Tucker rank as fraction of each dimension (default 0.5 → ~75% param reduction
        for equal in/out channels; may be raised to 0.75 for wider models).
    """

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
        # Tucker core (real + imag)
        self.Gr = nn.Parameter(torch.randn(r_m, r_i, r_o) * scale)
        self.Gi = nn.Parameter(torch.randn(r_m, r_i, r_o) * scale)
        # Factor matrices (shared between real/imag)
        self.Um = nn.Parameter(torch.randn(n_modes, r_m) * scale)
        self.Ui = nn.Parameter(torch.randn(in_ch,   r_i) * scale)
        self.Uo = nn.Parameter(torch.randn(out_ch,  r_o) * scale)

    def _reconstruct(self):
        """Reconstruct full [n_modes, in_ch, out_ch] weight from Tucker factors."""
        Wr = torch.einsum("abc,ma,ib,oc->mio", self.Gr, self.Um, self.Ui, self.Uo)
        Wi = torch.einsum("abc,ma,ib,oc->mio", self.Gi, self.Um, self.Ui, self.Uo)
        return Wr, Wi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x_ft = torch.fft.rfft(x, dim=1)
        xr   = x_ft[:, :self.n_modes, :].real
        xi   = x_ft[:, :self.n_modes, :].imag

        Wr, Wi = self._reconstruct()
        out_r = (torch.einsum("bmi,mio->bmo", xr, Wr)
               - torch.einsum("bmi,mio->bmo", xi, Wi))
        out_i = (torch.einsum("bmi,mio->bmo", xr, Wi)
               + torch.einsum("bmi,mio->bmo", xi, Wr))

        out_modes = out_r + 1j * out_i
        n_rfft = N // 2 + 1
        if n_rfft > self.n_modes:
            pad    = torch.zeros(B, n_rfft - self.n_modes, self.out_ch, dtype=x_ft.dtype, device=x.device)
            out_ft = torch.cat([out_modes, pad], dim=1)
        else:
            out_ft = out_modes
        return torch.fft.irfft(out_ft, n=N, dim=1)


# ── CP spectral conv (1D) ─────────────────────────────────────────────────────

class CPSpectralConv1d(nn.Module):
    """CP-factorized 1-D Fourier spectral convolution.

    W[m, i, o] ≈ Σ_r A[m,r] * B[i,r] * C[o,r]

    Extremely compact: R*(n_modes + in_ch + out_ch) params.
    """

    def __init__(self, in_ch: int, out_ch: int, n_modes: int, rank: int = 8):
        super().__init__()
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.n_modes = n_modes
        self.rank    = rank
        scale = (in_ch * out_ch) ** -0.5
        # CP factors (real + imag)
        self.Ar = nn.Parameter(torch.randn(n_modes, rank) * scale)
        self.Ai = nn.Parameter(torch.randn(n_modes, rank) * scale)
        self.Br = nn.Parameter(torch.randn(in_ch,   rank) * scale)
        self.Bi = nn.Parameter(torch.randn(in_ch,   rank) * scale)
        self.Cr = nn.Parameter(torch.randn(out_ch,  rank) * scale)
        self.Ci = nn.Parameter(torch.randn(out_ch,  rank) * scale)

    def _reconstruct(self):
        # W[m,i,o] = Σ_r A[m,r]*B[i,r]*C[o,r]  (treating as separable)
        # For complex weights: (Ar+iAi)*(Br+iBi)*(Cr+iCi) — use full complex product
        # Simplified: reconstruct real and imag independently via the real factors
        Wr = torch.einsum("mr,ir,or->mio", self.Ar, self.Br, self.Cr)
        Wi = torch.einsum("mr,ir,or->mio", self.Ai, self.Bi, self.Ci)
        return Wr, Wi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x_ft = torch.fft.rfft(x, dim=1)
        xr   = x_ft[:, :self.n_modes, :].real
        xi   = x_ft[:, :self.n_modes, :].imag

        Wr, Wi = self._reconstruct()
        out_r = (torch.einsum("bmi,mio->bmo", xr, Wr)
               - torch.einsum("bmi,mio->bmo", xi, Wi))
        out_i = (torch.einsum("bmi,mio->bmo", xr, Wi)
               + torch.einsum("bmi,mio->bmo", xi, Wr))

        out_modes = out_r + 1j * out_i
        n_rfft = N // 2 + 1
        if n_rfft > self.n_modes:
            pad    = torch.zeros(B, n_rfft - self.n_modes, self.out_ch, dtype=x_ft.dtype, device=x.device)
            out_ft = torch.cat([out_modes, pad], dim=1)
        else:
            out_ft = out_modes
        return torch.fft.irfft(out_ft, n=N, dim=1)


# ── TFNO block (1D) ──────────────────────────────────────────────────────────

class TFNOBlock1d(nn.Module):
    """1D TFNO layer: Tucker-spectral conv + pointwise linear + GELU."""
    def __init__(self, channels: int, n_modes: int, rank_ratio: float = 0.5):
        super().__init__()
        self.spec = TuckerSpectralConv1d(channels, channels, n_modes, rank_ratio)
        self.w    = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.w(x))


class TFNOBlockResidual1d(nn.Module):
    """Pre-LN residual TFNO block (Pre-LN + Tucker spectral conv)."""
    def __init__(self, channels: int, n_modes: int, rank_ratio: float = 0.5):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.spec = TuckerSpectralConv1d(channels, channels, n_modes, rank_ratio)
        self.w    = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + F.gelu(self.spec(h) + self.w(h))


# ── TFNO1d ────────────────────────────────────────────────────────────────────

class TFNO1d(nn.Module):
    """Tucker-Factorized Fourier Neural Operator (1-D).

    Drop-in for FNO1d with Tucker-factorized spectral weights.
    Hyperparameter guide: same as FNO1d; add rank_ratio (default 0.5).
    Use higher rank_ratio (0.75) for harder problems; lower (0.25) to stress-test
    low-rank bias.
    """

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int,
                 in_ch: int = 2, rank_ratio: float = 0.5):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([TFNOBlock1d(hidden_dim, n_modes, rank_ratio)
                                     for _ in range(n_layers)])
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N  = u0.shape
        grid  = torch.linspace(0.0, 1.0, N, device=u0.device, dtype=u0.dtype).unsqueeze(0).expand(B, -1)
        x     = torch.stack([u0, grid], dim=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(x))
        return self.proj2(x)[:, :, 0]


class RTFNO1d(nn.Module):
    """Residual Tucker-Factorized FNO (Pre-LN + Tucker spectral conv).

    Combines RFNO's pre-norm residual stability with TFNO's low-rank spectral
    weights — best of both for deep stacks on hard benchmarks.
    """

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int,
                 in_ch: int = 2, rank_ratio: float = 0.5):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([TFNOBlockResidual1d(hidden_dim, n_modes, rank_ratio)
                                     for _ in range(n_layers)])
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


# ── CPFNO1d ───────────────────────────────────────────────────────────────────

class CPFNOBlock1d(nn.Module):
    """1D CP-FNO layer."""
    def __init__(self, channels: int, n_modes: int, rank: int = 8):
        super().__init__()
        self.spec = CPSpectralConv1d(channels, channels, n_modes, rank)
        self.w    = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.w(x))


class CPFNO1d(nn.Module):
    """CP-Factorized FNO (1-D). Most compact spectral operator.

    With rank=8 and hidden_dim=128, n_modes=24:
      Full FNO spectral params:  2 * 24 * 128 * 128 = 786,432
      CP-FNO spectral params:    2 * 8 * (24 + 128 + 128) = 4,480  (~175x reduction)

    Useful for probing whether low-rank spectral operators can still learn well.
    """

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int,
                 in_ch: int = 2, rank: int = 8):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([CPFNOBlock1d(hidden_dim, n_modes, rank)
                                     for _ in range(n_layers)])
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N  = u0.shape
        grid  = torch.linspace(0.0, 1.0, N, device=u0.device, dtype=u0.dtype).unsqueeze(0).expand(B, -1)
        x     = torch.stack([u0, grid], dim=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(x))
        return self.proj2(x)[:, :, 0]


# ── Tucker SpectralConv2d ─────────────────────────────────────────────────────

class TuckerSpectralConv2d(nn.Module):
    """Tucker-factorized 2-D Fourier spectral convolution.

    Factorizes both (modes1, modes2, in_ch, out_ch) weight tensors.
    Two sets of weights needed for the rfft2 quadrant symmetry.
    """

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
        self.G1r = nn.Parameter(torch.randn(r1, r2, ri, ro) * scale)
        self.G1i = nn.Parameter(torch.randn(r1, r2, ri, ro) * scale)
        self.G2r = nn.Parameter(torch.randn(r1, r2, ri, ro) * scale)
        self.G2i = nn.Parameter(torch.randn(r1, r2, ri, ro) * scale)
        # Shared factor matrices across quadrants
        self.Um1 = nn.Parameter(torch.randn(n_modes1, r1) * scale)
        self.Um2 = nn.Parameter(torch.randn(n_modes2, r2) * scale)
        self.Ui  = nn.Parameter(torch.randn(in_ch,    ri) * scale)
        self.Uo  = nn.Parameter(torch.randn(out_ch,   ro) * scale)

    def _reconstruct(self, Gr, Gi):
        Wr = torch.einsum("abcd,ma,nb,ic,oc->mnio", Gr, self.Um1, self.Um2, self.Ui, self.Uo)
        Wi = torch.einsum("abcd,ma,nb,ic,oc->mnio", Gi, self.Um1, self.Um2, self.Ui, self.Uo)
        return Wr, Wi

    def _cmul2d(self, xr, xi, Wr, Wi):
        out_r = (torch.einsum("bmki,mkio->bmko", xr, Wr)
               - torch.einsum("bmki,mkio->bmko", xi, Wi))
        out_i = (torch.einsum("bmki,mkio->bmko", xr, Wi)
               + torch.einsum("bmki,mkio->bmko", xi, Wr))
        return out_r, out_i

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N1, N2, _ = x.shape
        x_ft = torch.fft.rfft2(x, dim=(1, 2))
        out_ft = torch.zeros(B, N1, N2 // 2 + 1, self.out_ch, dtype=x_ft.dtype, device=x.device)

        Wr1, Wi1 = self._reconstruct(self.G1r, self.G1i)
        xr1 = x_ft[:, :self.n_modes1, :self.n_modes2, :].real
        xi1 = x_ft[:, :self.n_modes1, :self.n_modes2, :].imag
        or1, oi1 = self._cmul2d(xr1, xi1, Wr1, Wi1)
        out_ft[:, :self.n_modes1, :self.n_modes2, :] = or1 + 1j * oi1

        Wr2, Wi2 = self._reconstruct(self.G2r, self.G2i)
        xr2 = x_ft[:, -self.n_modes1:, :self.n_modes2, :].real
        xi2 = x_ft[:, -self.n_modes1:, :self.n_modes2, :].imag
        or2, oi2 = self._cmul2d(xr2, xi2, Wr2, Wi2)
        out_ft[:, -self.n_modes1:, :self.n_modes2, :] = or2 + 1j * oi2

        return torch.fft.irfft2(out_ft, s=(N1, N2), dim=(1, 2))


class TFNOBlock2d(nn.Module):
    """2D TFNO layer."""
    def __init__(self, channels: int, n_modes1: int, n_modes2: int,
                 rank_ratio: float = 0.5):
        super().__init__()
        self.spec = TuckerSpectralConv2d(channels, channels, n_modes1, n_modes2, rank_ratio)
        self.w    = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.w(x))


class TFNO2d(nn.Module):
    """Tucker-Factorized FNO for 2-D operator learning.

    Drop-in for FNO2d with Tucker-factorized spectral weights.
    Well-suited for darcy_2d (low-rank permeability → pressure mapping)
    and ns_2d (vorticity → stream function).
    """

    def __init__(self, n_modes1: int, n_modes2: int, hidden_dim: int, n_layers: int,
                 in_ch: int = 3, rank_ratio: float = 0.5):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([TFNOBlock2d(hidden_dim, n_modes1, n_modes2, rank_ratio)
                                     for _ in range(n_layers)])
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N1, N2 = u0.shape
        grid1 = torch.linspace(0.0, 1.0, N1, device=u0.device, dtype=u0.dtype).reshape(1, N1, 1).expand(B, -1, N2)
        grid2 = torch.linspace(0.0, 1.0, N2, device=u0.device, dtype=u0.dtype).reshape(1, 1, N2).expand(B, N1, -1)
        x     = torch.stack([u0, grid1, grid2], dim=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(x))
        return self.proj2(x)[:, :, :, 0]
