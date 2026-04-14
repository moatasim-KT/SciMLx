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

import mlx.core as mx
import mlx.nn as nn

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
        self.Gr = mx.random.normal([r_m, r_i, r_o]) * scale
        self.Gi = mx.random.normal([r_m, r_i, r_o]) * scale
        # Factor matrices (shared between real/imag)
        self.Um = mx.random.normal([n_modes, r_m]) * scale
        self.Ui = mx.random.normal([in_ch,   r_i]) * scale
        self.Uo = mx.random.normal([out_ch,  r_o]) * scale

    def _reconstruct(self):
        """Reconstruct full [n_modes, in_ch, out_ch] weight from Tucker factors."""
        Wr = mx.einsum("abc,ma,ib,oc->mio", self.Gr, self.Um, self.Ui, self.Uo)
        Wi = mx.einsum("abc,ma,ib,oc->mio", self.Gi, self.Um, self.Ui, self.Uo)
        return Wr, Wi

    def __call__(self, x: mx.array) -> mx.array:
        B, N, _ = x.shape
        x_ft = mx.fft.rfft(x, axis=1)
        xr   = x_ft[:, :self.n_modes, :].real
        xi   = x_ft[:, :self.n_modes, :].imag

        Wr, Wi = self._reconstruct()
        out_r = (mx.einsum("bmi,mio->bmo", xr, Wr)
               - mx.einsum("bmi,mio->bmo", xi, Wi))
        out_i = (mx.einsum("bmi,mio->bmo", xr, Wi)
               + mx.einsum("bmi,mio->bmo", xi, Wr))

        out_modes = out_r + 1j * out_i
        n_rfft = N // 2 + 1
        if n_rfft > self.n_modes:
            pad    = mx.zeros([B, n_rfft - self.n_modes, self.out_ch], dtype=mx.complex64)
            out_ft = mx.concatenate([out_modes, pad], axis=1)
        else:
            out_ft = out_modes
        return mx.fft.irfft(out_ft, n=N, axis=1)


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
        self.Ar = mx.random.normal([n_modes, rank]) * scale
        self.Ai = mx.random.normal([n_modes, rank]) * scale
        self.Br = mx.random.normal([in_ch,   rank]) * scale
        self.Bi = mx.random.normal([in_ch,   rank]) * scale
        self.Cr = mx.random.normal([out_ch,  rank]) * scale
        self.Ci = mx.random.normal([out_ch,  rank]) * scale

    def _reconstruct(self):
        # W[m,i,o] = Σ_r A[m,r]*B[i,r]*C[o,r]  (treating as separable)
        # For complex weights: (Ar+iAi)*(Br+iBi)*(Cr+iCi) — use full complex product
        # Simplified: reconstruct real and imag independently via the real factors
        Wr = mx.einsum("mr,ir,or->mio", self.Ar, self.Br, self.Cr)
        Wi = mx.einsum("mr,ir,or->mio", self.Ai, self.Bi, self.Ci)
        return Wr, Wi

    def __call__(self, x: mx.array) -> mx.array:
        B, N, _ = x.shape
        x_ft = mx.fft.rfft(x, axis=1)
        xr   = x_ft[:, :self.n_modes, :].real
        xi   = x_ft[:, :self.n_modes, :].imag

        Wr, Wi = self._reconstruct()
        out_r = (mx.einsum("bmi,mio->bmo", xr, Wr)
               - mx.einsum("bmi,mio->bmo", xi, Wi))
        out_i = (mx.einsum("bmi,mio->bmo", xr, Wi)
               + mx.einsum("bmi,mio->bmo", xi, Wr))

        out_modes = out_r + 1j * out_i
        n_rfft = N // 2 + 1
        if n_rfft > self.n_modes:
            pad    = mx.zeros([B, n_rfft - self.n_modes, self.out_ch], dtype=mx.complex64)
            out_ft = mx.concatenate([out_modes, pad], axis=1)
        else:
            out_ft = out_modes
        return mx.fft.irfft(out_ft, n=N, axis=1)


# ── TFNO block (1D) ──────────────────────────────────────────────────────────

class TFNOBlock1d(nn.Module):
    """1D TFNO layer: Tucker-spectral conv + pointwise linear + GELU."""
    def __init__(self, channels: int, n_modes: int, rank_ratio: float = 0.5):
        super().__init__()
        self.spec = TuckerSpectralConv1d(channels, channels, n_modes, rank_ratio)
        self.w    = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.spec(x) + self.w(x))


class TFNOBlockResidual1d(nn.Module):
    """Pre-LN residual TFNO block (Pre-LN + Tucker spectral conv)."""
    def __init__(self, channels: int, n_modes: int, rank_ratio: float = 0.5):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.spec = TuckerSpectralConv1d(channels, channels, n_modes, rank_ratio)
        self.w    = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)
        return x + nn.gelu(self.spec(h) + self.w(h))


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
        self.blocks = [TFNOBlock1d(hidden_dim, n_modes, rank_ratio)
                       for _ in range(n_layers)]
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


class RTFNO1d(nn.Module):
    """Residual Tucker-Factorized FNO (Pre-LN + Tucker spectral conv).

    Combines RFNO's pre-norm residual stability with TFNO's low-rank spectral
    weights — best of both for deep stacks on hard benchmarks.
    """

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int,
                 in_ch: int = 2, rank_ratio: float = 0.5):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = [TFNOBlockResidual1d(hidden_dim, n_modes, rank_ratio)
                       for _ in range(n_layers)]
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


# ── CPFNO1d ───────────────────────────────────────────────────────────────────

class CPFNOBlock1d(nn.Module):
    """1D CP-FNO layer."""
    def __init__(self, channels: int, n_modes: int, rank: int = 8):
        super().__init__()
        self.spec = CPSpectralConv1d(channels, channels, n_modes, rank)
        self.w    = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.spec(x) + self.w(x))


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
        self.blocks = [CPFNOBlock1d(hidden_dim, n_modes, rank)
                       for _ in range(n_layers)]
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
        for k in ("1", "2"):
            for part in ("r", "i"):
                setattr(self, f"G{k}{part}", mx.random.normal([r1, r2, ri, ro]) * scale)
        # Shared factor matrices across quadrants
        self.Um1 = mx.random.normal([n_modes1, r1]) * scale
        self.Um2 = mx.random.normal([n_modes2, r2]) * scale
        self.Ui  = mx.random.normal([in_ch,    ri]) * scale
        self.Uo  = mx.random.normal([out_ch,   ro]) * scale

    def _reconstruct(self, Gr, Gi):
        Wr = mx.einsum("abcd,ma,nb,ic,oc->mnio", Gr, self.Um1, self.Um2, self.Ui, self.Uo)
        Wi = mx.einsum("abcd,ma,nb,ic,oc->mnio", Gi, self.Um1, self.Um2, self.Ui, self.Uo)
        return Wr, Wi

    def _cmul2d(self, xr, xi, Wr, Wi):
        out_r = (mx.einsum("bmki,mkio->bmko", xr, Wr)
               - mx.einsum("bmki,mkio->bmko", xi, Wi))
        out_i = (mx.einsum("bmki,mkio->bmko", xr, Wi)
               + mx.einsum("bmki,mkio->bmko", xi, Wr))
        return out_r, out_i

    def __call__(self, x: mx.array) -> mx.array:
        B, N1, N2, _ = x.shape
        x_ft = mx.fft.rfft2(x, axes=(1, 2))
        out_ft = mx.zeros([B, N1, N2 // 2 + 1, self.out_ch], dtype=mx.complex64)

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

        return mx.fft.irfft2(out_ft, s=(N1, N2), axes=(1, 2))


class TFNOBlock2d(nn.Module):
    """2D TFNO layer."""
    def __init__(self, channels: int, n_modes1: int, n_modes2: int,
                 rank_ratio: float = 0.5):
        super().__init__()
        self.spec = TuckerSpectralConv2d(channels, channels, n_modes1, n_modes2, rank_ratio)
        self.w    = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.spec(x) + self.w(x))


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
        self.blocks = [TFNOBlock2d(hidden_dim, n_modes1, n_modes2, rank_ratio)
                       for _ in range(n_layers)]
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, u0: mx.array) -> mx.array:
        B, N1, N2 = u0.shape
        grid1 = mx.broadcast_to(mx.linspace(0.0, 1.0, N1).reshape(1, N1, 1), (B, N1, N2))
        grid2 = mx.broadcast_to(mx.linspace(0.0, 1.0, N2).reshape(1, 1, N2), (B, N1, N2))
        x     = mx.stack([u0, grid1, grid2], axis=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = nn.gelu(self.proj1(x))
        return self.proj2(x)[:, :, :, 0]
