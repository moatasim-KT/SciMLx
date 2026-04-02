import mlx.core as mx
import mlx.nn as nn


# ── 1D Spectral Components ────────────────────────────────────────────────────

class SpectralConv1d(nn.Module):
    """1-D Fourier spectral convolution."""

    def __init__(self, in_ch: int, out_ch: int, n_modes: int):
        super().__init__()
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.n_modes = n_modes
        scale = (in_ch * out_ch) ** -0.5
        self.wr = mx.random.normal([n_modes, in_ch, out_ch]) * scale
        self.wi = mx.random.normal([n_modes, in_ch, out_ch]) * scale

    def __call__(self, x: mx.array) -> mx.array:
        B, N, _ = x.shape
        x_ft = mx.fft.rfft(x, axis=1)
        xr   = x_ft[:, :self.n_modes, :].real
        xi   = x_ft[:, :self.n_modes, :].imag

        out_r = (mx.einsum("bmi,mio->bmo", xr, self.wr)
               - mx.einsum("bmi,mio->bmo", xi, self.wi))
        out_i = (mx.einsum("bmi,mio->bmo", xr, self.wi)
               + mx.einsum("bmi,mio->bmo", xi, self.wr))

        out_modes = out_r + 1j * out_i
        n_rfft = N // 2 + 1
        if n_rfft > self.n_modes:
            pad    = mx.zeros([B, n_rfft - self.n_modes, self.out_ch], dtype=mx.complex64)
            out_ft = mx.concatenate([out_modes, pad], axis=1)
        else:
            out_ft = out_modes
        return mx.fft.irfft(out_ft, n=N, axis=1)


class FNOBlock1d(nn.Module):
    """1D FNO layer: spectral conv + pointwise linear + GELU."""
    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.spec = SpectralConv1d(channels, channels, n_modes)
        self.w    = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.spec(x) + self.w(x))


class FNO1d(nn.Module):
    """Fourier Neural Operator for 1-D operator learning."""
    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, in_ch: int = 2):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = [FNOBlock1d(hidden_dim, n_modes) for _ in range(n_layers)]
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


# ── Residual FNO (Pre-LN) ─────────────────────────────────────────────────────

class FNOBlockResidual1d(nn.Module):
    """Pre-LN residual FNO block: x = x + GELU(spec(LN(x)) + w(LN(x))).

    The residual connection + pre-norm pattern (from Pre-LN Transformers) gives
    better gradient flow through deep stacks — same param count as FNOBlock1d.
    """

    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.spec = SpectralConv1d(channels, channels, n_modes)
        self.w    = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)
        return x + nn.gelu(self.spec(h) + self.w(h))


class RFNO1d(nn.Module):
    """Residual Fourier Neural Operator for 1-D operator learning.

    Drop-in replacement for FNO1d that uses Pre-LN residual blocks.
    Intended to unlock deeper stacks (l ≥ 10) without gradient degradation.

    Hyperparameter guide: same as FNO1d.  Start with n_modes=24, hidden=128,
    n_layers=10 or 12 — the residual connections should handle the extra depth.
    """

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, in_ch: int = 2):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = [FNOBlockResidual1d(hidden_dim, n_modes) for _ in range(n_layers)]
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


class SpectralConv2d(nn.Module):
    """2-D Fourier spectral convolution."""

    def __init__(self, in_ch: int, out_ch: int, n_modes1: int, n_modes2: int):
        super().__init__()
        self.in_ch    = in_ch
        self.out_ch   = out_ch
        self.n_modes1 = n_modes1
        self.n_modes2 = n_modes2
        scale = (in_ch * out_ch) ** -0.5
        # We need two sets of weights because rfft2 is asymmetric
        self.wr1 = mx.random.normal([n_modes1, n_modes2, in_ch, out_ch]) * scale
        self.wi1 = mx.random.normal([n_modes1, n_modes2, in_ch, out_ch]) * scale
        self.wr2 = mx.random.normal([n_modes1, n_modes2, in_ch, out_ch]) * scale
        self.wi2 = mx.random.normal([n_modes1, n_modes2, in_ch, out_ch]) * scale

    def __call__(self, x: mx.array) -> mx.array:
        # x : [B, N1, N2, C_in]
        B, N1, N2, C = x.shape
        x_ft = mx.fft.rfft2(x, axes=(1, 2))  # [B, N1, N2//2+1, C] complex

        # Handle the two symmetric modes in the first dimension
        out_ft = mx.zeros([B, N1, N2 // 2 + 1, self.out_ch], dtype=mx.complex64)

        # Mode 1: [0:n_modes1, 0:n_modes2]
        xr1 = x_ft[:, :self.n_modes1, :self.n_modes2, :].real
        xi1 = x_ft[:, :self.n_modes1, :self.n_modes2, :].imag
        out_r1 = (mx.einsum("bmki,mkio->bmko", xr1, self.wr1)
                - mx.einsum("bmki,mkio->bmko", xi1, self.wi1))
        out_i1 = (mx.einsum("bmki,mkio->bmko", xr1, self.wi1)
                + mx.einsum("bmki,mkio->bmko", xi1, self.wr1))
        out_ft[:, :self.n_modes1, :self.n_modes2, :] = out_r1 + 1j * out_i1

        # Mode 2: [-n_modes1:, 0:n_modes2]
        xr2 = x_ft[:, -self.n_modes1:, :self.n_modes2, :].real
        xi2 = x_ft[:, -self.n_modes1:, :self.n_modes2, :].imag
        out_r2 = (mx.einsum("bmki,mkio->bmko", xr2, self.wr2)
                - mx.einsum("bmki,mkio->bmko", xi2, self.wi2))
        out_i2 = (mx.einsum("bmki,mkio->bmko", xr2, self.wi2)
                + mx.einsum("bmki,mkio->bmko", xi2, self.wr2))
        out_ft[:, -self.n_modes1:, :self.n_modes2, :] = out_r2 + 1j * out_i2

        return mx.fft.irfft2(out_ft, s=(N1, N2), axes=(1, 2))


class FNOBlock2d(nn.Module):
    """2D FNO layer: spectral conv + pointwise linear + GELU."""
    def __init__(self, channels: int, n_modes1: int, n_modes2: int):
        super().__init__()
        self.spec = SpectralConv2d(channels, channels, n_modes1, n_modes2)
        self.w    = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.spec(x) + self.w(x))


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2-D operator learning."""
    def __init__(self, n_modes1: int, n_modes2: int, hidden_dim: int, n_layers: int, in_ch: int = 3):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = [FNOBlock2d(hidden_dim, n_modes1, n_modes2) for _ in range(n_layers)]
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, u0: mx.array) -> mx.array:
        # u0 : [B, N1, N2]
        B, N1, N2 = u0.shape
        grid1 = mx.broadcast_to(mx.linspace(0.0, 1.0, N1).reshape(1, N1, 1), (B, N1, N2))
        grid2 = mx.broadcast_to(mx.linspace(0.0, 1.0, N2).reshape(1, 1, N2), (B, N1, N2))
        x     = mx.stack([u0, grid1, grid2], axis=-1)  # [B, N1, N2, 3]
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = nn.gelu(self.proj1(x))
        return self.proj2(x)[:, :, :, 0]


# ── U-shaped Neural Operator (UNO) ────────────────────────────────────────────

class UNO1d(nn.Module):
    """U-shaped Neural Operator for 1-D problems.

    Encoder-decoder with FNO layers at each scale, skip connections between
    symmetric levels, and spectral subsampling (slice every other point) for
    downsampling.  Captures both global (low-freq) and local (high-freq)
    features simultaneously.

    Reference: Rahman et al. (2022) "U-NO: U-shaped Neural Operators"
    (arXiv:2204.11127)

    Architecture (hidden_dim=h, N=64 default):
        lift → enc0(N,h) → enc1(N/2,2h) → bottleneck(N/4,4h)
             ↓ skip0          ↓ skip1
        dec0(N,h) ← dec1(N/2,2h) ← upsample
    """

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int = 2,
                 in_ch: int = 2):
        super().__init__()
        h = hidden_dim
        m = n_modes
        # Encoder
        self.lift  = nn.Linear(in_ch, h)
        self.enc0  = nn.Sequential(*[FNOBlock1d(h, m) for _ in range(n_layers)])
        self.down0 = nn.Linear(h, 2 * h)
        self.enc1  = nn.Sequential(*[FNOBlock1d(2 * h, max(m // 2, 1))
                                      for _ in range(n_layers)])
        self.down1 = nn.Linear(2 * h, 4 * h)
        # Bottleneck
        self.bot   = nn.Sequential(*[FNOBlock1d(4 * h, max(m // 4, 1))
                                      for _ in range(n_layers)])
        # Decoder
        self.up1   = nn.Linear(4 * h + 2 * h, 2 * h)
        self.dec1  = nn.Sequential(*[FNOBlock1d(2 * h, max(m // 2, 1))
                                      for _ in range(n_layers)])
        self.up0   = nn.Linear(2 * h + h, h)
        self.dec0  = nn.Sequential(*[FNOBlock1d(h, m) for _ in range(n_layers)])
        # Output projection
        self.proj1 = nn.Linear(h, h // 2)
        self.proj2 = nn.Linear(h // 2, 1)

    def __call__(self, u0: mx.array) -> mx.array:
        B, N = u0.shape
        grid = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N), (B, N))
        x    = mx.stack([u0, grid], axis=-1)
        x    = self.lift(x)                         # [B, N,    h]

        # Encode
        s0   = self.enc0(x)                         # [B, N,    h]  skip0
        x    = self.down0(s0)[:, ::2, :]            # [B, N//2, 2h]
        s1   = self.enc1(x)                         # [B, N//2, 2h]  skip1
        x    = self.down1(s1)[:, ::2, :]            # [B, N//4, 4h]

        # Bottleneck
        x    = self.bot(x)                          # [B, N//4, 4h]

        # Decode
        x    = mx.repeat(x, 2, axis=1)             # [B, N//2, 4h]
        x    = mx.concatenate([x, s1], axis=-1)     # [B, N//2, 6h]
        x    = self.up1(x)                          # [B, N//2, 2h]
        x    = self.dec1(x)                         # [B, N//2, 2h]
        x    = mx.repeat(x, 2, axis=1)             # [B, N,    2h]
        x    = mx.concatenate([x, s0], axis=-1)     # [B, N,    3h]
        x    = self.up0(x)                          # [B, N,    h]
        x    = self.dec0(x)                         # [B, N,    h]

        x    = nn.gelu(self.proj1(x))
        return self.proj2(x)[:, :, 0]
