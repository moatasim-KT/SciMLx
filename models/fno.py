import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1D Spectral Components ────────────────────────────────────────────────────

class SpectralConv1d(nn.Module):
    """1-D Fourier spectral convolution."""

    def __init__(self, in_ch: int, out_ch: int, n_modes: int):
        super().__init__()
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.n_modes = n_modes
        scale = (in_ch * out_ch) ** -0.5
        self.wr = nn.Parameter(torch.randn([n_modes, in_ch, out_ch]) * scale)
        self.wi = nn.Parameter(torch.randn([n_modes, in_ch, out_ch]) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x_ft = torch.fft.rfft(x, dim=1)
        xr   = x_ft[:, :self.n_modes, :].real
        xi   = x_ft[:, :self.n_modes, :].imag

        out_r = (torch.einsum("bmi,mio->bmo", xr, self.wr)
               - torch.einsum("bmi,mio->bmo", xi, self.wi))
        out_i = (torch.einsum("bmi,mio->bmo", xr, self.wi)
               + torch.einsum("bmi,mio->bmo", xi, self.wr))

        out_modes = out_r + 1j * out_i
        n_rfft = N // 2 + 1
        if n_rfft > self.n_modes:
            pad    = torch.zeros([B, n_rfft - self.n_modes, self.out_ch], dtype=torch.complex64, device=x.device)
            out_ft = torch.cat([out_modes, pad], dim=1)
        else:
            out_ft = out_modes
        return torch.fft.irfft(out_ft, n=N, dim=1)


class FNOBlock1d(nn.Module):
    """1D FNO layer: spectral conv + pointwise linear + GELU."""
    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.spec = SpectralConv1d(channels, channels, n_modes)
        self.w    = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.w(x))


class FNO1d(nn.Module):
    """Fourier Neural Operator for 1-D operator learning."""
    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, in_ch: int = 2):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([FNOBlock1d(hidden_dim, n_modes) for _ in range(n_layers)])
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N  = u0.shape
        grid  = torch.linspace(0.0, 1.0, N, device=u0.device).unsqueeze(0).expand(B, -1)
        x     = torch.stack([u0, grid], dim=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(x))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + F.gelu(self.spec(h) + self.w(h))


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
        self.blocks = nn.ModuleList([FNOBlockResidual1d(hidden_dim, n_modes) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(hidden_dim)
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        if u0.ndim == 3:
            B, N, _ = u0.shape
            u_scalar = u0[:, :, 0]  # use first channel for grid stacking
        else:
            B, N  = u0.shape
            u_scalar = u0
        grid  = torch.linspace(0.0, 1.0, N, device=u0.device).unsqueeze(0).expand(B, -1)
        x     = torch.stack([u_scalar, grid], dim=-1)
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(self.norm(x)))
        return self.proj2(x)[:, :, 0]


class FNO1dMC(nn.Module):
    """Multi-channel Fourier Neural Operator for 1-D operator learning.

    Handles inputs of shape [B, N, C_in] -> [B, N, C_out].
    Extends FNO1d to multi-component PDEs (e.g., Euler: rho,u,p -> rho,u,p).
    An appended spatial coordinate is used as an extra input channel.
    """

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int,
                 in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.in_ch  = in_channels
        self.out_ch = out_channels
        self.lift   = nn.Linear(in_channels + 1, hidden_dim)   # +1 for grid
        self.blocks = nn.ModuleList([FNOBlock1d(hidden_dim, n_modes) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(hidden_dim)
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, out_channels)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N, _ = u0.shape
        grid    = torch.linspace(0.0, 1.0, N, device=u0.device).view(1, N, 1).expand(B, -1, -1)
        x = torch.cat([u0, grid], dim=-1)  # [B, N, C+1]
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.proj1(self.norm(x)))
        return self.proj2(x)                     # [B, N, C_out]


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
        self.wr1 = nn.Parameter(torch.randn([n_modes1, n_modes2, in_ch, out_ch]) * scale)
        self.wi1 = nn.Parameter(torch.randn([n_modes1, n_modes2, in_ch, out_ch]) * scale)
        self.wr2 = nn.Parameter(torch.randn([n_modes1, n_modes2, in_ch, out_ch]) * scale)
        self.wi2 = nn.Parameter(torch.randn([n_modes1, n_modes2, in_ch, out_ch]) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, N1, N2, C_in]
        B, N1, N2, _ = x.shape
        x_ft = torch.fft.rfft2(x, dim=(1, 2))  # [B, N1, N2//2+1, C] complex

        # Handle the two symmetric modes in the first dimension
        out_ft = torch.zeros([B, N1, N2 // 2 + 1, self.out_ch], dtype=torch.complex64, device=x.device)

        # Mode 1: [0:n_modes1, 0:n_modes2]
        xr1 = x_ft[:, :self.n_modes1, :self.n_modes2, :].real
        xi1 = x_ft[:, :self.n_modes1, :self.n_modes2, :].imag
        out_r1 = (torch.einsum("bmki,mkio->bmko", xr1, self.wr1)
                - torch.einsum("bmki,mkio->bmko", xi1, self.wi1))
        out_i1 = (torch.einsum("bmki,mkio->bmko", xr1, self.wi1)
                + torch.einsum("bmki,mkio->bmko", xi1, self.wr1))
        out_ft[:, :self.n_modes1, :self.n_modes2, :] = out_r1 + 1j * out_i1

        # Mode 2: [-n_modes1:, 0:n_modes2]
        xr2 = x_ft[:, -self.n_modes1:, :self.n_modes2, :].real
        xi2 = x_ft[:, -self.n_modes1:, :self.n_modes2, :].imag
        out_r2 = (torch.einsum("bmki,mkio->bmko", xr2, self.wr2)
                - torch.einsum("bmki,mkio->bmko", xi2, self.wi2))
        out_i2 = (torch.einsum("bmki,mkio->bmko", xr2, self.wi2)
                + torch.einsum("bmki,mkio->bmko", xi2, self.wr2))
        out_ft[:, -self.n_modes1:, :self.n_modes2, :] = out_r2 + 1j * out_i2

        return torch.fft.irfft2(out_ft, s=(N1, N2), dim=(1, 2))


class FNOBlock2d(nn.Module):
    """2D FNO layer: spectral conv + pointwise linear + GELU."""
    def __init__(self, channels: int, n_modes1: int, n_modes2: int):
        super().__init__()
        self.spec = SpectralConv2d(channels, channels, n_modes1, n_modes2)
        self.w    = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.w(x))


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2-D operator learning."""
    def __init__(self, n_modes1: int, n_modes2: int, hidden_dim: int, n_layers: int, in_channels: int = 1):
        super().__init__()
        # Internal lifting dimension: data channels + 2 spatial grids
        self.lift   = nn.Linear(in_channels + 2, hidden_dim)
        self.blocks = nn.ModuleList([FNOBlock2d(hidden_dim, n_modes1, n_modes2) for _ in range(n_layers)])
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, N1, N2] or [B, N1, N2, C]
        if x.ndim == 3:
            B, N1, N2 = x.shape
            x = x[..., None]
        else:
            B, N1, N2, _ = x.shape

        grid1 = torch.linspace(0.0, 1.0, N1, device=x.device).view(1, N1, 1, 1).expand(B, -1, N2, -1)
        grid2 = torch.linspace(0.0, 1.0, N2, device=x.device).view(1, 1, N2, 1).expand(B, N1, -1, -1)
        x     = torch.cat([x, grid1, grid2], dim=-1)  # [B, N1, N2, C+2]

        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(x))
        out   = self.proj2(x)

        if out.shape[-1] == 1:
            return out[:, :, :, 0]
        return out


class FNOBlockResidual2d(nn.Module):
    """Pre-LN residual FNO block for 2-D."""
    def __init__(self, channels: int, n_modes1: int, n_modes2: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.spec = SpectralConv2d(channels, channels, n_modes1, n_modes2)
        self.w    = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + F.gelu(self.spec(h) + self.w(h))


class RFNO2d(nn.Module):
    """Residual Fourier Neural Operator for 2-D operator learning."""
    def __init__(self, n_modes1: int, n_modes2: int, hidden_dim: int, n_layers: int, in_channels: int = 1):
        super().__init__()
        self.lift   = nn.Linear(in_channels + 2, hidden_dim)
        self.blocks = nn.ModuleList([FNOBlockResidual2d(hidden_dim, n_modes1, n_modes2) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(hidden_dim)
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, N1, N2] or [B, N1, N2, C]
        if x.ndim == 3:
            B, N1, N2 = x.shape
            x = x[..., None]
        else:
            B, N1, N2, _ = x.shape

        grid1 = torch.linspace(0.0, 1.0, N1, device=x.device).view(1, N1, 1, 1).expand(B, -1, N2, -1)
        grid2 = torch.linspace(0.0, 1.0, N2, device=x.device).view(1, 1, N2, 1).expand(B, N1, -1, -1)
        x     = torch.cat([x, grid1, grid2], dim=-1)  # [B, N1, N2, C+2]

        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(self.norm(x)))
        out   = self.proj2(x)

        if out.shape[-1] == 1:
            return out[:, :, :, 0]
        return out


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

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N = u0.shape
        grid = torch.linspace(0.0, 1.0, N, device=u0.device).unsqueeze(0).expand(B, -1)
        x    = torch.stack([u0, grid], dim=-1)
        x    = self.lift(x)                         # [B, N,    h]

        # Encode
        s0   = self.enc0(x)                         # [B, N,    h]  skip0
        x    = self.down0(s0)[:, ::2, :]            # [B, N//2, 2h]
        s1   = self.enc1(x)                         # [B, N//2, 2h]  skip1
        x    = self.down1(s1)[:, ::2, :]            # [B, N//4, 4h]

        # Bottleneck
        x    = self.bot(x)                          # [B, N//4, 4h]

        # Decode
        x    = x.repeat_interleave(2, dim=1)       # [B, N//2, 4h]
        x    = torch.cat([x, s1], dim=-1)           # [B, N//2, 6h]
        x    = self.up1(x)                          # [B, N//2, 2h]
        x    = self.dec1(x)                         # [B, N//2, 2h]
        x    = x.repeat_interleave(2, dim=1)       # [B, N,    2h]
        x    = torch.cat([x, s0], dim=-1)           # [B, N,    3h]
        x    = self.up0(x)                          # [B, N,    h]
        x    = self.dec0(x)                         # [B, N,    h]

        x    = F.gelu(self.proj1(x))
        return self.proj2(x)[:, :, 0]

class UNO2d(nn.Module):
    """U-shaped Neural Operator for 2D problems.
    
    Architecture (hidden_dim=h, N=64 default):
        lift → enc0(64,h) → enc1(32,2h) → bottleneck(16,4h)
             ↓ skip0          ↓ skip1
        dec0(64,h) ← dec1(32,2h) ← upsample
    """
    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int = 2, in_channels: int = 1):
        super().__init__()
        h = hidden_dim
        m = n_modes
        # Encoder
        self.lift  = nn.Linear(in_channels + 2, h)
        self.enc0  = nn.Sequential(*[FNOBlock2d(h, m, m) for _ in range(n_layers)])
        self.down0 = nn.Linear(h, 2 * h)
        # For lower levels, we halve modes to keep spectral compression
        self.enc1  = nn.Sequential(*[FNOBlock2d(2 * h, max(m // 2, 4), max(m // 2, 4))
                                      for _ in range(n_layers)])
        self.down1 = nn.Linear(2 * h, 4 * h)

        # Bottleneck
        self.bot   = nn.Sequential(*[FNOBlock2d(4 * h, max(m // 4, 2), max(m // 4, 2))
                                      for _ in range(n_layers)])

        # Decoder
        self.up1   = nn.Linear(4 * h + 2 * h, 2 * h)
        self.dec1  = nn.Sequential(*[FNOBlock2d(2 * h, max(m // 2, 4), max(m // 2, 4))
                                      for _ in range(n_layers)])
        self.up0   = nn.Linear(2 * h + h, h)
        self.dec0  = nn.Sequential(*[FNOBlock2d(h, m, m) for _ in range(n_layers)])

        self.proj1 = nn.Linear(h, h // 2)
        self.proj2 = nn.Linear(h // 2, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            B, N1, N2 = x.shape
            x = x[..., None]
        else:
            B, N1, N2, _ = x.shape

        grid1 = torch.linspace(0.0, 1.0, N1, device=x.device).view(1, N1, 1, 1).expand(B, -1, N2, -1)
        grid2 = torch.linspace(0.0, 1.0, N2, device=x.device).view(1, 1, N2, 1).expand(B, N1, -1, -1)
        x     = torch.cat([x, grid1, grid2], dim=-1)

        x = self.lift(x)                            # [B, N, N, h]

        # Encode
        s0 = self.enc0(x)                           # [B, N, N, h]
        x  = self.down0(s0)[:, ::2, ::2, :]         # [B, N/2, N/2, 2h]
        s1 = self.enc1(x)                           # [B, N/2, N/2, 2h]
        x  = self.down1(s1)[:, ::2, ::2, :]         # [B, N/4, N/4, 4h]

        # Bottleneck
        x  = self.bot(x)                            # [B, N/4, N/4, 4h]

        # Decode
        x  = x.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2) # [B, N/2, N/2, 4h]
        x  = torch.cat([x, s1], dim=-1)             # [B, N/2, N/2, 6h]
        x  = self.up1(x)                            # [B, N/2, N/2, 2h]
        x  = self.dec1(x)                           # [B, N/2, N/2, 2h]

        x  = x.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2) # [B, N, N, 2h]
        x  = torch.cat([x, s0], dim=-1)             # [B, N, N, 3h]
        x  = self.up0(x)                            # [B, N, N, h]
        x  = self.dec0(x)                           # [B, N, N, h]

        x = F.gelu(self.proj1(x))
        out = self.proj2(x)
        if out.shape[-1] == 1:
            return out[:, :, :, 0]
        return out
