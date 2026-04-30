"""Fourier Neural Operator (FNO) implementations (PyTorch/CUDA)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── 1D Spectral Components ────────────────────────────────────────────────────

class SpectralConv1d(nn.Module):
    """1-D Fourier spectral convolution."""

    def __init__(self, in_ch: int, out_ch: int, n_modes: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_modes = n_modes
        scale = (in_ch * out_ch) ** -0.5
        self.weights = nn.Parameter(
            scale * torch.randn(n_modes, in_ch, out_ch, dtype=torch.complex64)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x_ft = torch.fft.rfft(x, dim=1)
        
        out_ft = torch.zeros(B, N // 2 + 1, self.out_ch, device=x.device, dtype=torch.complex64)
        out_ft[:, :self.n_modes, :] = torch.einsum("bmi,mio->bmo", x_ft[:, :self.n_modes, :], self.weights)
        
        return torch.fft.irfft(out_ft, n=N, dim=1)


class FNOBlock1d(nn.Module):
    """1D FNO layer: spectral conv + pointwise linear + GELU."""
    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.spec = SpectralConv1d(channels, channels, n_modes)
        self.w = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.w(x))


class FNO1d(nn.Module):
    """Fourier Neural Operator for 1-D operator learning."""
    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, in_ch: int = 2):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([FNOBlock1d(hidden_dim, n_modes) for _ in range(n_layers)])
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        if not isinstance(u0, torch.Tensor):
            u0 = torch.tensor(u0, dtype=torch.float32)
        
        # Ensure u0 is on the same device as the model weights
        device = next(self.parameters()).device
        u0 = u0.to(device)

        if u0.ndim == 1:
            u0 = u0.unsqueeze(0)
        B, N = u0.shape
        grid = torch.linspace(0.0, 1.0, N, device=device).view(1, N).expand(B, N)
        x = torch.stack([u0, grid], dim=-1)
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.proj1(x))
        return self.proj2(x).squeeze(-1)


# ── 2D Spectral Components ────────────────────────────────────────────────────

class SpectralConv2d(nn.Module):
    """2-D Fourier spectral convolution."""

    def __init__(self, in_ch: int, out_ch: int, n_modes1: int, n_modes2: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_modes1 = n_modes1
        self.n_modes2 = n_modes2
        scale = (in_ch * out_ch) ** -0.5
        self.weights1 = nn.Parameter(scale * torch.randn(n_modes1, n_modes2, in_ch, out_ch, dtype=torch.complex64))
        self.weights2 = nn.Parameter(scale * torch.randn(n_modes1, n_modes2, in_ch, out_ch, dtype=torch.complex64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N1, N2, C = x.shape
        x_ft = torch.fft.rfft2(x, dim=(1, 2))
        out_ft = torch.zeros(B, N1, N2 // 2 + 1, self.out_ch, device=x.device, dtype=torch.complex64)
        
        # Multiply relevant Fourier modes
        out_ft[:, :self.n_modes1, :self.n_modes2, :] = \
            torch.einsum("bnmi,nmio->bnmo", x_ft[:, :self.n_modes1, :self.n_modes2, :], self.weights1)
        out_ft[:, -self.n_modes1:, :self.n_modes2, :] = \
            torch.einsum("bnmi,nmio->bnmo", x_ft[:, -self.n_modes1:, :self.n_modes2, :], self.weights2)
        
        return torch.fft.irfft2(out_ft, s=(N1, N2), dim=(1, 2))


class FNOBlock2d(nn.Module):
    """2D FNO layer: spectral conv + pointwise linear + GELU."""
    def __init__(self, channels: int, n_modes1: int, n_modes2: int):
        super().__init__()
        self.spec = SpectralConv2d(channels, channels, n_modes1, n_modes2)
        self.w = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.w(x))


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2-D operator learning."""
    def __init__(self, n_modes1: int, n_modes2: int, hidden_dim: int, n_layers: int, in_ch: int = 3):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([FNOBlock2d(hidden_dim, n_modes1, n_modes2) for _ in range(n_layers)])
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        if not isinstance(u0, torch.Tensor):
            u0 = torch.tensor(u0, dtype=torch.float32)
        
        # Ensure u0 is on the same device as the model weights
        device = next(self.parameters()).device
        u0 = u0.to(device)

        B, N1, N2 = u0.shape
        grid1 = torch.linspace(0.0, 1.0, N1, device=device).view(1, N1, 1).expand(B, N1, N2)
        grid2 = torch.linspace(0.0, 1.0, N2, device=device).view(1, 1, N2).expand(B, N1, N2)
        x = torch.stack([u0, grid1, grid2], dim=-1)
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.proj1(x))
        return self.proj2(x).squeeze(-1)


# ── Residual FNO (Pre-LN) ─────────────────────────────────────────────────────

class FNOBlockResidual1d(nn.Module):
    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.spec = SpectralConv1d(channels, channels, n_modes)
        self.w = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + F.gelu(self.spec(h) + self.w(h))


class RFNO1d(nn.Module):
    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, in_ch: int = 2):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([FNOBlockResidual1d(hidden_dim, n_modes) for _ in range(n_layers)])
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        if not isinstance(u0, torch.Tensor):
            u0 = torch.tensor(u0, dtype=torch.float32)
        
        # Ensure u0 is on the same device as the model weights
        device = next(self.parameters()).device
        u0 = u0.to(device)

        if u0.ndim == 1:
            u0 = u0.unsqueeze(0)
        B, N = u0.shape
        grid = torch.linspace(0.0, 1.0, N, device=device).view(1, N).expand(B, N)
        x = torch.stack([u0, grid], dim=-1)
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.proj1(x))
        return self.proj2(x).squeeze(-1)


class FNOBlockResidual2d(nn.Module):
    def __init__(self, channels: int, n_modes1: int, n_modes2: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.spec = SpectralConv2d(channels, channels, n_modes1, n_modes2)
        self.w = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + F.gelu(self.spec(h) + self.w(h))


class RFNO2d(nn.Module):
    def __init__(self, n_modes1: int, n_modes2: int, hidden_dim: int, n_layers: int, in_ch: int = 3):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([FNOBlockResidual2d(hidden_dim, n_modes1, n_modes2) for _ in range(n_layers)])
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        if not isinstance(u0, torch.Tensor):
            u0 = torch.tensor(u0, dtype=torch.float32)
        
        # Ensure u0 is on the same device as the model weights
        device = next(self.parameters()).device
        u0 = u0.to(device)

        B, N1, N2 = u0.shape
        grid1 = torch.linspace(0.0, 1.0, N1, device=device).view(1, N1, 1).expand(B, N1, N2)
        grid2 = torch.linspace(0.0, 1.0, N2, device=device).view(1, 1, N2).expand(B, N1, N2)
        x = torch.stack([u0, grid1, grid2], dim=-1)
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.proj1(x))
        return self.proj2(x).squeeze(-1)


# ── U-Net FNO (UNO) ──────────────────────────────────────────────────────────

class UNO1d(nn.Module):
    """U-Net FNO for 1D. Simplified version with spectral down/up-sampling."""
    def __init__(self, n_modes: int, hidden_dim: int, in_ch: int = 2):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        
        self.enc1 = FNOBlock1d(hidden_dim, n_modes)
        self.enc2 = FNOBlock1d(hidden_dim * 2, n_modes // 2)
        
        self.dec2 = FNOBlock1d(hidden_dim * 4, n_modes // 2)
        self.dec1 = FNOBlock1d(hidden_dim * 2, n_modes)
        
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        if not isinstance(u0, torch.Tensor):
            u0 = torch.tensor(u0, dtype=torch.float32)
        
        # Ensure u0 is on the same device as the model weights
        device = next(self.parameters()).device
        u0 = u0.to(device)

        if u0.ndim == 1: u0 = u0.unsqueeze(0)
        B, N = u0.shape
        grid = torch.linspace(0.0, 1.0, N, device=device).view(1, N).expand(B, N)
        x = self.lift(torch.stack([u0, grid], dim=-1))
        
        x1 = self.enc1(x)
        # Spectral pooling (downsample)
        x2_in = torch.cat([x1, x1], dim=-1) 
        x2 = self.enc2(x2_in)
        
        # Upsample and concat
        x1_up = torch.cat([x2, x2], dim=-1)
        y1 = self.dec1(x1_up + x1)
        
        return self.proj(y1).squeeze(-1)


class UNO2d(nn.Module):
    """U-Net FNO for 2D."""
    def __init__(self, n_modes1: int, n_modes2: int, hidden_dim: int, in_ch: int = 3):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        self.enc1 = FNOBlock2d(hidden_dim, n_modes1, n_modes2)
        self.dec1 = FNOBlock2d(hidden_dim, n_modes1, n_modes2)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        if not isinstance(u0, torch.Tensor):
            u0 = torch.tensor(u0, dtype=torch.float32)
        
        # Ensure u0 is on the same device as the model weights
        device = next(self.parameters()).device
        u0 = u0.to(device)

        B, N1, N2 = u0.shape
        grid1 = torch.linspace(0.0, 1.0, N1, device=device).view(1, N1, 1).expand(B, N1, N2)
        grid2 = torch.linspace(0.0, 1.0, N2, device=device).view(1, 1, N2).expand(B, N1, N2)
        x = self.lift(torch.stack([u0, grid1, grid2], dim=-1))
        x = self.enc1(x)
        x = self.dec1(x)
        return self.proj(x).squeeze(-1)
