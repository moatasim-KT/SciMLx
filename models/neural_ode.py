"""Neural ODEs and Universal Differential Equations (UDEs) for 1-D PDEs.

References:
  1. "Neural Ordinary Differential Equations"
     Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud
     NeurIPS 2018 — arXiv:1806.07366

  2. "Universal Differential Equations for Scientific Machine Learning"
     Christopher Rackauckas et al.
     arXiv:2001.04385

Source repo:
  https://github.com/moatasim-KT/SciML-and-Physics-Informed-Machine-Learning-Examples
  (universal-differential-equations example)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from core.device import DEVICE

# ── Derivative Network (shared by NeuralODE and UDE) ─────────────────────────

class DerivNet1d(nn.Module):
    """Neural network approximating the time derivative ∂u/∂t."""

    def __init__(self, hidden_dim: int = 32, n_modes: int = 16, n_layers: int = 3):
        super().__init__()
        from .fno import FNOBlock1d
        self.lift   = nn.Linear(2, hidden_dim)
        self.blocks = nn.ModuleList([FNOBlock1d(hidden_dim, n_modes) for _ in range(n_layers)])
        self.proj   = nn.Linear(hidden_dim, 1)
        self.to(DEVICE)

    def forward(self, u: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """u, grid : [B, N] → tendency : [B, N]"""
        x = torch.stack([u, grid], dim=-1)   # [B, N, 2]
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)[:, :, 0]       # [B, N]


# ── NeuralODE1d ───────────────────────────────────────────────────────────────

class NeuralODE1d(nn.Module):
    """Neural ODE for 1-D PDE operator learning."""

    def __init__(self, n_modes: int = 16, hidden_dim: int = 64, n_layers: int = 4,
                 n_steps: int = 20):
        super().__init__()
        self.n_steps = n_steps
        self.dt      = 1.0 / n_steps
        self.deriv   = DerivNet1d(hidden_dim=hidden_dim,
                                   n_modes=n_modes,
                                   n_layers=n_layers)
        self.to(DEVICE)

    def _rk4_step(self, u: torch.Tensor, grid: torch.Tensor, dt: float) -> torch.Tensor:
        """One RK4 step: u → u + dt*F(u)."""
        k1 = self.deriv(u,               grid)
        k2 = self.deriv(u + 0.5*dt*k1,  grid)
        k3 = self.deriv(u + 0.5*dt*k2,  grid)
        k4 = self.deriv(u + dt*k3,       grid)
        return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N = u0.shape
        grid = torch.linspace(0.0, 1.0, N, device=DEVICE).unsqueeze(0).expand(B, N)
        u = u0
        for _ in range(self.n_steps):
            u = self._rk4_step(u, grid, self.dt)
        return u


# ── UniversalDE1d ─────────────────────────────────────────────────────────────

class UniversalDE1d(nn.Module):
    """Universal Differential Equation for 1-D Burgers."""

    def __init__(self, n_modes: int = 16, hidden_dim: int = 32, n_layers: int = 3,
                 n_steps: int = 20, nu: float = 0.0):
        super().__init__()
        self.n_steps = n_steps
        self.dt      = 1.0 / n_steps
        self.nu      = nu
        self.correction = DerivNet1d(hidden_dim=hidden_dim,
                                      n_modes=n_modes,
                                      n_layers=n_layers)
        self.to(DEVICE)

    def _spectral_advection(self, u: torch.Tensor) -> torch.Tensor:
        """Compute -u * ∂u/∂x via spectral differentiation with 2/3 dealiasing."""
        _, N = u.shape
        u_hat = torch.fft.rfft(u, dim=-1)
        n_rfft = N // 2 + 1
        k = torch.arange(n_rfft, dtype=torch.float32, device=DEVICE)

        # 2/3 dealiasing
        k_max = int(n_rfft * 2 / 3)
        mask = (k < k_max).float()
        u_hat_d = u_hat * mask.unsqueeze(0)

        # Spectral derivative ∂u/∂x
        ux_hat = torch.complex(-u_hat_d.imag * k.unsqueeze(0), u_hat_d.real * k.unsqueeze(0))
        ux = torch.fft.irfft(ux_hat, n=N, dim=-1)
        return -u * ux

    def _spectral_diffusion(self, u: torch.Tensor) -> torch.Tensor:
        """Compute nu * d^2u/dx^2 via spectral differentiation."""
        if self.nu == 0.0:
            return torch.zeros_like(u)
        _, N = u.shape
        u_hat  = torch.fft.rfft(u, dim=-1)
        n_rfft = N // 2 + 1
        k      = torch.arange(n_rfft, dtype=torch.float32, device=DEVICE)
        uxx_hat = -(k ** 2).unsqueeze(0) * u_hat
        uxx    = torch.fft.irfft(uxx_hat, n=N, dim=-1)
        return self.nu * uxx

    def _tendency(self, u: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Full tendency: known physics + NN correction."""
        known = self._spectral_advection(u) + self._spectral_diffusion(u)
        nn_corr = self.correction(u, grid)
        tendency = known + nn_corr
        return torch.clamp(tendency, -1e4, 1e4)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N = u0.shape
        grid = torch.linspace(0.0, 1.0, N, device=DEVICE).unsqueeze(0).expand(B, N)
        u = u0
        dt = self.dt
        for _ in range(self.n_steps):
            k1 = self._tendency(u,               grid)
            k2 = self._tendency(u + 0.5*dt*k1,  grid)
            k3 = self._tendency(u + 0.5*dt*k2,  grid)
            k4 = self._tendency(u + dt*k3,       grid)
            u  = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return u


# ── LatentODE1d ───────────────────────────────────────────────────────────────

class LatentODE1d(nn.Module):
    """Latent Neural ODE for 1-D operator learning."""

    def __init__(self, n_sensors: int, hidden_dim: int = 64, latent_dim: int = 0,
                 n_layers: int = 4, n_steps: int = 20, n_modes: int = 16):
        super().__init__()
        if latent_dim == 0:
            latent_dim = hidden_dim // 2
        self.latent_dim = latent_dim
        self.n_steps    = n_steps
        self.dt         = 1.0 / n_steps

        def _mlp(in_d, out_d, act_cls=nn.GELU):
            dims = [in_d] + [hidden_dim] * n_layers + [out_d]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(act_cls())
            return nn.Sequential(*layers)

        self.encoder = _mlp(n_sensors, latent_dim)
        self.ode_fn  = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.Tanh(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.trunk   = _mlp(1, latent_dim)
        self.bias    = nn.Parameter(torch.zeros(1))
        self.to(DEVICE)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        _, N = u0.shape
        z = self.encoder(u0)
        for _ in range(self.n_steps):
            z = z + self.dt * self.ode_fn(z)
        grid = torch.linspace(0.0, 1.0, N, device=DEVICE).unsqueeze(-1)
        T    = self.trunk(grid)
        out  = torch.matmul(z, T.T) + self.bias
        return out
