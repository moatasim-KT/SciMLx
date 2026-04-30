"""WaveBench — 2D Wave propagation with high-frequency components.

PDE on [0,2π)², periodic:
    u_tt = c^2 ∇²u

Simulates harmonic wave propagation to test spectral bias.
Exact Fourier-mode solution:
    u_hat(k, T) = u_hat_0(k) * cos(c|k|T) + u_t_hat_0(k) * sin(c|k|T) / (c|k|)

Here we release from rest, so u_t(0) = 0.
"""

import math
import torch
import numpy as np

from data.prepare import _random_ic_2d
from core.device import DEVICE, TORCH_DEVICE

C_SPEED = 2.0
T_FINAL = 1.0

METADATA = {
    "pde":      "u_tt = c²∇²u (High-frequency harmonic wave propagation)",
    "domain":   "[0,2π)², periodic",
    "solver":   "Analytic Fourier propagator (exact)",
    "t_final":  T_FINAL,
    "n_steps":  1,
    "in_shape": "B,N,N",
    "out_shape": "B,N,N",
    "notes":    "Tests model ability to resolve high-frequency waves (spectral bias).",
}

def make_ic(n: int, N: int, rng: np.random.RandomState) -> torch.Tensor:
    """Random surface anomaly with higher mode frequencies."""
    # scale higher modes to test high-frequency bias
    ic_np = _random_ic_2d(n, N, rng, n_modes=12, scale=0.5, offset=0.0)
    return torch.from_numpy(ic_np).to(TORCH_DEVICE)

def solve_batch(u0: torch.Tensor, T: float = T_FINAL) -> torch.Tensor:
    B, N, _ = u0.shape
    # k_int = np.fft.fftfreq(N, d=1.0 / N)
    k_int = torch.fft.fftfreq(N, d=1.0 / N, device=TORCH_DEVICE)
    kx, ky = torch.meshgrid(k_int, k_int, indexing="ij")
    omega = C_SPEED * torch.sqrt(kx**2 + ky**2)
    
    propagator = torch.cos(omega * T)[None, :, :]
    
    u0_d = u0.to(torch.float32)
    u_hat = torch.fft.fft2(u0_d, dim=(1, 2))
    uT_hat = u_hat * propagator
    uT = torch.fft.ifft2(uT_hat, dim=(1, 2)).real
    return uT.to(torch.float32)

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    inputs = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    # Ensure [B, N, N, 1] shape
    return inputs[..., None], targets[..., None]
