"""PDEBench — Broad PDE family (Advection-Diffusion-Reaction).

PDE on [0,2π)², periodic:
    u_t = D ∇²u - c_x u_x - c_y u_y + R u (1 - u)

Input: [B, N, N]
Output: [B, N, N]
"""

import math
import torch
import numpy as np
from core.device import DEVICE, TORCH_DEVICE
from data.prepare import _random_ic_2d

T_FINAL = 1.0
D = 0.02
C_X, C_Y = 0.5, 0.5
R = 0.1

METADATA = {
    "pde":      "Advection-Diffusion-Reaction (PDEBench proxy)",
    "domain":   "[0,2π)², periodic",
    "solver":   "Fourier pseudo-spectral with Euler stepping",
    "t_final":  T_FINAL,
    "n_steps":  1,
    "in_shape": "B,N,N",
    "out_shape": "B,N,N",
    "notes":    "Stress test for multiscale and non-linear interactions.",
}

def make_ic(n: int, N: int, rng: np.random.RandomState) -> torch.Tensor:
    u0 = _random_ic_2d(n, N, rng, n_modes=4, scale=0.1, offset=0.1)
    return torch.from_numpy(u0).to(TORCH_DEVICE)

def solve_batch(u0: torch.Tensor | np.ndarray, T: float = T_FINAL) -> torch.Tensor:
    if isinstance(u0, np.ndarray):
        u0 = torch.from_numpy(u0).to(TORCH_DEVICE)
    else:
        u0 = u0.to(TORCH_DEVICE)

    B, N, _ = u0.shape
    k_int = torch.fft.fftfreq(N, d=1.0 / N, device=TORCH_DEVICE)
    kx, ky = torch.meshgrid(k_int, k_int, indexing="ij")
    k_sq = kx**2 + ky**2
    
    u = u0.to(torch.float32)
    steps = 500
    dt = T / steps
    
    for _ in range(steps):
        u_hat = torch.fft.fft2(u, dim=(1, 2))
        
        # linear part
        u_hat_lin = -D * k_sq * u_hat - 1j * C_X * kx * u_hat - 1j * C_Y * ky * u_hat
        u_lin = torch.fft.ifft2(u_hat_lin, dim=(1, 2)).real
        
        # non-linear part (reaction)
        u_react = R * u * (1 - u)
        
        u = u + dt * (u_lin + u_react)

    return u.to(torch.float32)

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    inputs = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    # Ensure [B, N, N] shape (standard 2D convention)
    return inputs.to(torch.float32), targets.to(torch.float32)
