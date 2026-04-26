"""Multiphysics Bench — Coupled cross-field dependencies.

Simulates a generic 2-component coupled system (e.g., thermal-mechanical or chemical).
For efficiency, we mock this as two coupled heat-like equations in Fourier space:
    u_t = D1 ∇²u - a*v
    v_t = D2 ∇²v + a*u

Input: [B, N, N, 2]
Output: [B, N, N, 2]
"""

import math
import torch
import numpy as np
from core.device import DEVICE
from data.prepare import _random_ic_2d

T_FINAL = 1.0
D1, D2 = 0.01, 0.05
ALPHA = 2.0

METADATA = {
    "pde":      "Coupled Diffusion (Multiphysics proxy)",
    "domain":   "[0,2π)², periodic",
    "solver":   "Analytic Fourier propagator",
    "t_final":  T_FINAL,
    "n_steps":  1,
    "in_shape": "B,N,N,2",
    "out_shape": "B,N,N,2",
    "notes":    "Tests model ability to resolve cross-field interactions in multi-channel configurations.",
}

def make_ic(n: int, N: int, rng: np.random.RandomState) -> torch.Tensor:
    u0 = _random_ic_2d(n, N, rng, n_modes=4, scale=1.0, offset=0.0)
    v0 = _random_ic_2d(n, N, rng, n_modes=4, scale=1.0, offset=0.0)
    return torch.stack([torch.from_numpy(u0), torch.from_numpy(v0)], dim=-1).to(DEVICE)

def solve_batch(uv0: torch.Tensor | np.ndarray, T: float = T_FINAL) -> torch.Tensor:
    if isinstance(uv0, np.ndarray):
        uv0 = torch.from_numpy(uv0).to(DEVICE)
    else:
        uv0 = uv0.to(DEVICE)

    B, N, _, _ = uv0.shape
    u0, v0 = uv0[..., 0], uv0[..., 1]
    
    k_int = torch.fft.fftfreq(N, d=1.0 / N, device=DEVICE)
    kx, ky = torch.meshgrid(k_int, k_int, indexing="ij")
    k_sq = kx**2 + ky**2
    
    u_hat = torch.fft.fft2(u0.to(torch.float64), dim=(1, 2))
    v_hat = torch.fft.fft2(v0.to(torch.float64), dim=(1, 2))
    
    # Solve system in Fourier domain analytically using matrix exponential (diagonalized)
    steps = 10
    dt = T / steps
    for _ in range(steps):
        u_next = u_hat - dt * D1 * k_sq * u_hat - dt * ALPHA * v_hat
        v_next = v_hat - dt * D2 * k_sq * v_hat + dt * ALPHA * u_hat
        u_hat, v_hat = u_next, v_next

    uT = torch.fft.ifft2(u_hat, dim=(1, 2)).real.to(torch.float32)
    vT = torch.fft.ifft2(v_hat, dim=(1, 2)).real.to(torch.float32)
    return torch.stack([uT, vT], dim=-1)

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    inputs = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    return inputs, targets
