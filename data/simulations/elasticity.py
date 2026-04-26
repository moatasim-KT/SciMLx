"""Elasticity & Solid-Mechanics — Linear Elasticity proxy.

Solves ∇·σ + F = 0 for a given force field F.
Produces 2-component displacement field U = (Ux, Uy).

Input: [B, N, N, 2] (force field)
Output: [B, N, N, 2] (displacement)
"""

import math
import torch
import numpy as np
from core.device import DEVICE
from data.prepare import _random_ic_2d

METADATA = {
    "pde":      "Linear Elasticity (force to displacement)",
    "domain":   "[0,2π)², periodic",
    "solver":   "Fourier method",
    "t_final":  1.0,
    "n_steps":  1,
    "in_shape": "B,N,N,2",
    "out_shape": "B,N,N,2",
    "notes":    "Challenges multi-component output handling (tensor-valued).",
}

def make_ic(n: int, N: int, rng: np.random.RandomState) -> torch.Tensor:
    fx = _random_ic_2d(n, N, rng, n_modes=3, scale=1.0, offset=0.0)
    fy = _random_ic_2d(n, N, rng, n_modes=3, scale=1.0, offset=0.0)
    return torch.stack([torch.from_numpy(fx), torch.from_numpy(fy)], dim=-1).to(DEVICE)

def solve_batch(F: torch.Tensor | np.ndarray, T: float = 1.0) -> torch.Tensor:
    # Very simplified proxy for linear elasticity.
    if isinstance(F, np.ndarray):
        F = torch.from_numpy(F).to(DEVICE)
    else:
        F = F.to(DEVICE)

    B, N, _, _ = F.shape
    k_int = torch.fft.fftfreq(N, d=1.0 / N, device=DEVICE)
    kx, ky = torch.meshgrid(k_int, k_int, indexing="ij")
    k_sq = kx**2 + ky**2
    k_sq[0, 0] = 1.0 # avoid div by zero
    
    fx, fy = F[..., 0], F[..., 1]
    fx_hat = torch.fft.fft2(fx.to(torch.float64), dim=(1, 2))
    fy_hat = torch.fft.fft2(fy.to(torch.float64), dim=(1, 2))
    
    # Simple decoupled Poisson-like smoothing for mock structural mechanics
    ux_hat = fx_hat / k_sq
    uy_hat = fy_hat / k_sq
    
    ux = torch.fft.ifft2(ux_hat, dim=(1, 2)).real
    uy = torch.fft.ifft2(uy_hat, dim=(1, 2)).real
    
    # zero out mean
    ux -= torch.mean(ux, dim=(1, 2), keepdim=True)
    uy -= torch.mean(uy, dim=(1, 2), keepdim=True)

    return torch.stack([ux, uy], dim=-1).to(torch.float32)

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    inputs = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    return inputs, targets
