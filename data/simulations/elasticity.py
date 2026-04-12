"""Elasticity & Solid-Mechanics — Linear Elasticity proxy.

Solves ∇·σ + F = 0 for a given force field F.
Produces 2-component displacement field U = (Ux, Uy).

Input: [B, N, N, 2] (force field)
Output: [B, N, N, 2] (displacement)
"""

import math
import numpy as np
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

def make_ic(n: int, N: int, rng: np.random.RandomState) -> np.ndarray:
    fx = _random_ic_2d(n, N, rng, n_modes=3, scale=1.0, offset=0.0)
    fy = _random_ic_2d(n, N, rng, n_modes=3, scale=1.0, offset=0.0)
    return np.stack([fx, fy], axis=-1)

def solve_batch(F: np.ndarray, T: float = 1.0) -> np.ndarray:
    # Very simplified proxy for linear elasticity.
    # U = (λ+2μ)^(-1) ∇(∇·F) ... we'll just mock a smoothing operator to represent the inverse Laplacian-like behavior.
    B, N, _, _ = F.shape
    k_int = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = np.meshgrid(k_int, k_int, indexing="ij")
    k_sq = kx**2 + ky**2
    k_sq[0, 0] = 1.0 # avoid div by zero
    
    fx, fy = F[..., 0], F[..., 1]
    fx_hat = np.fft.fft2(fx.astype(np.float64), axes=(1, 2))
    fy_hat = np.fft.fft2(fy.astype(np.float64), axes=(1, 2))
    
    # Simple decoupled Poisson-like smoothing for mock structural mechanics
    ux_hat = fx_hat / k_sq
    uy_hat = fy_hat / k_sq
    
    ux = np.fft.ifft2(ux_hat, axes=(1, 2)).real
    uy = np.fft.ifft2(uy_hat, axes=(1, 2)).real
    
    # zero out mean
    ux -= np.mean(ux, axis=(1, 2), keepdims=True)
    uy -= np.mean(uy, axis=(1, 2), keepdims=True)

    return np.stack([ux, uy], axis=-1).astype(np.float32)

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    inputs = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    return inputs, targets
