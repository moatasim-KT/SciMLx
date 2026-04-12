"""Radiative Transfer — Integro-differential scattering proxy.

Domain is 1D space, 1D angle, wrapped into a [B, N, N] tensor.
    u(x, θ)
This module acts as a proxy for highly forward-peaked scattering.

Input: [B, N, N]
Output: [B, N, N]
"""

import math
import numpy as np
from prepare import _random_ic_2d

T_FINAL = 1.0

METADATA = {
    "pde":      "Radiative Transfer Proxy",
    "domain":   "x ∈ [0,2π), θ ∈ [0,2π)",
    "solver":   "Fourier pseudo-spectral",
    "t_final":  T_FINAL,
    "n_steps":  1,
    "in_shape": "B,N,N",
    "out_shape": "B,N,N",
    "notes":    "Promotes hybrids that embed angular attention.",
}

def make_ic(n: int, N: int, rng: np.random.RandomState) -> np.ndarray:
    return _random_ic_2d(n, N, rng, n_modes=4, scale=1.0, offset=0.0)

def solve_batch(u0: np.ndarray, T: float = T_FINAL) -> np.ndarray:
    B, N, _ = u0.shape
    u = u0.astype(np.float64)
    # mock logic: 1D advection in x, scattering/blurring in θ
    k_int = np.fft.fftfreq(N, d=1.0 / N)
    kx, ktheta = np.meshgrid(k_int, k_int, indexing="ij")
    
    # dt advection
    steps = 10
    dt = T / steps
    for _ in range(steps):
        u_hat = np.fft.fft2(u, axes=(1, 2))
        
        # c * u_x + scattering in theta
        # u_theta_theta acts as blurring
        op = -0.5 * 1j * kx - 0.1 * (ktheta**2)
        u_hat = u_hat + dt * op * u_hat
        u = np.fft.ifft2(u_hat, axes=(1, 2)).real

    return u.astype(np.float32)

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    inputs = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    # Ensure [B, N, N, 1] shape
    return inputs[..., None], targets[..., None]
