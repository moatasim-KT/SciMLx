"""PDEBench — Broad PDE family (Advection-Diffusion-Reaction).

PDE on [0,2π)², periodic:
    u_t = D ∇²u - c_x u_x - c_y u_y + R u (1 - u)

Input: [B, N, N]
Output: [B, N, N]
"""

import math
import numpy as np
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

def make_ic(n: int, N: int, rng: np.random.RandomState) -> np.ndarray:
    return _random_ic_2d(n, N, rng, n_modes=4, scale=0.1, offset=0.1)

def solve_batch(u0: np.ndarray, T: float = T_FINAL) -> np.ndarray:
    B, N, _ = u0.shape
    k_int = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = np.meshgrid(k_int, k_int, indexing="ij")
    k_sq = kx**2 + ky**2
    
    u = u0.astype(np.float64)
    steps = 500
    dt = T / steps
    
    for _ in range(steps):
        u_hat = np.fft.fft2(u, axes=(1, 2))
        
        # linear part
        u_hat_lin = -D * k_sq * u_hat - 1j * C_X * kx * u_hat - 1j * C_Y * ky * u_hat
        u_lin = np.fft.ifft2(u_hat_lin, axes=(1, 2)).real
        
        # non-linear part (reaction)
        u_react = R * u * (1 - u)
        
        u = u + dt * (u_lin + u_react)

    return u.astype(np.float32)

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    inputs = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    # Ensure [B, N, N] shape (standard 2D convention)
    return inputs.astype(np.float32), targets.astype(np.float32)
