"""WaveBench — 2D Wave propagation with high-frequency components.

PDE on [0,2π)², periodic:
    u_tt = c^2 ∇²u

Simulates harmonic wave propagation to test spectral bias.
Exact Fourier-mode solution:
    u_hat(k, T) = u_hat_0(k) * cos(c|k|T) + u_t_hat_0(k) * sin(c|k|T) / (c|k|)

Here we release from rest, so u_t(0) = 0.
"""

import math
import numpy as np

from data.prepare import _random_ic_2d

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

def make_ic(n: int, N: int, rng: np.random.RandomState) -> np.ndarray:
    """Random surface anomaly with higher mode frequencies."""
    # scale higher modes to test high-frequency bias
    return _random_ic_2d(n, N, rng, n_modes=12, scale=0.5, offset=0.0)

def solve_batch(u0: np.ndarray, T: float = T_FINAL) -> np.ndarray:
    B, N, _ = u0.shape
    k_int = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = np.meshgrid(k_int, k_int, indexing="ij")
    omega = C_SPEED * np.sqrt(kx**2 + ky**2)
    
    propagator = np.cos(omega * T)[None, :, :]
    
    u0_d = u0.astype(np.float64)
    u_hat = np.fft.fft2(u0_d, axes=(1, 2))
    uT_hat = u_hat * propagator
    uT = np.fft.ifft2(uT_hat, axes=(1, 2)).real
    return uT.astype(np.float32)

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    inputs = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    # Ensure [B, N, N, 1] shape
    return inputs[..., None], targets[..., None]
