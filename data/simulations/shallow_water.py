"""2D Linearized Shallow Water Equations — analytic spectral propagator.

PDE (linearized around quiescent state H₀=1, u=v=0) on [0,2π)², periodic:
    h'_t + H₀(u_x + v_y) = 0
    u_t  + g·h'_x = 0
    v_t  + g·h'_y = 0

    g = 9.81 m/s²,  H₀ = 1.0 m,  c = √(gH₀) ≈ 3.13 m/s (gravity wave speed)

"Released from rest" IC (u₀=v₀=0) reduces to a 2D dispersive wave equation:
    h'_tt = gH₀ ∇²h'

with exact Fourier-mode solution:
    ĥ'(k, T) = ĥ'₀(k) · cos(ω_k · T),    ω_k = c·|k|

This is machine-precision "exact" — no time-stepping truncation error.

ML interface: h₀(x,y) → h_T(x,y) as [B, N, N] float32 arrays.

References:
    Pedlosky (1987) "Geophysical Fluid Dynamics" §3.2
    Vallis (2006) "Atmospheric and Oceanic Fluid Dynamics" §3.3
"""

import math
import numpy as np

from data.prepare import _random_ic_2d

# ── Physical constants ─────────────────────────────────────────────────────────

GRAVITY   = 9.81
MEAN_DEPTH = 1.0
C_WAVE    = math.sqrt(GRAVITY * MEAN_DEPTH)   # ≈ 3.13 m/s
T_FINAL   = 1.0    # gravity waves cross domain ~0.5× at T=1

METADATA = {
    "pde":      "h_tt = g·H₀·∇²h  (linearized SWE, gravity wave equation)",
    "domain":   "[0,2π)², periodic",
    "solver":   "Analytic Fourier propagator (exact, zero truncation error)",
    "t_final":  T_FINAL,
    "n_steps":  1,
    "in_shape": "B,N,N",
    "out_shape": "B,N,N",
    "notes":    ("Scalar input/output: h₀→h_T. Released from rest (u₀=v₀=0). "
                 "Tests FNO's ability to learn dispersive wave operators."),
}


# ── IC generator ───────────────────────────────────────────────────────────────

def make_ic(n: int, N: int, rng: np.random.RandomState) -> np.ndarray:
    """Random smooth surface height anomaly (zero mean).

    Returns float32 [n, N, N], values ~ [-0.3, 0.3] (amplitude ≪ H₀=1).
    """
    return _random_ic_2d(n, N, rng, n_modes=4, scale=0.15, offset=0.0)


# ── Analytic solver ────────────────────────────────────────────────────────────

def solve_batch(h0: np.ndarray, T: float = T_FINAL) -> np.ndarray:
    """Propagate surface height h₀ to time T using exact Fourier solution.

    Args:
        h0: [B, N, N] float32 — initial surface height anomaly
        T:  final time

    Returns:
        [B, N, N] float32 — surface height at time T
    """
    B, N, _ = h0.shape

    # Integer wavenumbers on [0, 2π)²
    k_int = np.fft.fftfreq(N, d=1.0 / N)         # [0,1,...,N/2,-N/2+1,...,-1]
    kx, ky = np.meshgrid(k_int, k_int, indexing="ij")  # [N, N]

    # Dispersion relation: ω_k = c * |k|  (gravity waves)
    omega = C_WAVE * np.sqrt(kx**2 + ky**2)       # [N, N]

    # Exact propagation: ĥ(T) = ĥ₀ · cos(ωT)
    propagator = np.cos(omega * T)[None, :, :]    # [1, N, N]

    h0_d  = h0.astype(np.float64)
    h_hat = np.fft.fft2(h0_d, axes=(1, 2))       # [B, N, N] complex
    hT_hat = h_hat * propagator
    hT = np.fft.ifft2(hT_hat, axes=(1, 2)).real

    return hT.astype(np.float32)


# ── Dataset helper ────────────────────────────────────────────────────────────

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng     = np.random.RandomState(seed)
    inputs  = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    return inputs, targets
