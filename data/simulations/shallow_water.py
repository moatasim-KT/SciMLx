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
import torch
import numpy as np
from core.device import DEVICE

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

def make_ic(n: int, N: int, rng: np.random.RandomState) -> torch.Tensor:
    """Random smooth surface height anomaly (zero mean)."""
    h0 = _random_ic_2d(n, N, rng, n_modes=4, scale=0.15, offset=0.0)
    return torch.from_numpy(h0).to(DEVICE)


# ── Analytic solver ────────────────────────────────────────────────────────────

def solve_batch(h0: torch.Tensor | np.ndarray, T: float = T_FINAL) -> torch.Tensor:
    """Propagate surface height h₀ to time T using exact Fourier solution."""
    if isinstance(h0, np.ndarray):
        h0 = torch.from_numpy(h0).to(DEVICE)
    else:
        h0 = h0.to(DEVICE)

    B, N, _ = h0.shape

    # Integer wavenumbers on [0, 2π)²
    k_int = torch.fft.fftfreq(N, d=1.0 / N, device=DEVICE)         
    kx, ky = torch.meshgrid(k_int, k_int, indexing="ij")  

    # Dispersion relation: ω_k = c * |k|  (gravity waves)
    omega = C_WAVE * torch.sqrt(kx**2 + ky**2)       

    # Exact propagation: ĥ(T) = ĥ₀ · cos(ωT)
    propagator = torch.cos(omega * T)[None, :, :]    

    h0_d  = h0.to(torch.float64)
    h_hat = torch.fft.fft2(h0_d, dim=(1, 2))       
    hT_hat = h_hat * propagator
    hT = torch.fft.ifft2(hT_hat, dim=(1, 2)).real

    return hT.to(torch.float32)


# ── Dataset helper ────────────────────────────────────────────────────────────

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    rng     = np.random.RandomState(seed)
    inputs  = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    return inputs, targets
