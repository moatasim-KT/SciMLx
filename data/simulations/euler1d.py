"""Compressible Euler 1D — HLL finite volume solver.

PDE (conservative form) on [0, 2π), periodic BCs:
    ρ_t  + (ρu)_x           = 0        (mass)
    (ρu)_t + (ρu² + p)_x   = 0        (momentum)
    E_t  + ((E+p)u)_x       = 0        (energy)

    E = p/(γ−1) + ½ρu²,   γ = 1.4 (ideal gas)

Scheme: MUSCL linear reconstruction (minmod limiter) + HLL Riemann solver + SSP-RK2.
ML interface: (ρ₀, u₀, p₀) → (ρ_T, u_T, p_T) as [B, N, 3] float32 arrays.

References:
    Toro (2009) "Riemann Solvers and Numerical Methods for Fluid Dynamics" Ch.10
    Shu & Osher (1988) "Efficient implementation of essentially non-oscillatory schemes"
"""

import math
import torch
import numpy as np
from core.device import DEVICE, TORCH_DEVICE

# ── Physical constants ─────────────────────────────────────────────────────────

GAMMA     = 1.4
T_FINAL   = 1.0
N_STEPS   = 300    # CFL ≈ 0.04 for smooth subsonic ICs — very conservative
CFL       = 0.45   # Courant number for adaptive dt selection
N_CHANNELS = 3     # (ρ, u, p) channels

METADATA = {
    "pde":      "ρ_t+(ρu)_x=0; (ρu)_t+(ρu²+p)_x=0; E_t+((E+p)u)_x=0",
    "domain":   "[0,2π), periodic",
    "solver":   "MUSCL-HLL + SSP-RK2  (finite volume, 2nd-order)",
    "t_final":  T_FINAL,
    "n_steps":  N_STEPS,
    "in_shape": "B,N,3",
    "out_shape": "B,N,3",
    "notes":    "Multi-channel: (rho,u,p) → (rho,u,p). Smooth subsonic ICs.",
}


# ── IC generator ───────────────────────────────────────────────────────────────

def make_ic(n: int, N: int, rng: np.random.RandomState) -> torch.Tensor:
    """Generate random smooth ICs for compressible Euler on [0, 2π)."""
    x = 2.0 * math.pi * torch.arange(N, dtype=torch.float32, device=TORCH_DEVICE) / N

    def _fourier(n_modes: int, amp: float) -> torch.Tensor:
        k       = torch.arange(1, n_modes + 1, dtype=torch.float32, device=TORCH_DEVICE)
        decay   = k ** -1.5
        cos_c   = torch.from_numpy(rng.randn(n, n_modes)).to(TORCH_DEVICE) * decay * amp
        sin_c   = torch.from_numpy(rng.randn(n, n_modes)).to(TORCH_DEVICE) * decay * amp
        angles  = k[:, None] * x[None, :]      # [n_modes, N]
        return (cos_c @ torch.cos(angles) + sin_c @ torch.sin(angles))  # [n, N]

    rho = torch.clamp(1.0 + _fourier(8, 0.15), 0.4, 3.0)
    u   = _fourier(8, 0.10)
    p   = torch.clamp(1.0 + _fourier(8, 0.12), 0.2, 3.0)

    return torch.stack([rho, u, p], dim=-1).to(torch.float32)  # [n, N, 3]


# ── Conservative ↔ primitive conversions ──────────────────────────────────────

def _prim2cons(prims: torch.Tensor) -> torch.Tensor:
    """(ρ, u, p) → (ρ, ρu, E).  prims: [..., 3]"""
    rho, u, p = prims[..., 0], prims[..., 1], prims[..., 2]
    return torch.stack([rho,
                     rho * u,
                     p / (GAMMA - 1.0) + 0.5 * rho * u**2], dim=-1)


def _cons2prim(cons: torch.Tensor) -> torch.Tensor:
    """(ρ, ρu, E) → (ρ, u, p).  cons: [..., 3]"""
    rho = torch.maximum(cons[..., 0], torch.tensor(1e-8, device=TORCH_DEVICE, dtype=cons.dtype))
    u   = cons[..., 1] / rho
    E   = cons[..., 2]
    p   = torch.maximum((GAMMA - 1.0) * (E - 0.5 * rho * u**2), torch.tensor(1e-8, device=TORCH_DEVICE, dtype=cons.dtype))
    return torch.stack([rho, u, p], dim=-1)


def _flux(cons: torch.Tensor) -> torch.Tensor:
    """Physical Euler flux F(U).  cons: [..., 3] → [..., 3]"""
    rho = torch.maximum(cons[..., 0], torch.tensor(1e-8, device=TORCH_DEVICE, dtype=cons.dtype))
    u   = cons[..., 1] / rho
    E   = cons[..., 2]
    p   = torch.maximum((GAMMA - 1.0) * (E - 0.5 * rho * u**2), torch.tensor(1e-8, device=TORCH_DEVICE, dtype=cons.dtype))
    return torch.stack([rho * u,
                     rho * u**2 + p,
                     (E + p) * u], dim=-1)


# ── HLL Riemann solver ─────────────────────────────────────────────────────────

def _hll_flux(UL: torch.Tensor, UR: torch.Tensor) -> torch.Tensor:
    """HLL numerical flux at cell interfaces.  UL, UR: [B, N, 3]"""
    primL = _cons2prim(UL)
    primR = _cons2prim(UR)
    rhoL, uL, pL = primL[..., 0], primL[..., 1], primL[..., 2]
    rhoR, uR, pR = primR[..., 0], primR[..., 1], primR[..., 2]

    aL = torch.sqrt(GAMMA * pL / rhoL)
    aR = torch.sqrt(GAMMA * pR / rhoR)

    sL = torch.minimum(uL - aL, uR - aR)             # left signal speed
    sR = torch.maximum(uL + aL, uR + aR)             # right signal speed

    FL, FR  = _flux(UL), _flux(UR)
    denom   = torch.maximum(sR - sL, torch.tensor(1e-10, device=TORCH_DEVICE, dtype=sR.dtype))[..., None]
    F_hll   = (sR[..., None] * FL - sL[..., None] * FR
               + sL[..., None] * sR[..., None] * (UR - UL)) / denom

    mask_L = (sL >= 0)[..., None]
    mask_R = (sR <= 0)[..., None]
    
    return torch.where(mask_L, FL, torch.where(mask_R, FR, F_hll))


# ── MUSCL reconstruction + RHS ────────────────────────────────────────────────

def _minmod2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(a * b <= 0, torch.zeros_like(a), torch.where(torch.abs(a) < torch.abs(b), a, b))


def _rhs(U: torch.Tensor, dx: float) -> torch.Tensor:
    """Conservative finite-volume RHS: −(F_{i+1/2} − F_{i−1/2}) / dx."""
    Um1   = torch.roll(U, 1, dims=1)
    Up1   = torch.roll(U, -1, dims=1)
    slope = _minmod2(U - Um1, Up1 - U)

    UL    = U + 0.5 * slope                       # left  state at face i+1/2
    UR    = torch.roll(U - 0.5 * slope, -1, dims=1)  # right state at face i+1/2

    F     = _hll_flux(UL, UR)
    return -(F - torch.roll(F, 1, dims=1)) / dx


# ── Batch solver ──────────────────────────────────────────────────────────────

def solve_batch(prims0: torch.Tensor | np.ndarray,
                T: float     = T_FINAL,
                n_steps: int = N_STEPS) -> torch.Tensor:
    """Evolve Euler 1D from t=0 to T."""
    if isinstance(prims0, np.ndarray):
        prims0 = torch.from_numpy(prims0).to(TORCH_DEVICE)
    else:
        prims0 = prims0.to(TORCH_DEVICE)

    B, N, _ = prims0.shape
    dx = 2.0 * math.pi / N
    dt = T / n_steps

    U = _prim2cons(prims0.to(torch.float32))

    for _ in range(n_steps):
        # SSP-RK2 (Shu-Osher)
        L0 = _rhs(U, dx)
        U1 = U + dt * L0
        # positivity guard
        U1[..., 0] = torch.maximum(U1[..., 0], torch.tensor(1e-8, device=TORCH_DEVICE, dtype=U1.dtype))
        L1 = _rhs(U1, dx)
        U  = 0.5 * (U + U1 + dt * L1)
        U[..., 0] = torch.maximum(U[..., 0], torch.tensor(1e-8, device=TORCH_DEVICE, dtype=U.dtype))   # density floor

    return _cons2prim(U).to(torch.float32)


# ── Dataset helper ────────────────────────────────────────────────────────────

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    rng    = np.random.RandomState(seed)
    inputs = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    return inputs, targets
