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
import numpy as np

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

def make_ic(n: int, N: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate random smooth ICs for compressible Euler on [0, 2π).

    Returns float32 array of shape [n, N, 3] with channels (ρ, u, p).
    Amplitudes chosen for smooth subsonic flow (no shocks at T=1).
    """
    x = 2.0 * math.pi * np.arange(N, dtype=np.float64) / N

    def _fourier(n_modes: int, amp: float) -> np.ndarray:
        k       = np.arange(1, n_modes + 1, dtype=np.float64)
        decay   = k ** -1.5
        cos_c   = rng.randn(n, n_modes) * decay * amp
        sin_c   = rng.randn(n, n_modes) * decay * amp
        angles  = k[:, None] * x[None, :]      # [n_modes, N]
        return (cos_c @ np.cos(angles) + sin_c @ np.sin(angles))  # [n, N]

    rho = np.clip(1.0 + _fourier(8, 0.15), 0.4, 3.0)
    u   = _fourier(8, 0.10)
    p   = np.clip(1.0 + _fourier(8, 0.12), 0.2, 3.0)

    return np.stack([rho, u, p], axis=-1).astype(np.float32)  # [n, N, 3]


# ── Conservative ↔ primitive conversions ──────────────────────────────────────

def _prim2cons(prims: np.ndarray) -> np.ndarray:
    """(ρ, u, p) → (ρ, ρu, E).  prims: [..., 3]"""
    rho, u, p = prims[..., 0], prims[..., 1], prims[..., 2]
    return np.stack([rho,
                     rho * u,
                     p / (GAMMA - 1.0) + 0.5 * rho * u**2], axis=-1)


def _cons2prim(cons: np.ndarray) -> np.ndarray:
    """(ρ, ρu, E) → (ρ, u, p).  cons: [..., 3]"""
    rho = np.maximum(cons[..., 0], 1e-8)
    u   = cons[..., 1] / rho
    E   = cons[..., 2]
    p   = np.maximum((GAMMA - 1.0) * (E - 0.5 * rho * u**2), 1e-8)
    return np.stack([rho, u, p], axis=-1)


def _flux(cons: np.ndarray) -> np.ndarray:
    """Physical Euler flux F(U).  cons: [..., 3] → [..., 3]"""
    rho = np.maximum(cons[..., 0], 1e-8)
    u   = cons[..., 1] / rho
    E   = cons[..., 2]
    p   = np.maximum((GAMMA - 1.0) * (E - 0.5 * rho * u**2), 1e-8)
    return np.stack([rho * u,
                     rho * u**2 + p,
                     (E + p) * u], axis=-1)


# ── HLL Riemann solver ─────────────────────────────────────────────────────────

def _hll_flux(UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
    """HLL numerical flux at cell interfaces.  UL, UR: [B, N, 3]"""
    primL = _cons2prim(UL)
    primR = _cons2prim(UR)
    rhoL, uL, pL = primL[..., 0], primL[..., 1], primL[..., 2]
    rhoR, uR, pR = primR[..., 0], primR[..., 1], primR[..., 2]

    aL = np.sqrt(GAMMA * pL / rhoL)
    aR = np.sqrt(GAMMA * pR / rhoR)

    sL = np.minimum(uL - aL, uR - aR)             # left signal speed
    sR = np.maximum(uL + aL, uR + aR)             # right signal speed

    FL, FR  = _flux(UL), _flux(UR)
    denom   = np.maximum(sR - sL, 1e-10)[..., None]
    F_hll   = (sR[..., None] * FL - sL[..., None] * FR
               + sL[..., None] * sR[..., None] * (UR - UL)) / denom

    return np.where(sL[..., None] >= 0, FL,
           np.where(sR[..., None] <= 0, FR, F_hll))


# ── MUSCL reconstruction + RHS ────────────────────────────────────────────────

def _minmod2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b <= 0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))


def _rhs(U: np.ndarray, dx: float) -> np.ndarray:
    """Conservative finite-volume RHS: −(F_{i+1/2} − F_{i−1/2}) / dx."""
    Um1   = np.roll(U, 1, axis=1)
    Up1   = np.roll(U, -1, axis=1)
    slope = _minmod2(U - Um1, Up1 - U)

    UL    = U + 0.5 * slope                       # left  state at face i+1/2
    UR    = np.roll(U - 0.5 * slope, -1, axis=1)  # right state at face i+1/2

    F     = _hll_flux(UL, UR)
    return -(F - np.roll(F, 1, axis=1)) / dx


# ── Batch solver ──────────────────────────────────────────────────────────────

def solve_batch(prims0: np.ndarray,
                T: float     = T_FINAL,
                n_steps: int = N_STEPS) -> np.ndarray:
    """Evolve Euler 1D from t=0 to T.

    Args:
        prims0: [B, N, 3] float32 — initial (ρ, u, p)
        T:      final time
        n_steps: number of SSP-RK2 steps (fixed dt = T / n_steps)

    Returns:
        [B, N, 3] float32 — final (ρ, u, p)
    """
    B, N, _ = prims0.shape
    dx = 2.0 * math.pi / N
    dt = T / n_steps

    U = _prim2cons(prims0.astype(np.float64))

    for _ in range(n_steps):
        # SSP-RK2 (Shu-Osher)
        L0 = _rhs(U, dx)
        U1 = U + dt * L0
        # positivity guard
        U1[..., 0] = np.maximum(U1[..., 0], 1e-8)
        L1 = _rhs(U1, dx)
        U  = 0.5 * (U + U1 + dt * L1)
        U[..., 0] = np.maximum(U[..., 0], 1e-8)   # density floor

    return _cons2prim(U).astype(np.float32)


# ── Dataset helper ────────────────────────────────────────────────────────────

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng    = np.random.RandomState(seed)
    inputs = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    return inputs, targets
