"""2D Allen-Cahn phase field equation — ETDRK2 spectral solver.

PDE on [0, 1]², periodic BCs:
    φ_t = ε²·∇²φ + φ − φ³,    φ ∈ [−1, 1]

    ε = 0.05  (interface width parameter; ε/dx ≈ 3.2 cells at N=64)

The Ginzburg-Landau double-well potential F(φ) = ¼(φ²−1)² drives phase
separation: φ→+1 (phase A) or φ→−1 (phase B) with diffuse interfaces of
width ~ε. This PDE models grain growth, solidification, and topology changes.

Scheme: ETDRK2 (Cox-Matthews exponential time differencing, 2nd order).
  Split: u_t = L(u) + N(u)
    L̂ = −ε²|k|²          (linear stiff diffusion, integrated exactly)
    N(φ) = φ − φ³         (cubic nonlinearity, explicit)

  Exact integrating factor avoids CFL constraint on the diffusion term.

ML interface: φ₀(x,y) → φ_T(x,y) as [B, N, N] float32 arrays.

References:
    Cox & Matthews (2002) "Exponential Time Differencing for Stiff Systems"
    Allen & Cahn (1979) "A microscopic theory for antiphase boundary motion"
"""

import torch
import numpy as np
from core.device import DEVICE

from data.prepare import _random_ic_2d

# ── Physical constants ─────────────────────────────────────────────────────────

EPSILON   = 0.05    # interface width (≥ 2·dx = 2/64 ≈ 0.031 for N=64)
T_FINAL   = 1.0
N_STEPS   = 200     # dt = 0.005, well within ETDRK2 stability for this N

METADATA = {
    "pde":      "φ_t = ε²·∇²φ + φ − φ³,  ε=0.05",
    "domain":   "[0,1]², periodic",
    "solver":   "ETDRK2 (Cox-Matthews), spectral, 2nd-order in time",
    "t_final":  T_FINAL,
    "n_steps":  N_STEPS,
    "in_shape": "B,N,N",
    "out_shape": "B,N,N",
    "notes":    ("Phase field: interface coarsening and topology changes. "
                 "φ initialized near ±1 transitions — physically realistic ICs."),
}


# ── IC generator ───────────────────────────────────────────────────────────────

def make_ic(n: int, N: int, rng: np.random.RandomState) -> torch.Tensor:
    """Random mixed-phase initial conditions via tanh-smoothed GRF."""
    raw   = _random_ic_2d(n, N, rng, n_modes=6, scale=1.5, offset=0.0)
    raw   = torch.from_numpy(raw).to(DEVICE)
    scale = 1.0 / (np.sqrt(2.0) * EPSILON)
    return torch.tanh(raw * scale).to(torch.float32)


# ── ETDRK2 solver ─────────────────────────────────────────────────────────────

def _etdrk2_coeffs(L_hat: torch.Tensor, dt: float):
    """Precompute ETDRK2 (Cox-Matthews) integration coefficients."""
    eps_zero = 1e-10
    E   = torch.exp(L_hat * dt)
    Ls  = torch.where(torch.abs(L_hat) < eps_zero, torch.tensor(eps_zero, device=DEVICE, dtype=L_hat.dtype), L_hat)

    # φ₁(z) = (e^z - 1)/z  →  dt at z→0
    c1  = torch.where(torch.abs(L_hat) < eps_zero, torch.tensor(dt, device=DEVICE, dtype=L_hat.dtype),
                   (E - 1.0) / Ls)

    # φ₂(z) = (e^z - 1 - z) / z² dt  →  dt/2 at z→0
    c2  = torch.where(torch.abs(L_hat) < eps_zero, torch.tensor(dt / 2.0, device=DEVICE, dtype=L_hat.dtype),
                   (E - 1.0 - L_hat * dt) / (Ls**2 * dt))
    return E, c1, c2


def solve_batch(phi0: torch.Tensor | np.ndarray,
                T: float     = T_FINAL,
                n_steps: int = N_STEPS) -> torch.Tensor:
    """Evolve Allen-Cahn phase field from t=0 to T via ETDRK2."""
    if isinstance(phi0, np.ndarray):
        phi0 = torch.from_numpy(phi0).to(DEVICE)
    else:
        phi0 = phi0.to(DEVICE)

    B, N, _ = phi0.shape
    dt = T / n_steps

    # Spectral Laplacian on [0,1]²: ∇² → −|2πk|²
    k_int = torch.fft.fftfreq(N, d=1.0 / N, device=DEVICE)
    kx, ky = torch.meshgrid(k_int, k_int, indexing="ij")
    k_sq = kx**2 + ky**2                              
    L_hat = -(EPSILON**2) * (2.0 * np.pi)**2 * k_sq   

    # Precompute ETDRK2 coefficients
    E, c1, c2 = _etdrk2_coeffs(L_hat, dt)
    E   = E[None]    
    c1  = c1[None]
    c2  = c2[None]

    def _N_hat(phi):
        """Nonlinear term N(φ) = φ − φ³, returned in spectral space."""
        Nphys = phi - phi**3
        return torch.fft.fft2(Nphys, dim=(1, 2))

    phi = phi0.to(torch.float64)

    for _ in range(n_steps):
        phi_hat = torch.fft.fft2(phi, dim=(1, 2))
        N0_hat  = _N_hat(phi)

        # ETDRK2 predictor
        phi_hat_star = E * phi_hat + c1 * N0_hat
        phi_star     = torch.fft.ifft2(phi_hat_star, dim=(1, 2)).real

        # ETDRK2 corrector
        Na_hat   = _N_hat(phi_star)
        phi_hat  = phi_hat_star + c2 * (Na_hat - N0_hat)
        phi      = torch.fft.ifft2(phi_hat, dim=(1, 2)).real

        # Bound projection (numerical stability guard)
        phi = torch.clamp(phi, -1.0 - 1e-4, 1.0 + 1e-4)

    return phi.to(torch.float32)


# ── Dataset helper ────────────────────────────────────────────────────────────

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    rng     = np.random.RandomState(seed)
    inputs  = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    return inputs, targets
