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

import numpy as np

from prepare import _random_ic_2d

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

def make_ic(n: int, N: int, rng: np.random.RandomState) -> np.ndarray:
    """Random mixed-phase initial conditions via tanh-smoothed GRF.

    A raw GRF mapped through tanh gives a field with diffuse ±1 regions
    (physical Allen-Cahn initial data: random droplets/domains).
    The scale factor 2/ε stretches the GRF to create sharp but resolved
    interfaces of width ~ε at the zero crossings.
    """
    raw   = _random_ic_2d(n, N, rng, n_modes=6, scale=1.5, offset=0.0)
    scale = 1.0 / (np.sqrt(2.0) * EPSILON)
    return np.tanh(raw * scale).astype(np.float32)


# ── ETDRK2 solver ─────────────────────────────────────────────────────────────

def _etdrk2_coeffs(L_hat: np.ndarray, dt: float):
    """Precompute ETDRK2 (Cox-Matthews) integration coefficients.

    Returns E, c1, c2 arrays with L'Hôpital limits applied at L→0.
    """
    eps_zero = 1e-10
    E   = np.exp(L_hat * dt)
    Ls  = np.where(np.abs(L_hat) < eps_zero, eps_zero, L_hat)

    # φ₁(z) = (e^z - 1)/z  →  dt at z→0
    c1  = np.where(np.abs(L_hat) < eps_zero, dt,
                   (E - 1.0) / Ls)

    # φ₂(z) = (e^z - 1 - z) / z² dt  →  dt/2 at z→0
    c2  = np.where(np.abs(L_hat) < eps_zero, dt / 2.0,
                   (E - 1.0 - L_hat * dt) / (Ls**2 * dt))
    return E, c1, c2


def solve_batch(phi0: np.ndarray,
                T: float     = T_FINAL,
                n_steps: int = N_STEPS) -> np.ndarray:
    """Evolve Allen-Cahn phase field from t=0 to T via ETDRK2.

    Args:
        phi0:    [B, N, N] float32 — initial phase field
        T:       final time
        n_steps: number of ETDRK2 steps (dt = T / n_steps)

    Returns:
        [B, N, N] float32 — phase field at time T
    """
    B, N, _ = phi0.shape
    dt = T / n_steps

    # Spectral Laplacian on [0,1]²: ∇² → −|2πk|²
    k_int = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = np.meshgrid(k_int, k_int, indexing="ij")
    k_sq = kx**2 + ky**2                              # [N, N]  (integer-wavenumber squared)
    L_hat = -(EPSILON**2) * (2.0 * np.pi)**2 * k_sq   # spectral linear operator

    # Precompute ETDRK2 coefficients (same for every step)
    E, c1, c2 = _etdrk2_coeffs(L_hat, dt)
    E   = E[None]    # [1, N, N] for broadcasting with [B, N, N]
    c1  = c1[None]
    c2  = c2[None]

    def _N_hat(phi):
        """Nonlinear term N(φ) = φ − φ³, returned in spectral space."""
        Nphys = phi - phi**3
        return np.fft.fft2(Nphys, axes=(1, 2))

    phi = phi0.astype(np.float64)

    for _ in range(n_steps):
        phi_hat = np.fft.fft2(phi, axes=(1, 2))
        N0_hat  = _N_hat(phi)

        # ETDRK2 predictor
        phi_hat_star = E * phi_hat + c1 * N0_hat
        phi_star     = np.fft.ifft2(phi_hat_star, axes=(1, 2)).real

        # ETDRK2 corrector
        Na_hat   = _N_hat(phi_star)
        phi_hat  = phi_hat_star + c2 * (Na_hat - N0_hat)
        phi      = np.fft.ifft2(phi_hat, axes=(1, 2)).real

        # Bound projection (numerical stability guard)
        phi = np.clip(phi, -1.0 - 1e-4, 1.0 + 1e-4)

    return phi.astype(np.float32)


# ── Dataset helper ────────────────────────────────────────────────────────────

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng     = np.random.RandomState(seed)
    inputs  = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    return inputs, targets
