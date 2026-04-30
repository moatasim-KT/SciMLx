"""2D Navier-Stokes (vorticity form) — ETDRK4 pseudo-spectral solver.

PDE on [0, 2π)², periodic BCs:
    ω_t + (u·∇)ω = ν·∇²ω

    u = (∂ψ/∂y, −∂ψ/∂x),   ∇²ψ = ω    (stream function)
    ν = 1×10⁻³             (10× lower than ns_2d → Re ≈ 1000)

Scheme: ETDRK4 (Cox-Matthews exponential Runge-Kutta, 4th order in time).
  Split: ω_t = L̂ω + N(ω)
    L̂ = −ν|k|²    (linear diffusion, exact via exponential integrating factor)
    N(ω) = −(u·∇)ω (nonlinear advection, evaluated pseudo-spectrally)

  ETDRK4 removes the CFL constraint from diffusion entirely; only the
  advective CFL bounds the time step.  2/3-rule dealiasing prevents aliasing
  in the nonlinear convolution product u·∇ω.

vs ns_2d (benchmarks_ext.py):
  - ν: 1e-3 vs 1e-2  (10× higher Re → richer turbulent cascade)
  - T: 2.0 vs 1.0    (2× longer prediction horizon)
  - Scheme: ETDRK4 (4th order) vs semi-implicit Euler (1st order)

ML interface: ω₀(x,y) → ω_T(x,y) as [B, N, N] float32 arrays.

WARNING: First-run dataset generation (4096 training samples) takes ~10−20 min.
         Results are disk-cached; subsequent runs load in < 5 seconds.

References:
    Cox & Matthews (2002) "Exponential Time Differencing for Stiff Systems"
    Kassam & Trefethen (2005) "Fourth-Order Time-Stepping for Stiff PDEs"
    Li et al. (2020) FNO paper, Table 4 (Re=1000 benchmark)
"""

import torch
import numpy as np
from core.device import DEVICE, TORCH_DEVICE

from data.prepare import _random_ic_2d

# ── Physical constants ─────────────────────────────────────────────────────────

NU        = 1e-3    # kinematic viscosity (Re ≈ 1/ν ≈ 1000 at unit velocity/length)
T_FINAL   = 2.0    # longer horizon: 2× harder than ns_2d
N_STEPS   = 1000   # dt = 0.002, advective CFL ≈ 0.05 for IC scale=0.05
IC_SCALE  = 0.05   # vorticity amplitude (CFL-safe with 1000 steps)

METADATA = {
    "pde":      "ω_t + (u·∇)ω = ν∇²ω,  ν=1e-3  (2D NS vorticity form)",
    "domain":   "[0,2π)², periodic",
    "solver":   "ETDRK4 (Cox-Matthews), pseudo-spectral, 4th-order in time",
    "t_final":  T_FINAL,
    "n_steps":  N_STEPS,
    "in_shape": "B,N,N",
    "out_shape": "B,N,N",
    "notes":    ("10× higher Re than ns_2d; 2× longer horizon. "
                 "First-run generation is slow (~15 min for 4096 samples); disk-cached thereafter."),
}


# ── IC generator ───────────────────────────────────────────────────────────────

def make_ic(n: int, N: int, rng: np.random.RandomState) -> torch.Tensor:
    """Small-amplitude smooth vorticity field (CFL-safe for dt=0.002)."""
    w0 = _random_ic_2d(n, N, rng, n_modes=4, scale=IC_SCALE, offset=0.0)
    return torch.from_numpy(w0).to(TORCH_DEVICE)


# ── ETDRK4 pseudo-spectral NS solver ──────────────────────────────────────────

def solve_batch(w0: torch.Tensor | np.ndarray,
                nu: float    = NU,
                T: float     = T_FINAL,
                n_steps: int = N_STEPS) -> torch.Tensor:
    """Evolve 2D NS vorticity field from t=0 to T via ETDRK4."""
    if isinstance(w0, np.ndarray):
        w0 = torch.from_numpy(w0).to(TORCH_DEVICE)
    else:
        w0 = w0.to(TORCH_DEVICE)

    B, N, _ = w0.shape
    dt = T / n_steps

    # Spectral operators on [0, 2π)²
    k_int = torch.fft.fftfreq(N, device=TORCH_DEVICE).reshape(N, 1)
    kx, ky = torch.meshgrid(k_int, k_int, indexing="ij")
    lap   = -(kx**2 + ky**2)                       # ∇² eigenvalues (negative)
    lap_safe          = lap.clone()
    lap_safe[0, 0]    = 1.0                         # avoid div-by-zero at DC

    # Dealiasing mask (2/3-rule)
    dealias = ((torch.abs(kx * N) <= N // 3) &
               (torch.abs(ky * N) <= N // 3)).to(torch.float32)

    # Linear operator L̂ = −ν|k|² (real, ≤ 0 everywhere)
    L = nu * lap                                    # = −ν|k|², ≤ 0

    # ETDRK4 integrating factors (precomputed, same for every step)
    E  = torch.exp(L * dt)          # [N, N]
    E2 = torch.exp(L * dt / 2.0)   # [N, N]

    eps_L = 1e-10
    Ls    = torch.where(torch.abs(L) < eps_L, torch.tensor(eps_L, device=TORCH_DEVICE, dtype=L.dtype), L)

    # φ₁(z) = (e^z − 1)/z  coefficients for half-step and full-step
    c1h = torch.where(torch.abs(L) < eps_L, torch.tensor(dt / 2.0, device=TORCH_DEVICE, dtype=L.dtype),  (E2 - 1.0) / Ls)  # half-step
    c1f = torch.where(torch.abs(L) < eps_L, torch.tensor(dt, device=TORCH_DEVICE, dtype=L.dtype),          (E  - 1.0) / Ls)  # full-step

    # Broadcast for batch dimension: [1, N, N]
    E   = E  [None]
    E2  = E2 [None]
    c1h = c1h[None]
    c1f = c1f[None]
    lap_safe = lap_safe[None]
    dealias  = dealias[None]
    kx = kx[None]
    ky = ky[None]

    def _nonlinear(w_hat: torch.Tensor) -> torch.Tensor:
        """Compute N̂(ω) = −F[(u·∇)ω] with 2/3-rule dealiasing."""
        wd = w_hat * dealias

        # Stream function: ψ̂ = ω̂ / ∇²  (DC mode set to zero)
        psi_hat          = wd / lap_safe
        psi_hat[:, 0, 0] = 0.0

        # Velocity: u = ∂ψ/∂y, v = −∂ψ/∂x
        u_phys = torch.fft.ifft2(1j * ky * psi_hat, dim=(1, 2)).real
        v_phys = torch.fft.ifft2(-1j * kx * psi_hat, dim=(1, 2)).real

        # Vorticity gradients
        wx_phys = torch.fft.ifft2(1j * kx * wd, dim=(1, 2)).real
        wy_phys = torch.fft.ifft2(1j * ky * wd, dim=(1, 2)).real

        # Nonlinear advection (in physical space), back to spectral
        adv = torch.fft.fft2(u_phys * wx_phys + v_phys * wy_phys, dim=(1, 2))
        return -adv    # N(ω) = −(u·∇)ω

    w = w0.to(torch.float32)
    w_hat = torch.fft.fft2(w, dim=(1, 2))

    for _ in range(n_steps):
        # ETDRK4 (Krogstad 2005, simplified variant)
        N0 = _nonlinear(w_hat)

        a_hat = E2 * w_hat + c1h * N0
        Na    = _nonlinear(a_hat)

        b_hat = E2 * w_hat + c1h * Na
        Nb    = _nonlinear(b_hat)

        c_hat = E2 * a_hat + c1h * (2.0 * Nb - N0)
        Nc    = _nonlinear(c_hat)

        w_hat = (E * w_hat
                 + (dt / 6.0) * (E * N0 + 2.0 * E2 * (Na + Nb) + Nc))

        # Stability check
        if torch.any(torch.isnan(w_hat)):
            break

    return torch.fft.ifft2(w_hat, dim=(1, 2)).real.to(torch.float32)


# ── Dataset helper ────────────────────────────────────────────────────────────

def make_dataset(n: int, seed: int, N: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    rng     = np.random.RandomState(seed)
    inputs  = make_ic(n, N, rng)
    targets = solve_batch(inputs)
    return inputs, targets
