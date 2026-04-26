"""Extended benchmark definitions for SciML experiments.

Adds KdV, Wave, and corrected 2D benchmarks on top of prepare.py.

Supported benchmarks:
    "kdv_1d"       - Korteweg-de Vries soliton dynamics   (ETDRK4 solver)
    "wave_1d"      - 1D wave equation  u_tt = c^2 u_xx     (Stormer-Verlet)
    "darcy_2d" - 2D Darcy with proper variable-coeff solver (Richardson iter)
    "ns_2d"    - 2D Navier-Stokes with stable IC amplitude (CFL < 1)

Why darcy_2d and ns_2d?
    prepare.py's darcy_2d solver uses only mean(a) → loses all spatial info;
    source term f uses a FIXED seed independent of a → u is uncorrelated with a.
    prepare.py's ns_2d solver uses IC scale=1.0 → CFL≈61 → immediate NaN.
    Both benchmarks are broken at the data level; prepare.py is read-only.
    These fixed versions provide correct, learnable benchmarks.

Data interface is identical to prepare.py:
    make_ext_dataloader(benchmark, split, batch_size)
    evaluate_l2_rel_ext(benchmark, model)

References:
    KdV:  Tran et al. (2023) "Factorized Fourier Neural Operators"
    Wave: Rahman et al. (2022) U-NO
    Darcy fix: Li et al. (2020) FNO paper, original Darcy benchmark setup
    NS fix: standard semi-implicit spectral NS with CFL-stable parameters
"""

import math
import os
import time

import torch
import numpy as np

from data.prepare import (
    GRID_SIZE, TIME_BUDGET, N_TRAIN, N_VAL, VAL_SEED, TRAIN_SEED,
    CACHE_DIR,
    solve_kdv_batch, solve_wave_batch, _random_ic, _random_ic_np, _random_ic_2d,
)

# ── Constants ─────────────────────────────────────────────────────────────────

EXT_BENCHMARKS = {"kdv_1d", "wave_1d", "darcy_2d", "ns_2d", "swe_2d", "allen_cahn_2d", "mhd_2d", "burgers_nu_01", "burgers_nu_001", "poisson_2d", "reionization_1d"}

EXT_N_CHANNELS = {
    "kdv_1d": 1,
    "wave_1d": 1,
    "darcy_2d": 1,
    "ns_2d": 1,
    "ns_hre_2d": 1,
    "swe_2d": 1,
    "allen_cahn_2d": 1,
    "mhd_2d": 2, # Vorticity (w) and Magnetic Potential (a)
    "burgers_nu_01": 1,
    "burgers_nu_001": 1,
    "poisson_2d": 1,
    "reionization_1d": 1,
    "ellipse_2d": 1,
}

# KdV parameters
KDV_T       = 1.0    # final time
KDV_NSTEPS  = 1000   # ETDRK4 steps

# Wave parameters
WAVE_C      = 1.0    # wave speed
WAVE_T      = 1.0    # final time
WAVE_NSTEPS = 400    # Störmer-Verlet steps

# Darcy fix parameters
DARCY_FIX_N_ITER  = 40   # PCG iterations
DARCY_FIX_MODES_F = 5    # source term Fourier modes

# NS fix parameters — reduces CFL from ~61 to ~0.6
NS_SCALE  = 0.1     # IC vorticity amplitude (vs 1.0 in prepare.py -> 10x smaller)
NS_NSTEPS = 1000    # time steps (vs 100) — gives dt=0.001, CFL≈0.6
NS_NU     = 1e-2    # kinematic viscosity (same as original)
NS_T      = 1.0     # final time

# Allen-Cahn parameters
AC_EPSILON = 0.01
AC_T       = 0.5
AC_NSTEPS  = 200

# SWE parameters
SWE_G     = 9.81
SWE_T     = 0.2
SWE_NSTEPS = 200

# MHD parameters
MHD_NU    = 1e-3
MHD_ETA   = 1e-3
MHD_T     = 0.5
MHD_NSTEPS = 500

from core.device import DEVICE

# ── 2D Solvers (corrected) ────────────────────────────────────────────────────

def solve_darcy_2d_batch(
    a: torch.Tensor,
    f: torch.Tensor,
    n_iter: int = DARCY_FIX_N_ITER,
) -> torch.Tensor:
    """Solve -∇·(a(x,y)∇u) = f on [0,1]² with periodic BCs.

    Uses Preconditioned Conjugate Gradient (PCG) with the constant-coefficient
    Poisson operator P = a_mean·(-Δ) as a preconditioner.
    """
    B, N, _ = a.shape
    a_d = a.to(torch.float64)
    f_d = f.to(torch.float64)

    # Physical wavenumbers on [0,1]²: d/dx ↔ multiply by 2πi·k_int
    k_int = torch.fft.fftfreq(N, d=1.0 / N, device=DEVICE)
    kx, ky = torch.meshgrid(2 * math.pi * k_int, 2 * math.pi * k_int, indexing="ij")
    lap_pos = kx ** 2 + ky ** 2
    lap_pos[0, 0] = 1.0

    a_mean = a_d.mean(dim=(1, 2), keepdim=True)

    def apply_A(v):
        """Compute A·v = -∇·(a∇v) via spectral differentiation."""
        v_hat = torch.fft.fft2(v, dim=(1, 2))
        vx = torch.fft.ifft2(1j * kx[None] * v_hat, dim=(1, 2)).real
        vy = torch.fft.ifft2(1j * ky[None] * v_hat, dim=(1, 2)).real
        Av = -torch.fft.ifft2(
            1j * kx[None] * torch.fft.fft2(a_d * vx, dim=(1, 2))
            + 1j * ky[None] * torch.fft.fft2(a_d * vy, dim=(1, 2)),
            dim=(1, 2),
        ).real
        return Av

    def apply_P_inv(r):
        """Preconditioned step: P⁻¹r = r̂ / (a_mean · |k|²)."""
        r_hat = torch.fft.fft2(r, dim=(1, 2))
        Pr = torch.fft.ifft2(r_hat / (a_mean * lap_pos[None]), dim=(1, 2)).real
        Pr -= Pr.mean(dim=(1, 2), keepdim=True)  # project to zero-mean space
        return Pr

    u = torch.zeros((B, N, N), dtype=torch.float64, device=DEVICE)
    r = f_d - apply_A(u)
    r -= r.mean(dim=(1, 2), keepdim=True)
    
    z = apply_P_inv(r)
    p = z.clone()
    
    rz_old = torch.sum(r * z, dim=(1, 2), keepdim=True)
    
    for _ in range(n_iter):
        Ap = apply_A(p)
        pAp = torch.sum(p * Ap, dim=(1, 2), keepdim=True)
        alpha = rz_old / (pAp + 1e-16)
        
        u += alpha * p
        r -= alpha * Ap
        # Project residual to zero-mean space to avoid drift
        r -= r.mean(dim=(1, 2), keepdim=True)
        
        z = apply_P_inv(r)
        rz_new = torch.sum(r * z, dim=(1, 2), keepdim=True)
        
        if torch.max(torch.abs(rz_new)) < 1e-18:
            break
            
        beta = rz_new / (rz_old + 1e-16)
        p = z + beta * p
        rz_old = rz_new

    u -= u.mean(dim=(1, 2), keepdim=True)
    return u.to(torch.float32)


def solve_ns_2d_batch(
    w0: torch.Tensor,
    nu: float = NS_NU,
    T: float = NS_T,
    n_steps: int = NS_NSTEPS,
) -> torch.Tensor:
    """Stable 2D Navier-Stokes solver (vorticity form) on [0, 2pi)^2."""
    B, N, _ = w0.shape
    dt = T / n_steps

    k = torch.fft.fftfreq(N, device=DEVICE)
    k1, k2 = torch.meshgrid(k, k, indexing="ij")
    laplacian = -(k1 ** 2 + k2 ** 2)
    laplacian[0, 0] = 1.0

    cutoff = (2 * N) // 3  # 2/3-rule dealiasing

    w_hat = torch.fft.fft2(w0.to(torch.float64), dim=(1, 2))

    for _ in range(n_steps):
        # Dealias
        w_hat_d = w_hat.clone()
        mask = (torch.abs(k1 * N) > cutoff) | (torch.abs(k2 * N) > cutoff)
        w_hat_d[:, mask] = 0.0

        # Stream function: Δψ = ω
        psi_hat = w_hat_d / laplacian
        psi_hat[:, 0, 0] = 0.0

        # Velocity: u = (∂ψ/∂y, -∂ψ/∂x)
        u = torch.fft.ifft2(1j * k2 * psi_hat).real
        v = torch.fft.ifft2(-1j * k1 * psi_hat).real

        # Non-linear term: (u·∇)ω
        wx = torch.fft.ifft2(1j * k1 * w_hat_d).real
        wy = torch.fft.ifft2(1j * k2 * w_hat_d).real
        nonlin = torch.fft.fft2(u * wx + v * wy)

        # Semi-implicit step: diffusion implicit, advection explicit
        w_hat = (w_hat - dt * nonlin) / (1.0 - dt * nu * laplacian)

    return torch.fft.ifft2(w_hat, dim=(1, 2)).real.to(torch.float32)


def solve_allen_cahn_2d_batch(u0: torch.Tensor, epsilon: float = AC_EPSILON, T: float = AC_T, n_steps: int = AC_NSTEPS) -> torch.Tensor:
    """Semi-implicit spectral solver for Allen-Cahn 2D."""
    B, N, _ = u0.shape
    dt = T / n_steps
    k = torch.fft.fftfreq(N, device=DEVICE)
    k1, k2 = torch.meshgrid(k, k, indexing="ij")
    laplacian = -(k1 ** 2 + k2 ** 2)
    u_hat = torch.fft.fft2(u0.to(torch.float64), dim=(1, 2))
    for _ in range(n_steps):
        u = torch.fft.ifft2(u_hat).real
        nonlin = torch.fft.fft2(u**3 - u)
        u_hat = (u_hat - dt * nonlin) / (1.0 - dt * epsilon * laplacian)
    return torch.fft.ifft2(u_hat).real.to(torch.float32)


def solve_swe_2d_batch(h0: torch.Tensor, T: float = SWE_T, n_steps: int = SWE_NSTEPS) -> torch.Tensor:
    """Spectral solver for 2D Shallow Water Equations (linearized height)."""
    B, N, _ = h0.shape
    dt = T / n_steps
    k = torch.fft.fftfreq(N, device=DEVICE)
    k1, k2 = torch.meshgrid(k, k, indexing="ij")
    # Spectral derivatives
    ik1, ik2 = 1j * k1 * N, 1j * k2 * N
    h_hat = torch.fft.fft2(h0.to(torch.float64), dim=(1, 2))
    u_hat = torch.zeros_like(h_hat)
    v_hat = torch.zeros_like(h_hat)
    for _ in range(n_steps):
        h_prev, u_prev, v_prev = h_hat.clone(), u_hat.clone(), v_hat.clone()
        # Continuity: dh/dt + d(hu)/dx + d(hv)/dy = 0 (linearized for speed)
        h_hat = h_prev - dt * (ik1 * u_prev + ik2 * v_prev)
        # Momentum
        u_hat = u_prev - dt * (ik1 * SWE_G * h_prev)
        v_hat = v_prev - dt * (ik2 * SWE_G * h_prev)
    return torch.fft.ifft2(h_hat).real.to(torch.float32)


def solve_mhd_2d_batch(w0: torch.Tensor, a0: torch.Tensor, T: float = MHD_T, n_steps: int = MHD_NSTEPS) -> torch.Tensor:
    """Spectral solver for 2D incompressible MHD (vorticity-potential form)."""
    B, N, _ = w0.shape
    dt = T / n_steps
    k = torch.fft.fftfreq(N, device=DEVICE)
    k1, k2 = torch.meshgrid(k, k, indexing="ij")
    lap = -(k1 ** 2 + k2 ** 2); lap[0, 0] = 1.0
    w_hat = torch.fft.fft2(w0.to(torch.float64), dim=(1, 2))
    a_hat = torch.fft.fft2(a0.to(torch.float64), dim=(1, 2))
    for _ in range(n_steps):
        psi_hat = w_hat / lap; psi_hat[:, 0, 0] = 0.0
        u = torch.fft.ifft2(1j * k2 * psi_hat).real
        v = torch.fft.ifft2(-1j * k1 * psi_hat).real
        bx = torch.fft.ifft2(1j * k2 * a_hat).real
        by = torch.fft.ifft2(-1j * k1 * a_hat).real
        aj = torch.fft.ifft2(lap * a_hat).real # Current density J
        
        nonlin_w = torch.fft.fft2(u * torch.fft.ifft2(1j * k1 * w_hat).real + v * torch.fft.ifft2(1j * k2 * w_hat).real - 
                               (bx * torch.fft.ifft2(1j * k1 * aj).real + by * torch.fft.ifft2(1j * k2 * aj).real))
        nonlin_a = torch.fft.fft2(u * torch.fft.ifft2(1j * k1 * a_hat).real + v * torch.fft.ifft2(1j * k2 * a_hat).real)
        
        w_hat = (w_hat - dt * nonlin_w) / (1.0 - dt * MHD_NU * lap)
        a_hat = (a_hat - dt * nonlin_a) / (1.0 - dt * MHD_ETA * lap)
    return torch.fft.ifft2(w_hat).real.to(torch.float32)


def solve_poisson_2d_batch(f: torch.Tensor) -> torch.Tensor:
    """Solve -Δu = f on [0, 1]² with periodic BCs using spectral method."""
    B, N, _ = f.shape
    f_d = f.to(torch.float64)
    f_d -= f_d.mean(dim=(1, 2), keepdim=True)  # ensure zero mean
    
    k_int = torch.fft.fftfreq(N, d=1.0 / N, device=DEVICE)
    kx, ky = torch.meshgrid(2 * math.pi * k_int, 2 * math.pi * k_int, indexing="ij")
    lap_pos = kx ** 2 + ky ** 2
    lap_pos[0, 0] = 1.0  # avoid div by zero for DC mode
    
    f_hat = torch.fft.fft2(f_d, dim=(1, 2))
    u_hat = f_hat / lap_pos[None]
    u_hat[:, 0, 0] = 0.0  # set DC mode to zero
    
    u = torch.fft.ifft2(u_hat, dim=(1, 2)).real
    return u.to(torch.float32)


def solve_reionization_1d_batch(f: torch.Tensor, T: float = 0.5, n_steps: int = 100) -> torch.Tensor:
    """Simple 1D ionization front model."""
    B, N = f.shape
    dt = T / n_steps
    c = 1.0
    alpha = 0.1
    u = f.to(torch.float64)
    dx = 1.0 / N
    for _ in range(n_steps):
        # Upwind advection
        u_shifted = torch.roll(u, 1, dims=1)
        u_x = (u - u_shifted) / dx
        u = u - dt * (c * u_x + alpha * u**2)
    return u.to(torch.float32)


def solve_ellipse_2d_batch(
    params: torch.Tensor, 
    N: int = 64
) -> torch.Tensor:
    """Semi-analytical potential flow solver for an ellipse in 2D."""
    B = params.shape[0]
    x = torch.linspace(-1, 1, N, device=DEVICE)
    y = torch.linspace(-1, 1, N, device=DEVICE)
    yy, xx = torch.meshgrid(y, x, indexing="ij") # meshgrid(y,x) to match np.meshgrid(x,y) with indexing='ij' logic?
    # Wait, np.meshgrid(x,y, indexing='ij') returns [N,N] where xx[i,j] = x[i] and yy[i,j] = y[j]
    # torch.meshgrid(x,y, indexing='ij') returns same.
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    
    # Pressure fields
    p = torch.zeros((B, N, N), dtype=torch.float32, device=DEVICE)
    
    for i in range(B):
        a, b, alpha = params[i]
        # Rotate coordinates
        xr = xx * torch.cos(alpha) + yy * torch.sin(alpha)
        yr = -xx * torch.sin(alpha) + yy * torch.cos(alpha)
        
        # Ellipse SDF
        sdf = torch.sqrt((xr/a)**2 + (yr/b)**2) - 1.0
        
        # Potential flow around ellipse (velocity magnitude approximation)
        v_mag = 1.0 + (a + b) / (a * torch.sin(torch.atan2(yr, xr))**2 + b * torch.cos(torch.atan2(yr, xr))**2 + 1e-6)
        v_mag[sdf < 0] = 0.0 # interior
        
        # Bernoulli pressure: p = 0.5 * rho * (U^2 - v^2)
        p[i] = 0.5 * (1.0 - v_mag**2)
        
    return p


def _ellipse_ic(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Random ellipse parameters [a, b, alpha]."""
    a = rng.uniform(0.2, 0.5, size=(n, 1))
    b = rng.uniform(0.1, 0.3, size=(n, 1))
    alpha = rng.uniform(0, np.pi, size=(n, 1))
    return np.concatenate([a, b, alpha], axis=1).astype(np.float32)


# ── IC generators ─────────────────────────────────────────────────────────────

def _kdv_ic(n: int, N: int, rng: np.random.RandomState) -> np.ndarray:
    """Random smooth ICs for KdV — same Fourier-series generator as Burgers."""
    return _random_ic_np(n, N, rng, n_modes=8)


def _wave_ic(n: int, N: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Random ICs for wave equation: (u0, ∂u/∂t|₀)."""
    u0  = _random_ic_np(n, N, rng, n_modes=8)
    ut0 = np.zeros_like(u0)   # released from rest: u(x,0)=u0(x), u_t(x,0)=0
    return u0, ut0


def _darcy_fix_ic(n: int, N: int, rng: np.random.RandomState
                  ) -> tuple[np.ndarray, np.ndarray]:
    """ICs for corrected Darcy benchmark."""
    # GRF with zero mean and scale 0.5
    z = _random_ic_2d(n, N, rng, n_modes=5, scale=0.5, offset=0.0)
    a = np.exp(z)
    
    # Generate fixed source f (same for all samples in all splits)
    f_rng = np.random.RandomState(12345)
    f_single = _random_ic_2d(1, N, f_rng, n_modes=DARCY_FIX_MODES_F, scale=1.0, offset=0.0)
    f = np.broadcast_to(f_single, (n, N, N))
    
    return a, f


def _ns_fix_ic(n: int, N: int, rng: np.random.RandomState) -> np.ndarray:
    """ICs for corrected NS benchmark: vorticity with small amplitude."""
    return _random_ic_2d(n, N, rng, n_modes=4, scale=NS_SCALE, offset=0.0)


def _swe_ic(n: int, N: int, rng: np.random.RandomState) -> np.ndarray:
    return _random_ic_2d(n, N, rng, n_modes=3, scale=0.1, offset=1.0)


def _allen_cahn_ic(n: int, N: int, rng: np.random.RandomState) -> np.ndarray:
    return _random_ic_2d(n, N, rng, n_modes=8, scale=0.5, offset=0.0)


def _mhd_ic(n: int, N: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    w0 = _random_ic_2d(n, N, rng, n_modes=4, scale=0.1, offset=0.0)
    a0 = _random_ic_2d(n, N, rng, n_modes=4, scale=0.1, offset=0.0)
    return w0, a0


# ── Dataset generation ─────────────────────────────────────────────────────────

def _generate_ext_dataset(benchmark: str, n: int, seed: int) -> tuple:
    rng = np.random.RandomState(seed)
    # To keep the dashboard alive and provide visibility, we generate in chunks
    chunk_size = 100 if "2d" in benchmark else 1000
    all_inputs = []
    all_targets = []
    
    import sys
    
    for i in range(0, n, chunk_size):
        curr_n = min(chunk_size, n - i)
        if i > 0 or n > chunk_size:
            print(f"  [{benchmark}] Generating samples {i}/{n}...", end="\r")
            sys.stdout.flush()
            
        if benchmark == "kdv_1d":
            inp_t = _kdv_ic(curr_n, GRID_SIZE, rng)
            tgt_t = solve_kdv_batch(inp_t, T=KDV_T, n_steps=KDV_NSTEPS)
            inp, tgt = inp_t.cpu().numpy(), tgt_t.cpu().numpy()
        elif benchmark == "wave_1d":
            u0_np, ut0_np = _wave_ic(curr_n, GRID_SIZE, rng)
            u0_t, ut0_t = torch.from_numpy(u0_np).to(DEVICE), torch.from_numpy(ut0_np).to(DEVICE)
            tgt_t = solve_wave_batch(u0_t, ut0_t, c=WAVE_C, T=WAVE_T, n_steps=WAVE_NSTEPS)
            inp, tgt = u0_np, tgt_t.cpu().numpy()
        elif benchmark == "darcy_2d":
            a_np, f_np = _darcy_fix_ic(curr_n, GRID_SIZE, rng)
            a_t, f_t = torch.from_numpy(a_np).to(DEVICE), torch.from_numpy(f_np).to(DEVICE)
            tgt_t = solve_darcy_2d_batch(a_t, f_t)
            inp, tgt = a_np[..., None], tgt_t.cpu().numpy()[..., None]
        elif benchmark == "ns_2d":
            w0_np = _ns_fix_ic(curr_n, GRID_SIZE, rng)
            w0_t = torch.from_numpy(w0_np).to(DEVICE)
            tgt_t = solve_ns_2d_batch(w0_t)
            inp, tgt = w0_np[..., None], tgt_t.cpu().numpy()[..., None]
        elif benchmark == "swe_2d":
            h0_np = _swe_ic(curr_n, GRID_SIZE, rng)
            h0_t = torch.from_numpy(h0_np).to(DEVICE)
            tgt_t = solve_swe_2d_batch(h0_t)
            inp, tgt = h0_np[..., None], tgt_t.cpu().numpy()[..., None]
        elif benchmark == "allen_cahn_2d":
            u0_np = _allen_cahn_ic(curr_n, GRID_SIZE, rng)
            u0_t = torch.from_numpy(u0_np).to(DEVICE)
            tgt_t = solve_allen_cahn_2d_batch(u0_t)
            inp, tgt = u0_np[..., None], tgt_t.cpu().numpy()[..., None]
        elif benchmark == "mhd_2d":
            w0_np, a0_np = _mhd_ic(curr_n, GRID_SIZE, rng)
            w0_t, a0_t = torch.from_numpy(w0_np).to(DEVICE), torch.from_numpy(a0_np).to(DEVICE)
            tgt_t = solve_mhd_2d_batch(w0_t, a0_t)
            inp = np.stack([w0_np, a0_np], axis=-1)
            tgt = tgt_t.cpu().numpy()[..., None]
        elif benchmark == "burgers_nu_01":
            inp_np = _random_ic_np(curr_n, GRID_SIZE, rng)
            from data.prepare import solve_burgers_batch
            inp_t = torch.from_numpy(inp_np).to(DEVICE)
            tgt_t = solve_burgers_batch(inp_t, nu=0.1)
            inp, tgt = inp_np, tgt_t.cpu().numpy()
        elif benchmark == "burgers_nu_001":
            inp_np = _random_ic_np(curr_n, GRID_SIZE, rng)
            from data.prepare import solve_burgers_batch
            inp_t = torch.from_numpy(inp_np).to(DEVICE)
            tgt_t = solve_burgers_batch(inp_t, nu=0.01)
            inp, tgt = inp_np, tgt_t.cpu().numpy()
        elif benchmark == "poisson_2d":
            f_np = _random_ic_2d(curr_n, GRID_SIZE, rng, n_modes=5, scale=1.0)
            f_t = torch.from_numpy(f_np).to(DEVICE)
            tgt_t = solve_poisson_2d_batch(f_t)
            inp, tgt = f_np[..., None], tgt_t.cpu().numpy()[..., None]
        elif benchmark == "reionization_1d":
            f_np = _random_ic_np(curr_n, GRID_SIZE, rng, n_modes=3) * 1.0 + 0.5
            f_t = torch.from_numpy(f_np).to(DEVICE)
            tgt_t = solve_reionization_1d_batch(f_t)
            inp, tgt = f_np, tgt_t.cpu().numpy()
        elif benchmark == "ellipse_2d":
            params_np = _ellipse_ic(curr_n, rng)
            params_t = torch.from_numpy(params_np).to(DEVICE)
            # Input is the SDF of the ellipse
            x = np.linspace(-1, 1, GRID_SIZE)
            y = np.linspace(-1, 1, GRID_SIZE)
            xx, yy = np.meshgrid(x, y)
            inp_list = []
            for j in range(curr_n):
                a, b, alpha = params_np[j]
                xr = xx * np.cos(alpha) + yy * np.sin(alpha)
                yr = -xx * np.sin(alpha) + yy * np.cos(alpha)
                sdf = np.sqrt((xr/a)**2 + (yr/b)**2) - 1.0
                inp_list.append(sdf[..., None])
            inp = np.array(inp_list)
            tgt_t = solve_ellipse_2d_batch(params_t, GRID_SIZE)
            tgt = tgt_t.cpu().numpy()[..., None]
        else:
            raise ValueError(f"Unknown extended benchmark: {benchmark!r}")
            
        all_inputs.append(inp)
        all_targets.append(tgt)

    if n > chunk_size:
        print(f"  [{benchmark}] Generating samples {n}/{n}... Done.")
        sys.stdout.flush()

    return np.concatenate(all_inputs, axis=0), np.concatenate(all_targets, axis=0)


def _get_ext_val_cache(benchmark: str) -> str:
    return os.path.join(CACHE_DIR, f"{benchmark}_val_N{GRID_SIZE}_ext.npz")


def _load_or_gen_ext_val(benchmark: str) -> tuple:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = _get_ext_val_cache(benchmark)
    if os.path.exists(cache):
        data = np.load(cache)
        return data["inputs"], data["targets"]
    print(f"Generating {benchmark} val set ({N_VAL} samples, seed={VAL_SEED})…")
    t0 = time.time()
    inp, tgt = _generate_ext_dataset(benchmark, N_VAL, VAL_SEED)
    np.savez(cache, inputs=inp, targets=tgt)
    print(f"  Cached {N_VAL} samples in {time.time()-t0:.1f}s → {cache}")
    return inp, tgt


def _get_ext_train_cache_path(benchmark: str) -> str:
    return os.path.join(CACHE_DIR, f"{benchmark}_train_N{N_TRAIN}_ext.npz")


_ext_train_cache: dict = {}


def _get_ext_train(benchmark: str) -> tuple:
    if benchmark not in _ext_train_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = _get_ext_train_cache_path(benchmark)
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            _ext_train_cache[benchmark] = (data["inputs"], data["targets"])
        else:
            print(f"Generating {benchmark} train data ({N_TRAIN} samples)…")
            t0 = time.time()
            inputs, targets = _generate_ext_dataset(benchmark, N_TRAIN, TRAIN_SEED)
            np.savez(cache_path, inputs=inputs, targets=targets)
            print(f"  {N_TRAIN} samples in {time.time()-t0:.1f}s → {cache_path}")
            _ext_train_cache[benchmark] = (inputs, targets)
    return _ext_train_cache[benchmark]


from data.prepare import PDEDataset

# ── Public dataloader (same interface as prepare.make_dataloader) ─────────────

def make_ext_dataloader(benchmark: str, split: str, batch_size: int,
                        seed: int | None = None):
    """Yielding (inputs, targets) as PyTorch tensors."""
    assert split in ("train", "val")
    if split == "val":
        inp, tgt = _load_or_gen_ext_val(benchmark)
        dataset = PDEDataset(torch.from_numpy(inp), torch.from_numpy(tgt))
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        inp, tgt = _get_ext_train(benchmark)
        dataset = PDEDataset(torch.from_numpy(inp), torch.from_numpy(tgt))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            generator=torch.Generator().manual_seed(seed if seed is not None else 12345)
        )
        def infinite_loader():
            while True:
                for batch in loader:
                    yield batch
        return infinite_loader()


def evaluate_l2_rel_ext(benchmark: str, model, batch_size: int = 64) -> float:
    """Mean relative L2 error on fixed val set.  Same metric as prepare.py."""
    val_loader = make_ext_dataloader(benchmark, "val", batch_size)
    total_err  = 0.0
    total_norm = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y   = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x)
            diff   = (y_pred - y).float()
            y_f    = y.float()
            axes   = tuple(range(1, y.ndim))
            err    = torch.sqrt(torch.mean(diff ** 2, dim=axes))
            nrm    = torch.sqrt(torch.mean(y_f  ** 2, dim=axes))
            total_err  += torch.sum(err).item()
            total_norm += torch.sum(nrm).item()
    return total_err / max(total_norm, 1e-8)


# ── Extended SOTA targets ──────────────────────────────────────────────────────

EXT_SOTA = {
    "kdv_1d":       0.010,   # FNO on KdV, Tran et al. 2023
    "wave_1d":      0.005,   # Wave equation: easier than Burgers, FNO near-exact
    "darcy_2d": 0.0108,  # Li et al. 2020 FNO on Darcy (proper solver)
    "ns_2d":    0.0128,  # Li et al. 2020 FNO on NS (T=1, nu=1e-2)
    "ns_hre_2d": 0.0700, # Estimated SOTA for Re=1000
    "swe_2d":    0.0020, # FNO on SWE
    "allen_cahn_2d": 0.020, # SOTA near 0.02
    "mhd_2d":    0.0350, # MHD targets from PhysicsNeMo
}


# ── Benchmark metadata ────────────────────────────────────────────────────────

EXT_BENCHMARK_INFO = {
    "kdv_1d": {
        "pde":        "u_t + u·u_x + u_xxx = 0  (Korteweg-de Vries)",
        "domain":     "[0, 2π), periodic",
        "ic_type":    "smooth random Fourier series",
        "solver":     "ETDRK4 (exponential time differencing Runge-Kutta 4)",
        "t_final":    KDV_T,
        "n_steps":    KDV_NSTEPS,
        "sota_model": "FNO",
        "notes":      "Soliton dynamics; FNO handles well due to periodicity",
    },
    "wave_1d": {
        "pde":        "u_tt = c² u_xx  (1D wave, c=1)",
        "domain":     "[0, 2π), periodic",
        "ic_type":    "smooth random Fourier series for u0 and du/dt",
        "solver":     "Störmer-Verlet (symplectic, energy-conserving)",
        "t_final":    WAVE_T,
        "n_steps":    WAVE_NSTEPS,
        "sota_model": "FNO",
        "notes":      "Linear PDE; FNO can achieve near-zero error easily",
    },
    "darcy_2d": {
        "pde":        "-∇·(a(x,y)∇u) = f  (2D Darcy flow)",
        "domain":     "[0, 1]², periodic",
        "ic_type":    "GRF permeability a ∈ [0.6, 1.4]; zero-mean GRF source f",
        "solver":     "Richardson iteration + spectral preconditioner (40 iters)",
        "t_final":    None,
        "n_steps":    DARCY_FIX_N_ITER,
        "sota_model": "FNO",
        "notes":      "Fixed: proper variable-coeff solve; prepare.py used only mean(a)",
        "known_issue_in_prepare_py":
            "solve_darcy_2d_batch uses a_avg (scalar) → u independent of spatial a; "
            "f uses fixed seed=42 → u uncorrelated with model input a",
    },
    "ns_2d": {
        "pde":        "w_t + (u*grad)w = nu * Laplacian(w)  (2D NS, vorticity form)",
        "domain":     "[0, 2π)², periodic",
        "ic_type":    "small-amplitude vorticity (scale=0.1) → CFL≈0.6 < 1",
        "solver":     "Semi-implicit Euler, 2/3-rule dealiasing, n_steps=1000",
        "t_final":    NS_T,
        "n_steps":    NS_NSTEPS,
        "sota_model": "FNO",
        "notes":      "Fixed: IC scale 1.0→0.1 reduces CFL from 61 to ~0.6",
        "known_issue_in_prepare_py":
            "solve_ns_2d_batch uses IC scale=1.0 → max_velocity≈95 → "
            "CFL≈61 → semi-implicit Euler explodes to NaN on step 1",
    },
    "swe_2d": {
        "pde": "Shallow Water Equations (height-vorticity)",
        "domain": "[0, 1]^2, periodic",
        "ic_type": "Random height bumps",
        "solver": "Spectral continuity + momentum",
        "t_final": SWE_T,
        "n_steps": SWE_NSTEPS,
        "sota_model": "MemNO",
        "notes": "Tests multi-scale wave dynamics",
    },
    "allen_cahn_2d": {
        "pde": "Allen-Cahn Phase Separation",
        "domain": "[0, 1]^2, periodic",
        "ic_type": "High-frequency random noise",
        "solver": "Semi-implicit spectral",
        "t_final": AC_T,
        "n_steps": AC_NSTEPS,
        "sota_model": "FNO",
        "notes": "Tests sharp interface capture",
    },
    "mhd_2d": {
        "pde": "Magnetohydrodynamics (vorticity-potential)",
        "domain": "[0, 1]^2, periodic",
        "ic_type": "Orszag-Tang inspired random fields",
        "solver": "Dual-field spectral",
        "t_final": MHD_T,
        "n_steps": MHD_NSTEPS,
        "sota_model": "TFNO",
        "notes": "Coupled fluid-magnetic dynamics",
    },
    "poisson_2d": {
        "pde": "-Δu = f  (2D Poisson equation)",
        "domain": "[0, 1]^2, periodic",
        "ic_type": "Zero-mean random source f",
        "solver": "Exact spectral solver",
        "t_final": None,
        "n_steps": None,
        "sota_model": "IterativeFNO",
        "notes": "Fundamental elliptic PDE; tests precision and convergence stabilities",
    },
    "reionization_1d": {
        "pde": "u_t + c·u_x = S(x) - alpha·u^2 (Cosmic Reionization toy model)",
        "domain": "[0, 1], periodic",
        "ic_type": "Source field S(x)",
        "solver": "Upwind scheme + recombination",
        "t_final": 0.5,
        "n_steps": 100,
        "sota_model": "PINN",
        "notes": "Non-linear advection-reaction; mimics ionization front propagation",
    },
    "ellipse_2d": {
        "pde": "Incompressible laminar flow around ellipse (surface pressure)",
        "domain": "[-1, 1]^2",
        "ic_type": "Random ellipse geometry (a, b, alpha)",
        "solver": "Potential flow analytical approximation",
        "t_final": None,
        "n_steps": None,
        "sota_model": "SAR",
        "notes": "Tests geometry-to-distribution mapping as proposed in Lino & Thuerey (2026)",
    },
}


if __name__ == "__main__":
    import time
    print("Extended benchmarks available:", sorted(EXT_BENCHMARKS))
    for bm in sorted(EXT_BENCHMARKS):
        info = EXT_BENCHMARK_INFO.get(bm, {"pde": "Unknown", "solver": "Unknown"})
        print(f"\n{bm}:")
        print(f"  PDE   : {info['pde']}")
        print(f"  Solver: {info['solver']}")
        print(f"  SOTA  : ~{EXT_SOTA.get(bm, 0.0):.4f} rel-L2")

        # Quick smoke test: generate 4 samples
        t0 = time.time()
        inp, tgt = _generate_ext_dataset(bm, 4, seed=0)
        elapsed = time.time() - t0
        print(f"  Shape : in={inp.shape} → out={tgt.shape}")
        print(f"  Gen   : {elapsed:.2f}s for 4 samples")
        print(f"  NaN?  : in={np.isnan(inp).any()}  out={np.isnan(tgt).any()}")
        if bm in ("darcy_2d",):
            from scipy.stats import pearsonr
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r, _ = pearsonr(inp[0].flatten(), tgt[0].flatten())
            print(f"  corr(a[0], u[0]): {r:.4f}  (should be non-trivial for learnable data)")
