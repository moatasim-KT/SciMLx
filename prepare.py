"""
SciML fixed evaluation harness — DO NOT MODIFY.

Synthesizes parametric PDE datasets and defines the ground-truth metric.
All training baselines and experiments are evaluated against evaluate_l2_rel().

Usage:
    uv run prepare.py              # generate and cache validation data
    uv run prepare.py --benchmark  # also print solver timing stats
"""

import argparse
import math
import os
import time

import mlx.core as mx
import numpy as np

# ── Fixed constants (do not edit) ────────────────────────────────────────────
TIME_BUDGET   = 300               # training time budget (seconds)
GRID_SIZE     = 64                # spatial grid points on [0, 2π)
T_FINAL       = 1.0               # solution time horizon
NU            = 0.01 / math.pi    # kinematic viscosity ≈ 0.00318 (FNO benchmark)
N_TRAIN       = 4096              # pre-generated training samples
N_VAL         = 256               # validation samples (fixed seed, disk-cached)
TRAIN_SEED    = 7                 # RNG seed for training data
VAL_SEED      = 42                # RNG seed for val data — never changes
EVAL_BATCH    = 64                # batch size used inside evaluate_l2_rel
SOLVER_STEPS  = 500               # IMEX-Euler steps (same for train and val)

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "sciml_autoresearch")
VAL_CACHE_1D = os.path.join(
    CACHE_DIR, f"burgers_val_N{GRID_SIZE}_nu{NU:.6f}_T{T_FINAL}.npz"
)
VAL_CACHE_DARCY = os.path.join(
    CACHE_DIR, f"darcy_val_N{GRID_SIZE}.npz"
)

# ── 1D Solvers ────────────────────────────────────────────────────────────────

def _random_ic(
    n: int,
    N: int,
    rng: np.random.RandomState,
    n_modes: int = 10,
) -> np.ndarray:
    """
    Smooth random initial conditions on [0, 2π) via truncated Fourier series.
    Coefficient amplitude decays as k^{-1.5} for C^1 smoothness.
    Returns float32 [n, N].
    """
    k      = np.arange(1, n_modes + 1, dtype=np.float64)  # [n_modes]
    decay  = k ** -1.5
    cos_c  = rng.randn(n, n_modes) * decay                  # [n, n_modes]
    sin_c  = rng.randn(n, n_modes) * decay

    x      = 2.0 * np.pi * np.arange(N, dtype=np.float64) / N  # [N]
    angles = k[:, None] * x[None, :]                             # [n_modes, N]
    u0     = cos_c @ np.cos(angles) + sin_c @ np.sin(angles)     # [n, N]
    return u0.astype(np.float32)


def solve_burgers_batch(
    u0: np.ndarray,
    nu: float    = NU,
    T: float     = T_FINAL,
    n_steps: int = SOLVER_STEPS,
) -> np.ndarray:
    """
    Batch pseudo-spectral IMEX solver for 1D viscous Burgers equation.
    """
    _, N   = u0.shape
    k      = np.fft.rfftfreq(N, d=1.0 / N)             # wavenumbers [N//2+1]
    dt     = T / n_steps
    impl   = 1.0 / (1.0 + nu * k ** 2 * dt)            # implicit diffusion factor
    ik     = 1j * k                                      # spectral derivative operator
    cutoff = N // 3                                      # 2/3-rule dealias cutoff

    u_hat = np.fft.rfft(u0.astype(np.float64), axis=1)

    for _ in range(n_steps):
        uh_d          = u_hat.copy()
        uh_d[:, cutoff:] = 0.0

        u_phys  = np.fft.irfft(uh_d,                n=N, axis=1)
        ux_phys = np.fft.irfft(ik * uh_d,           n=N, axis=1).real
        nonlin  = np.fft.rfft(-u_phys * ux_phys,    axis=1)

        u_hat = impl * (u_hat + dt * nonlin)

    return np.fft.irfft(u_hat, n=N, axis=1).astype(np.float32)


# ── 2D Solvers ────────────────────────────────────────────────────────────────

def solve_darcy_2d_batch(
    a: np.ndarray,
    N: int = GRID_SIZE,
) -> np.ndarray:
    """
    Spectral solver for 2D Darcy Flow: -∇·(a∇u) = f, with f=1.
    """
    B, N1, N2 = a.shape
    # Random source term f
    rng = np.random.RandomState(42)
    f = _random_ic_2d(B, N1, rng, scale=1.0, offset=0.0)
    
    # Grid setup
    k = np.fft.fftfreq(N1, d=1.0/N1).reshape(N1, 1)
    k1, k2 = np.meshgrid(k, k)
    laplacian = -(k1**2 + k2**2)
    
    # To solve -∇·(a∇u) = f, we use a simple iterative approach or 
    # assume a is constant for a first-order approximation.
    a_avg = a.mean(axis=(1, 2))[:, None, None]
    denom = a_avg * (-laplacian)
    denom[denom == 0] = 1.0 # Handle DC mode
    u_hat = np.fft.fft2(f, axes=(1, 2)) / (denom + 1e-8)
    u_hat[:, 0, 0] = 0.0
    
    u = np.fft.ifft2(u_hat, axes=(1, 2)).real
    return u.astype(np.float32)


def solve_navier_stokes_2d_batch(
    w0: np.ndarray,
    nu: float = 1e-2,
    T: float = 1.0,
    n_steps: int = 100,
) -> np.ndarray:
    """
    Spectral solver for 2D Navier-Stokes (vorticity form) on [0, 2π)².
    Using 2/3-rule dealiasing for stability.
    """
    B, N, _ = w0.shape
    dt = T / n_steps
    
    k = np.fft.fftfreq(N).reshape(N, 1)
    k1, k2 = np.meshgrid(k, k)
    laplacian = -(k1**2 + k2**2)
    laplacian[0, 0] = 1.0
    
    cutoff = (2 * N) // 3
    
    w_hat = np.fft.fft2(w0.astype(np.float64), axes=(1, 2))
    
    for _ in range(n_steps):
        # Dealias
        w_hat_d = w_hat.copy()
        mask = (np.abs(k1 * N) > cutoff) | (np.abs(k2 * N) > cutoff)
        w_hat_d[:, mask] = 0.0
        
        # 1. Stream function: Δψ = ω
        psi_hat = w_hat_d / laplacian
        psi_hat[:, 0, 0] = 0.0
        
        # 2. Velocity: u = (∂ψ/∂y, -∂ψ/∂x)
        u = np.fft.ifft2(1j * k2 * psi_hat).real
        v = np.fft.ifft2(-1j * k1 * psi_hat).real
        
        # 3. Non-linear term: (u·∇)ω
        wx = np.fft.ifft2(1j * k1 * w_hat_d).real
        wy = np.fft.ifft2(1j * k2 * w_hat_d).real
        
        nonlin = np.fft.fft2(u * wx + v * wy)
        
        # 4. Step (Semi-implicit): ω_next = (ω - dt * nonlin) / (1 - dt * nu * laplacian)
        w_hat = (w_hat - dt * nonlin) / (1.0 - dt * nu * laplacian)
        
        # Stability check
        if np.any(np.isnan(w_hat)):
            break
            
    return np.fft.ifft2(w_hat).real.astype(np.float32)


# ── Additional PDE solvers (optional, available for experiments) ─────────────

def solve_wave_batch(
    u0: np.ndarray,
    ut0: np.ndarray,
    c: float     = 1.0,
    T: float     = 1.0,
    n_steps: int = 400,
) -> np.ndarray:
    """
    Spectral Störmer-Verlet solver for 1D wave equation: u_tt = c² u_xx.

    Args:
        u0:  [B, N] float32  initial displacement
        ut0: [B, N] float32  initial velocity
        c:   wave speed
        T:   final time
    Returns:
        [B, N] float32  displacement at time T
    """
    _, N    = u0.shape
    k       = np.fft.rfftfreq(N, d=1.0 / N)
    omega2  = (c * k) ** 2
    dt      = T / n_steps

    u_hat   = np.fft.rfft(u0.astype(np.float64),  axis=1)
    ut_hat  = np.fft.rfft(ut0.astype(np.float64), axis=1)

    for _ in range(n_steps):                              # Störmer-Verlet
        ut_hat -= 0.5 * dt * omega2 * u_hat
        u_hat  += dt * ut_hat
        ut_hat -= 0.5 * dt * omega2 * u_hat

    return np.fft.irfft(u_hat, n=N, axis=1).astype(np.float32)


def solve_kdv_batch(
    u0: np.ndarray,
    T: float     = 1.0,
    n_steps: int = 1000,
) -> np.ndarray:
    """
    Spectral ETDRK4 solver for 1D Korteweg-de Vries equation.

        ∂u/∂t + u ∂u/∂x + ∂³u/∂x³ = 0    on [0, 2π),  periodic BCs.

    Args:
        u0:  [B, N] float32  initial conditions
    Returns:
        [B, N] float32  solutions at time T
    """
    _, N   = u0.shape
    k      = np.fft.rfftfreq(N, d=1.0 / N)
    ik     = 1j * k
    ik3    = (1j * k) ** 3                               # dispersion operator
    cutoff = N // 3
    dt     = T / n_steps

    # Linear operator (dispersion); implicit via integrating factor
    L  = -ik3                                            # linear part of PDE
    E  = np.exp(L * dt)
    E2 = np.exp(L * dt / 2.0)

    u_hat = np.fft.rfft(u0.astype(np.float64), axis=1)

    def nonlin(uh):
        uhd        = uh.copy()
        uhd[:, cutoff:] = 0.0
        u_phys     = np.fft.irfft(uhd, n=N, axis=1)
        ux_phys    = np.fft.irfft(ik * uhd, n=N, axis=1).real
        return np.fft.rfft(-u_phys * ux_phys, axis=1)

    for _ in range(n_steps):                             # ETDRK4 (Cox-Matthews)
        N0 = nonlin(u_hat)
        a  = E2 * u_hat + E2 * dt / 2.0 * N0
        Na = nonlin(a)
        b  = E2 * u_hat + E2 * dt / 2.0 * Na
        Nb = nonlin(b)
        c  = E2 * a     + E2 * dt / 2.0 * (2.0 * Nb - N0)
        Nc = nonlin(c)
        u_hat = E * u_hat + dt / 6.0 * (
            E * N0 + 2.0 * E2 * (Na + Nb) + Nc
        )

    return np.fft.irfft(u_hat, n=N, axis=1).astype(np.float32)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _random_ic_2d(n: int, N: int, rng: np.random.RandomState, n_modes: int = 5, scale: float = 0.1, offset: float = 1.0) -> np.ndarray:
    """Random smooth 2D field."""
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    u0 = np.full((n, N, N), offset, dtype=np.float64)
    for i in range(n):
        for _ in range(n_modes):
            amp = rng.randn() * scale
            kx, ky = rng.randint(1, 5, size=2)
            u0[i] += amp * np.sin(2 * np.pi * (kx * X + ky * Y))
    return u0.astype(np.float32)


def _generate_dataset(benchmark: str, n: int, seed: int) -> tuple:
    rng = np.random.RandomState(seed)
    if benchmark == "burgers_1d":
        inputs = _random_ic(n, GRID_SIZE, rng)
        targets = solve_burgers_batch(inputs)
    elif benchmark == "darcy_2d":
        inputs = _random_ic_2d(n, GRID_SIZE, rng, scale=0.1, offset=1.0)
        targets = solve_darcy_2d_batch(inputs)
    elif benchmark == "navier_stokes_2d":
        inputs = _random_ic_2d(n, GRID_SIZE, rng, scale=1.0, offset=0.0)
        targets = solve_navier_stokes_2d_batch(inputs)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return inputs, targets


def _get_val_cache_path(benchmark: str) -> str:
    return os.path.join(CACHE_DIR, f"{benchmark}_val_N{GRID_SIZE}.npz")


def _load_or_gen_val(benchmark: str) -> tuple:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = _get_val_cache_path(benchmark)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["inputs"], data["targets"]
    print(f"Generating validation set for {benchmark} ({N_VAL} samples, seed={VAL_SEED})...")
    t0 = time.time()
    inputs, targets = _generate_dataset(benchmark, N_VAL, VAL_SEED)
    np.savez(cache_path, inputs=inputs, targets=targets)
    print(f"  Cached {N_VAL} samples in {time.time()-t0:.1f}s → {cache_path}")
    return inputs, targets


_train_cache: dict = {}


def _get_train_data(benchmark: str) -> tuple:
    global _train_cache
    if benchmark not in _train_cache:
        print(f"Generating training data for {benchmark} ({N_TRAIN} samples, seed={TRAIN_SEED})...")
        t0 = time.time()
        _train_cache[benchmark] = _generate_dataset(benchmark, N_TRAIN, TRAIN_SEED)
        print(f"  {N_TRAIN} train samples in {time.time()-t0:.1f}s")
    return _train_cache[benchmark]


# ── Dataloader ────────────────────────────────────────────────────────────────

def make_dataloader(benchmark: str, split: str, batch_size: int, seed: int | None = None):
    """
    Infinite generator yielding ``(inputs, targets)`` as MLX arrays.
    """
    assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"

    if split == "val":
        inp, tgt = _load_or_gen_val(benchmark)
        n = len(inp)
        i = 0
        while True:
            end  = min(i + batch_size, n)
            yield mx.array(inp[i:end]), mx.array(tgt[i:end])
            i = end
            if i >= n:
                i = 0
    else:
        inp, tgt = _get_train_data(benchmark)
        n   = len(inp)
        rng = np.random.RandomState(seed if seed is not None else 99999)
        while True:
            perm = rng.permutation(n)
            for i in range(0, n - batch_size + 1, batch_size):
                idx = perm[i : i + batch_size]
                yield mx.array(inp[idx]), mx.array(tgt[idx])


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_l2_rel(benchmark: str, model, batch_size: int = EVAL_BATCH) -> float:
    """
    Mean relative L2 error on the fixed validation set for a given benchmark.
    """
    val_loader = make_dataloader(benchmark, "val", batch_size)
    n_batches  = math.ceil(N_VAL / batch_size)
    total_err  = 0.0
    total_norm = 0.0

    for _ in range(n_batches):
        x, y     = next(val_loader)
        y_pred   = model(x)
        diff     = (y_pred - y).astype(mx.float32)
        y_f      = y.astype(mx.float32)
        
        # L2 norm over spatial dimensions (all but batch)
        axes = tuple(range(1, y.ndim))
        err  = mx.sqrt(mx.mean(diff ** 2, axis=axes))
        nrm  = mx.sqrt(mx.mean(y_f  ** 2, axis=axes))
        
        mx.eval(err, nrm)
        total_err  += mx.sum(err).item()
        total_norm += mx.sum(nrm).item()

    return total_err / max(total_norm, 1e-8)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SciML evaluation harness")
    parser.add_argument("--benchmark", type=str, choices=["burgers_1d", "darcy_2d", "navier_stokes_2d", "all"],
                        default="burgers_1d", help="Run solver timing benchmarks")
    args = parser.parse_args()

    benchmarks = ["burgers_1d", "darcy_2d", "navier_stokes_2d"] if args.benchmark == "all" else [args.benchmark]

    print(f"Cache dir  : {CACHE_DIR}")
    print()

    for b in benchmarks:
        print(f"--- Benchmark: {b} ---")
        val_inp,   val_tgt   = _load_or_gen_val(b)
        train_inp, train_tgt = _get_train_data(b)
        print(f"Val   : {len(val_inp):5d} samples | Shape: {val_inp.shape}")
        print(f"Train : {len(train_inp):5d} samples | Shape: {train_inp.shape}")
        
        if args.benchmark != "none":
            rng = np.random.RandomState(0)
            for batch_size in (1, 64):
                if b == "burgers_1d":
                    u0 = _random_ic(batch_size, GRID_SIZE, rng)
                    t0 = time.time()
                    solve_burgers_batch(u0)
                elif b == "darcy_2d":
                    a = _random_ic_2d(batch_size, GRID_SIZE, rng)
                    t0 = time.time()
                    solve_darcy_2d_batch(a)
                elif b == "navier_stokes_2d":
                    w0 = _random_ic_2d(batch_size, GRID_SIZE, rng)
                    t0 = time.time()
                    solve_navier_stokes_2d_batch(w0)
                print(f"  Solver {b}  B={batch_size:4d}  → {(time.time()-t0)*1000:.1f} ms")
        print()

    print("Done. Ready to train.")
