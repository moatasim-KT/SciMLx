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

import torch
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

from core.device import DEVICE

# ── 1D Solvers ────────────────────────────────────────────────────────────────

def _random_ic_np(
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

def _random_ic(
    n: int,
    N: int,
    rng: np.random.RandomState,
    n_modes: int = 10,
) -> torch.Tensor:
    u0 = _random_ic_np(n, N, rng, n_modes)
    return torch.from_numpy(u0).to(DEVICE)


def solve_burgers_batch(
    u0: torch.Tensor,
    nu: float    = NU,
    T: float     = T_FINAL,
    n_steps: int = SOLVER_STEPS,
) -> torch.Tensor:
    """
    Batch pseudo-spectral IMEX solver for 1D viscous Burgers equation.
    """
    _, N   = u0.shape
    # k      = np.fft.rfftfreq(N, d=1.0 / N)             # wavenumbers [N//2+1]
    k      = torch.fft.rfftfreq(N, d=1.0 / N, device=DEVICE)
    dt     = T / n_steps
    impl   = 1.0 / (1.0 + nu * k ** 2 * dt)            # implicit diffusion factor
    ik     = 1j * k                                      # spectral derivative operator
    cutoff = N // 3                                      # 2/3-rule dealias cutoff

    u_hat = torch.fft.rfft(u0.to(torch.float64), dim=1)

    for _ in range(n_steps):
        uh_d          = u_hat.clone()
        uh_d[:, cutoff:] = 0.0

        u_phys  = torch.fft.irfft(uh_d,                n=N, dim=1)
        ux_phys = torch.fft.irfft(ik * uh_d,           n=N, dim=1).real
        nonlin  = torch.fft.rfft(-u_phys * ux_phys,    dim=1)

        u_hat = impl * (u_hat + dt * nonlin)

    return torch.fft.irfft(u_hat, n=N, dim=1).to(torch.float32)


# ── 2D Solvers ────────────────────────────────────────────────────────────────






# ── Additional PDE solvers (optional, available for experiments) ─────────────

def solve_wave_batch(
    u0: torch.Tensor,
    ut0: torch.Tensor,
    c: float     = 1.0,
    T: float     = 1.0,
    n_steps: int = 400,
) -> torch.Tensor:
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
    k       = torch.fft.rfftfreq(N, d=1.0 / N, device=DEVICE)
    omega2  = (c * k) ** 2
    dt      = T / n_steps

    u_hat   = torch.fft.rfft(u0.to(torch.float64),  dim=1)
    ut_hat  = torch.fft.rfft(ut0.to(torch.float64), dim=1)

    for _ in range(n_steps):                              # Störmer-Verlet
        ut_hat -= 0.5 * dt * omega2 * u_hat
        u_hat  += dt * ut_hat
        ut_hat -= 0.5 * dt * omega2 * u_hat

    return torch.fft.irfft(u_hat, n=N, dim=1).to(torch.float32)


def solve_kdv_batch(
    u0: torch.Tensor,
    T: float     = 1.0,
    n_steps: int = 1000,
) -> torch.Tensor:
    """
    Spectral ETDRK4 solver for 1D Korteweg-de Vries equation.

        ∂u/∂t + u ∂u/∂x + ∂³u/∂x³ = 0    on [0, 2π),  periodic BCs.

    Args:
        u0:  [B, N] float32  initial conditions
    Returns:
        [B, N] float32  solutions at time T
    """
    _, N   = u0.shape
    k      = torch.fft.rfftfreq(N, d=1.0 / N, device=DEVICE)
    ik     = 1j * k
    ik3    = (1j * k) ** 3                               # dispersion operator
    cutoff = N // 3
    dt     = T / n_steps

    # Linear operator (dispersion); implicit via integrating factor
    L  = -ik3                                            # linear part of PDE
    E  = torch.exp(L * dt)
    E2 = torch.exp(L * dt / 2.0)

    u_hat = torch.fft.rfft(u0.to(torch.float64), dim=1)

    def nonlin(uh):
        uhd        = uh.clone()
        uhd[:, cutoff:] = 0.0
        u_phys     = torch.fft.irfft(uhd, n=N, dim=1)
        ux_phys    = torch.fft.irfft(ik * uhd, n=N, dim=1).real
        return torch.fft.rfft(-u_phys * ux_phys, dim=1)

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

    return torch.fft.irfft(u_hat, n=N, dim=1).to(torch.float32)


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
        inputs_t = _random_ic(n, GRID_SIZE, rng)
        targets_t = solve_burgers_batch(inputs_t)
        inputs = inputs_t.cpu().numpy()
        targets = targets_t.cpu().numpy()
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

class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def make_dataloader(benchmark: str, split: str, batch_size: int, seed: int | None = None):
    """
    Yielding ``(inputs, targets)`` as PyTorch tensors.
    """
    assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"

    if split == "val":
        inp, tgt = _load_or_gen_val(benchmark)
        dataset = PDEDataset(torch.from_numpy(inp), torch.from_numpy(tgt))
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        inp, tgt = _get_train_data(benchmark)
        dataset = PDEDataset(torch.from_numpy(inp), torch.from_numpy(tgt))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            generator=torch.Generator().manual_seed(seed if seed is not None else 99999)
        )
        def infinite_loader():
            while True:
                for batch in loader:
                    yield batch
        return infinite_loader()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_l2_rel(benchmark: str, model, batch_size: int = EVAL_BATCH) -> float:
    """
    Mean relative L2 error on the fixed validation set for a given benchmark.
    """
    val_loader = make_dataloader(benchmark, "val", batch_size)
    total_err  = 0.0
    total_norm = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y     = x.to(DEVICE), y.to(DEVICE)
            y_pred   = model(x)
            diff     = (y_pred - y).float()
            y_f      = y.float()
            
            # L2 norm over spatial dimensions (all but batch)
            axes = tuple(range(1, y.ndim))
            err  = torch.sqrt(torch.mean(diff ** 2, dim=axes))
            nrm  = torch.sqrt(torch.mean(y_f  ** 2, dim=axes))
            
            total_err  += torch.sum(err).item()
            total_norm += torch.sum(nrm).item()

    return total_err / max(total_norm, 1e-8)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SciML evaluation harness")
    parser.add_argument("--benchmark", type=str, choices=["burgers_1d", "ns_2d", "all"],
                        default="burgers_1d", help="Run solver timing benchmarks")
    args = parser.parse_args()

    benchmarks = ["burgers_1d", "ns_2d"] if args.benchmark == "all" else [args.benchmark]

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
                elif b == "ns_2d":
                    w0 = _random_ic_2d(batch_size, GRID_SIZE, rng)
                    t0 = time.time()
                    from data.benchmarks_ext import solve_ns_2d_batch
                    solve_ns_2d_batch(w0)
                print(f"  Solver {b}  B={batch_size:4d}  → {(time.time()-t0)*1000:.1f} ms")
        print()

    print("Done. Ready to train.")
