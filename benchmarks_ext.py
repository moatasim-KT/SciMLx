"""Extended benchmark definitions for SciML experiments.

Adds KdV (Korteweg-de Vries) and Wave equation benchmarks on top of the
existing benchmarks in prepare.py.  Uses the same interface as prepare.py
so train.py can import from either module.

Supported benchmarks:
    "kdv_1d"    – Korteweg–de Vries soliton dynamics   (ETDRK4 solver)
    "wave_1d"   – 1D wave equation  u_tt = c² u_xx      (Störmer-Verlet)

Data interface is identical to prepare.py:
    make_ext_dataloader(benchmark, split, batch_size)
    evaluate_l2_rel_ext(benchmark, model)

Both yield (inputs, targets) as MLX float32 arrays with the same
val_l2_rel metric as the main harness.

Usage (in train.py):
    from benchmarks_ext import make_ext_dataloader, evaluate_l2_rel_ext, EXT_BENCHMARKS
    if BENCHMARK in EXT_BENCHMARKS:
        train_loader = make_ext_dataloader(BENCHMARK, "train", BATCH_SIZE)
        val_l2 = evaluate_l2_rel_ext(BENCHMARK, model)

References:
    KdV:  Tran et al. (2023) "Factorized Fourier Neural Operators" — KdV benchmark
    Wave: Rahman et al. (2022) U-NO uses wave equation as additional benchmark
"""

import math
import os
import time

import mlx.core as mx
import numpy as np

from prepare import (
    GRID_SIZE, TIME_BUDGET, N_TRAIN, N_VAL, VAL_SEED, TRAIN_SEED,
    CACHE_DIR,
    solve_kdv_batch, solve_wave_batch, _random_ic,
)

# ── Constants ─────────────────────────────────────────────────────────────────

EXT_BENCHMARKS = {"kdv_1d", "wave_1d"}

# KdV parameters
KDV_T       = 1.0    # final time
KDV_NSTEPS  = 1000   # ETDRK4 steps

# Wave parameters
WAVE_C      = 1.0    # wave speed
WAVE_T      = 1.0    # final time
WAVE_NSTEPS = 400    # Störmer-Verlet steps

# ── IC generators ─────────────────────────────────────────────────────────────

def _kdv_ic(n: int, N: int, rng: np.random.RandomState) -> np.ndarray:
    """Random smooth ICs for KdV — same Fourier-series generator as Burgers."""
    return _random_ic(n, N, rng, n_modes=8)


def _wave_ic(n: int, N: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Random ICs for wave equation: (u0, ∂u/∂t|₀)."""
    u0  = _random_ic(n, N, rng, n_modes=8)
    ut0 = _random_ic(n, N, rng, n_modes=6)
    return u0, ut0


# ── Dataset generation ─────────────────────────────────────────────────────────

def _generate_ext_dataset(benchmark: str, n: int, seed: int) -> tuple:
    rng = np.random.RandomState(seed)
    if benchmark == "kdv_1d":
        inputs  = _kdv_ic(n, GRID_SIZE, rng)
        targets = solve_kdv_batch(inputs, T=KDV_T, n_steps=KDV_NSTEPS)
    elif benchmark == "wave_1d":
        u0, ut0 = _wave_ic(n, GRID_SIZE, rng)
        inputs  = u0
        targets = solve_wave_batch(u0, ut0, c=WAVE_C, T=WAVE_T, n_steps=WAVE_NSTEPS)
    else:
        raise ValueError(f"Unknown extended benchmark: {benchmark!r}")
    return inputs, targets


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


_ext_train_cache: dict = {}


def _get_ext_train(benchmark: str) -> tuple:
    if benchmark not in _ext_train_cache:
        print(f"Generating {benchmark} train data ({N_TRAIN} samples)…")
        t0 = time.time()
        _ext_train_cache[benchmark] = _generate_ext_dataset(benchmark, N_TRAIN, TRAIN_SEED)
        print(f"  {N_TRAIN} samples in {time.time()-t0:.1f}s")
    return _ext_train_cache[benchmark]


# ── Public dataloader (same interface as prepare.make_dataloader) ─────────────

def make_ext_dataloader(benchmark: str, split: str, batch_size: int,
                        seed: int | None = None):
    """Infinite generator yielding (inputs, targets) as MLX arrays.

    Identical interface to prepare.make_dataloader.
    """
    assert split in ("train", "val")
    if split == "val":
        inp, tgt = _load_or_gen_ext_val(benchmark)
        n, i = len(inp), 0
        while True:
            end = min(i + batch_size, n)
            yield mx.array(inp[i:end]), mx.array(tgt[i:end])
            i = end
            if i >= n:
                i = 0
    else:
        inp, tgt = _get_ext_train(benchmark)
        n   = len(inp)
        rng = np.random.RandomState(seed if seed is not None else 12345)
        while True:
            perm = rng.permutation(n)
            for i in range(0, n - batch_size + 1, batch_size):
                idx = perm[i: i + batch_size]
                yield mx.array(inp[idx]), mx.array(tgt[idx])


def evaluate_l2_rel_ext(benchmark: str, model, batch_size: int = 64) -> float:
    """Mean relative L2 error on fixed val set.  Same metric as prepare.py."""
    val_loader = make_ext_dataloader(benchmark, "val", batch_size)
    n_batches  = math.ceil(N_VAL / batch_size)
    total_err  = 0.0
    total_norm = 0.0
    for _ in range(n_batches):
        x, y   = next(val_loader)
        y_pred = model(x)
        diff   = (y_pred - y).astype(mx.float32)
        y_f    = y.astype(mx.float32)
        axes   = tuple(range(1, y.ndim))
        err    = mx.sqrt(mx.mean(diff ** 2, axis=axes))
        nrm    = mx.sqrt(mx.mean(y_f  ** 2, axis=axes))
        mx.eval(err, nrm)
        total_err  += mx.sum(err).item()
        total_norm += mx.sum(nrm).item()
    return total_err / max(total_norm, 1e-8)


# ── Extended SOTA targets ──────────────────────────────────────────────────────

EXT_SOTA = {
    "kdv_1d":  0.010,   # FNO on KdV, Tran et al. 2023
    "wave_1d": 0.005,   # Wave equation: easier than Burgers, FNO near-exact
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
}


if __name__ == "__main__":
    print("Extended benchmarks available:", sorted(EXT_BENCHMARKS))
    for bm in sorted(EXT_BENCHMARKS):
        info = EXT_BENCHMARK_INFO[bm]
        print(f"\n{bm}:")
        print(f"  PDE   : {info['pde']}")
        print(f"  Solver: {info['solver']}")
        print(f"  SOTA  : ~{EXT_SOTA[bm]:.4f} rel-L2")

        # Quick smoke test: generate 4 samples
        t0 = time.time()
        inp, tgt = _generate_ext_dataset(bm, 4, seed=0)
        print(f"  Shape : in={inp.shape} → out={tgt.shape}")
        print(f"  Gen   : {time.time()-t0:.2f}s for 4 samples")
