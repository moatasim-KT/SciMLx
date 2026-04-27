"""High-fidelity simulation benchmarks for SciML operator learning.

Provides four new benchmarks that extend benchmarks_ext.py:

    euler_1d        Compressible Euler 1D   HLL+MUSCL+SSP-RK2  [B,N,3]→[B,N,3]
    swe_2d          2D Shallow Water        analytic spectral   [B,N,N]→[B,N,N]
    allen_cahn_2d   Allen-Cahn phase field  ETDRK2 spectral     [B,N,N]→[B,N,N]
    ns_hre_2d       NS 2D high-Re           ETDRK4 spectral     [B,N,N]→[B,N,N]

Dataset interface (identical to benchmarks_ext):
    make_sim_dataloader(benchmark, split, batch_size)  → infinite (x, y) generator
    evaluate_l2_rel_sim(benchmark, model)              → float (mean rel-L2)

All datasets are disk-cached under CACHE_DIR. First generation of ns_hre_2d
training data (~4096 samples) takes 10–20 minutes; subsequent loads take < 5s.
"""

import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from core.device import DEVICE, FRAMEWORK, to_array

if FRAMEWORK == "mlx":
    import mlx.core as mx

from data.prepare import GRID_SIZE, N_TRAIN, N_VAL, VAL_SEED, TRAIN_SEED, CACHE_DIR

from data.simulations import euler1d
from data.simulations import shallow_water
from data.simulations import allen_cahn
from data.simulations import ns_etdrk4
from data.simulations import wavebench
from data.simulations import multiphysics
from data.simulations import pdebench
from data.simulations import elasticity
from data.simulations import radiative

# ── Registry ──────────────────────────────────────────────────────────────────

SIM_BENCHMARKS: set[str] = {"euler_1d", "swe_2d", "allen_cahn_2d", "ns_hre_2d", "wavebench_2d", "multiphysics_2d", "pdebench_2d", "elasticity_2d", "radiative_2d"}

SIM_SOTA: dict[str, float] = {
    "euler_1d":      0.015,   # smooth subsonic Euler; comparable to Burgers
    "swe_2d":        0.002,   # linear dispersive waves; FNO near-exact
    "allen_cahn_2d": 0.020,   # phase-field coarsening; Geneva & Zabaras 2022
    "ns_hre_2d":     0.070,   # Li et al. 2020, Re=1000 (FNO Table 4)
    "wavebench_2d":  0.010,
    "multiphysics_2d": 0.020,
    "pdebench_2d":   0.030,
    "elasticity_2d": 0.040,
    "radiative_2d":  0.050,
}

# Metadata exposed for documentation / paper_registry
SIM_METADATA: dict[str, dict] = {
    "euler_1d":      euler1d.METADATA,
    "swe_2d":        shallow_water.METADATA,
    "allen_cahn_2d": allen_cahn.METADATA,
    "ns_hre_2d":     ns_etdrk4.METADATA,
    "wavebench_2d":  wavebench.METADATA,
    "multiphysics_2d": multiphysics.METADATA,
    "pdebench_2d":   pdebench.METADATA,
    "elasticity_2d": elasticity.METADATA,
    "radiative_2d":  radiative.METADATA,
}

# Whether the benchmark has multi-channel inputs/outputs
SIM_IS_MC: dict[str, bool] = {
    "euler_1d":      True,   # [B, N, 3]
    "swe_2d":        False,  # [B, N, N]
    "allen_cahn_2d": False,  # [B, N, N]
    "ns_hre_2d":     False,  # [B, N, N]
    "wavebench_2d":  False,  # [B, N, N]
    "multiphysics_2d": True,   # [B, N, N, 2]
    "pdebench_2d":   False,  # [B, N, N]
    "elasticity_2d": True,   # [B, N, N, 2]
    "radiative_2d":  False,  # [B, N, N]
}

SIM_N_CHANNELS: dict[str, int] = {
    "euler_1d":      3,
    "swe_2d":        1,
    "allen_cahn_2d": 1,
    "ns_hre_2d":     1,
    "wavebench_2d":  1,
    "multiphysics_2d": 2,
    "pdebench_2d":   1,
    "elasticity_2d": 2,
    "radiative_2d":  1,
}


# ── Dataset generation dispatch ───────────────────────────────────────────────

def _generate_sim_dataset(benchmark: str, n: int, seed: int) -> tuple:
    """Generate (inputs, targets) for a given benchmark."""
    N = GRID_SIZE
    if benchmark == "euler_1d":
        return euler1d.make_dataset(n, seed, N)
    if benchmark == "swe_2d":
        return shallow_water.make_dataset(n, seed, N)
    if benchmark == "allen_cahn_2d":
        return allen_cahn.make_dataset(n, seed, N)
    if benchmark == "ns_hre_2d":
        return ns_etdrk4.make_dataset(n, seed, N)
    if benchmark == "wavebench_2d":
        return wavebench.make_dataset(n, seed, N)
    if benchmark == "multiphysics_2d":
        return multiphysics.make_dataset(n, seed, N)
    if benchmark == "pdebench_2d":
        return pdebench.make_dataset(n, seed, N)
    if benchmark == "elasticity_2d":
        return elasticity.make_dataset(n, seed, N)
    if benchmark == "radiative_2d":
        return radiative.make_dataset(n, seed, N)
    raise ValueError(f"Unknown sim benchmark: {benchmark!r}")


# ── Disk cache ────────────────────────────────────────────────────────────────

def _cache_path(benchmark: str, split: str) -> str:
    seed = VAL_SEED if split == "val" else TRAIN_SEED
    n    = N_VAL    if split == "val" else N_TRAIN
    meta = SIM_METADATA[benchmark]
    tag  = f"{benchmark}_{split}_N{GRID_SIZE}_s{meta['n_steps']}_seed{seed}"
    return os.path.join(CACHE_DIR, f"{tag}.npz")


def _load_or_generate(benchmark: str, split: str) -> tuple:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = _cache_path(benchmark, split)
    if os.path.exists(cache):
        data = np.load(cache)
        return data["inputs"], data["targets"]

    seed = VAL_SEED if split == "val" else TRAIN_SEED
    n    = N_VAL    if split == "val" else N_TRAIN
    print(f"Generating {benchmark} {split} set ({n} samples, seed={seed})…")
    if split == "train" and benchmark == "ns_hre_2d":
        print(f"  [ns_hre_2d train] First-run generation may take 10–20 min.")
    t0       = time.time()
    inp, tgt = _generate_sim_dataset(benchmark, n, seed)
    np.savez(cache, inputs=inp, targets=tgt)
    print(f"  Cached {n} samples in {time.time()-t0:.1f}s → {cache}")
    return inp, tgt


_sim_train_cache: dict = {}


def _get_sim_train(benchmark: str) -> tuple:
    if benchmark not in _sim_train_cache:
        _sim_train_cache[benchmark] = _load_or_generate(benchmark, "train")
    return _sim_train_cache[benchmark]


# ── Public dataloader (same interface as prepare.make_dataloader) ─────────────

def make_sim_dataloader(benchmark: str, split: str, batch_size: int,
                        seed: int | None = None):
    """Infinite (inputs, targets) generator yielding framework-native arrays.

    Interface identical to prepare.make_dataloader and benchmarks_ext.make_ext_dataloader.
    """
    assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"

    if split == "val":
        inp, tgt = _load_or_generate(benchmark, "val")
        n, i = len(inp), 0
        while True:
            end = min(i + batch_size, n)
            yield to_array(inp[i:end]), to_array(tgt[i:end])
            i = end
            if i >= n:
                i = 0
    else:
        inp, tgt = _get_sim_train(benchmark)
        n   = len(inp)
        rng = np.random.RandomState(seed if seed is not None else 54321)
        while True:
            perm = rng.permutation(n)
            for i in range(0, n - batch_size + 1, batch_size):
                idx = perm[i: i + batch_size]
                yield to_array(inp[idx]), to_array(tgt[idx])


# ── Evaluator (same interface as benchmarks_ext.evaluate_l2_rel_ext) ─────────

def evaluate_l2_rel_sim(benchmark: str, model, batch_size: int = 64) -> float:
    """Mean relative L2 error on the fixed validation set.

    Works for both scalar [B, N, N] and multi-channel [B, N, C] outputs.
    """
    val_loader = make_sim_dataloader(benchmark, "val", batch_size)
    n_batches  = math.ceil(N_VAL / batch_size)
    total_err  = 0.0
    total_norm = 0.0

    if FRAMEWORK == "mlx":
        for _ in range(n_batches):
            x, y   = next(val_loader)
            y_pred = model(x)
            diff   = (y_pred - y).astype(mx.float32)
            y_f    = y.astype(mx.float32)
            axes   = tuple(range(1, y.ndim))    # all spatial+channel dims
            err    = mx.sqrt(mx.mean(diff**2, axis=axes))
            nrm    = mx.sqrt(mx.mean(y_f **2, axis=axes))
            mx.eval(err, nrm)
            total_err  += mx.sum(err).item()
            total_norm += mx.sum(nrm).item()
    else:
        with torch.no_grad():
            for _ in range(n_batches):
                x, y   = next(val_loader)
                # x and y are already moved to DEVICE by to_array in loader
                y_pred = model(x)
                diff   = (y_pred - y).float()
                y_f    = y.float()
                axes   = tuple(range(1, y.ndim))
                err    = torch.sqrt(torch.mean(diff**2, dim=axes))
                nrm    = torch.sqrt(torch.mean(y_f **2, dim=axes))
                total_err  += torch.sum(err).item()
                total_norm += torch.sum(nrm).item()

    return total_err / max(total_norm, 1e-8)


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SciML simulation benchmarks available:")
    for bm in sorted(SIM_BENCHMARKS):
        meta = SIM_METADATA[bm]
        mc   = "multi-channel" if SIM_IS_MC[bm] else "scalar"
        print(f"\n  {bm}  [{mc}]")
        print(f"    PDE   : {meta['pde']}")
        print(f"    Solver: {meta['solver']}")
        print(f"    T={meta['t_final']}, steps={meta['n_steps']}")
        print(f"    SOTA  : ~{SIM_SOTA[bm]:.4f} rel-L2")

        t0       = time.time()
        inp, tgt = _generate_sim_dataset(bm, 4, seed=99)
        elapsed  = time.time() - t0
        print(f"    Shape : in={inp.shape} → out={tgt.shape}")
        print(f"    Gen   : {elapsed:.2f}s for 4 samples")
        nan_in   = bool(np.isnan(inp).any())
        nan_out  = bool(np.isnan(tgt).any())
        print(f"    NaN?  : in={nan_in}  out={nan_out}")
