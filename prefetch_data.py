"""Pre-generate and disk-cache all PDE training datasets.

Run once before starting experiments:
    uv run prefetch_data.py

This eliminates per-experiment data generation overhead. Each benchmark
is generated in a separate thread. Total time: dominated by ns_hre_2d
(10–20 min). All others finish in under 5 min combined.

After this script completes, every train.py subprocess will load data
from disk in < 5s instead of regenerating from scratch.
"""

import os
import sys
import time
import threading
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Locate cache dir (mirrors prepare.py) ─────────────────────────────────────
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "sciml_autoresearch")
os.makedirs(CACHE_DIR, exist_ok=True)

_print_lock = threading.Lock()

def log(bm: str, msg: str):
    with _print_lock:
        print(f"  [{bm}] {msg}", flush=True)


# ── Per-source prefetch functions ─────────────────────────────────────────────

def prefetch_ext(benchmark: str) -> str:
    """Cache train+val for benchmarks_ext benchmarks (kdv, wave, darcy, ns_2d_fix)."""
    from benchmarks_ext import (
        _get_ext_train, _load_or_gen_ext_val,
        _get_ext_val_cache, _get_ext_train_cache_path, N_TRAIN,
    )
    results = []

    # Val
    val_path = _get_ext_val_cache(benchmark)
    if os.path.exists(val_path):
        log(benchmark, f"val already cached ({Path(val_path).stat().st_size // 1024}KB)")
    else:
        log(benchmark, "generating val set…")
        t0 = time.time()
        _load_or_gen_ext_val(benchmark)
        log(benchmark, f"val cached in {time.time()-t0:.0f}s → {Path(val_path).name}")
    results.append("val")

    # Train
    train_path = _get_ext_train_cache_path(benchmark)
    if os.path.exists(train_path):
        log(benchmark, f"train already cached ({Path(train_path).stat().st_size // 1024 // 1024}MB)")
    else:
        log(benchmark, f"generating train set ({N_TRAIN} samples)…")
        t0 = time.time()
        _get_ext_train(benchmark)
        log(benchmark, f"train cached in {time.time()-t0:.0f}s → {Path(train_path).name}")
    results.append("train")

    return f"{benchmark}: {', '.join(results)} ✓"


def prefetch_sim(benchmark: str) -> str:
    """Cache train+val for simulation benchmarks (euler_1d, swe_2d, allen_cahn_2d, ns_hre_2d)."""
    from simulations import _load_or_generate, _cache_path

    results = []
    for split in ("val", "train"):
        path = _cache_path(benchmark, split)
        if os.path.exists(path):
            size_mb = Path(path).stat().st_size // 1024 // 1024
            log(benchmark, f"{split} already cached ({size_mb}MB)")
        else:
            log(benchmark, f"generating {split} set…")
            t0 = time.time()
            _load_or_generate(benchmark, split)
            log(benchmark, f"{split} cached in {time.time()-t0:.0f}s → {Path(path).name}")
        results.append(split)

    return f"{benchmark}: {', '.join(results)} ✓"


def prefetch_burgers() -> str:
    """Cache burgers_1d train+val to disk (prepare.py only does in-memory caching)."""
    from prepare import (
        _generate_dataset, _load_or_gen_val, N_TRAIN, N_VAL, TRAIN_SEED, CACHE_DIR, GRID_SIZE
    )

    cache_train = os.path.join(CACHE_DIR, f"burgers_1d_train_N{GRID_SIZE}.npz")

    # Val (prepare.py already caches this, just warm it)
    log("burgers_1d", "warming val cache…")
    _load_or_gen_val("burgers_1d")
    log("burgers_1d", "val ready")

    # Train
    if os.path.exists(cache_train):
        size_mb = Path(cache_train).stat().st_size // 1024 // 1024
        log("burgers_1d", f"train already cached ({size_mb}MB)")
    else:
        log("burgers_1d", f"generating train set ({N_TRAIN} samples)…")
        t0 = time.time()
        inputs, targets = _generate_dataset("burgers_1d", N_TRAIN, TRAIN_SEED)
        np.savez(cache_train, inputs=inputs, targets=targets)
        log("burgers_1d", f"train cached in {time.time()-t0:.0f}s → {Path(cache_train).name}")

    return "burgers_1d: val, train ✓"




# ── Main ──────────────────────────────────────────────────────────────────────

TASKS = [
    # (label, fn, args)
    ("burgers_1d",    prefetch_burgers,        ()),
    ("kdv_1d",        prefetch_ext,            ("kdv_1d",)),
    ("wave_1d",       prefetch_ext,            ("wave_1d",)),
    ("darcy_2d",  prefetch_ext,            ("darcy_2d",)),
    ("ns_2d_fix",     prefetch_ext,            ("ns_2d_fix",)),
    ("euler_1d",      prefetch_sim,            ("euler_1d",)),
    ("swe_2d",        prefetch_sim,            ("swe_2d",)),
    ("allen_cahn_2d", prefetch_sim,            ("allen_cahn_2d",)),
    ("ns_hre_2d",     prefetch_sim,            ("ns_hre_2d",)),
]

# ns_hre_2d takes 10-20 min and is CPU-bound; run it alone so it gets full CPU
SERIAL_TASKS  = {"ns_hre_2d"}
PARALLEL_TASKS = [t for t in TASKS if t[0] not in SERIAL_TASKS]
SLOW_TASKS     = [t for t in TASKS if t[0] in SERIAL_TASKS]

if __name__ == "__main__":
    skip_slow = "--skip-slow" in sys.argv
    print(f"\nPrefetching all PDE datasets → {CACHE_DIR}")
    print(f"Parallel: {[t[0] for t in PARALLEL_TASKS]}")
    if skip_slow:
        print(f"Skipping slow: {[t[0] for t in SLOW_TASKS]} (pass without --skip-slow to include)")
    else:
        print(f"Serial (slow): {[t[0] for t in SLOW_TASKS]}")
    print()

    t_total = time.time()
    results = []

    # Run fast benchmarks in parallel
    with ThreadPoolExecutor(max_workers=len(PARALLEL_TASKS)) as pool:
        futures = {pool.submit(fn, *args): label for label, fn, args in PARALLEL_TASKS}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                label = futures[fut]
                results.append(f"{label}: ERROR — {e}")

    # Run slow benchmarks serially (give them full CPU)
    if not skip_slow:
        for label, fn, args in SLOW_TASKS:
            try:
                results.append(fn(*args))
            except Exception as e:
                results.append(f"{label}: ERROR — {e}")

    print(f"\n{'━'*60}")
    print(f"Done in {time.time()-t_total:.0f}s\n")
    for r in results:
        print(f"  {r}")
    print(f"\nAll datasets cached. Experiments will now load data in <5s.")
