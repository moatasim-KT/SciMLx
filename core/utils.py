"""Shared utilities for the SciML autoresearch loop.

Centralises path constants, SOTA targets, and results.tsv helpers that were
previously duplicated across analyze.py, auto_suggest.py, autorun.py, and viz.py.
"""

import csv
import math
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT     = Path(__file__).parent.parent
RESULTS_FILE  = REPO_ROOT / "results.json"  # results.tsv logic is now legacy or synced
LOGS_DIR      = REPO_ROOT / "logs"
TELEMETRY_DIR = LOGS_DIR / "telemetry"
SENTINEL_DIR  = LOGS_DIR / "sentinels"
FIGS_DIR      = REPO_ROOT / "figs"
PAPERS_DIR    = REPO_ROOT / "docs" / "papers"

# Ensure directories exist
for d in [LOGS_DIR, TELEMETRY_DIR, SENTINEL_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── SOTA targets ──────────────────────────────────────────────────────────────

SOTA: dict[str, float] = {
    # prepare.py benchmarks
    "burgers_1d":       0.0149,
    "darcy_2d":         0.0108,
    "ns_2d": 0.0128,
    # benchmarks_ext.py
    "kdv_1d":           0.010,
    "wave_1d":          0.005,
    "ns_2d":        0.0128,
    # simulations/ (high-fidelity)
    "euler_1d":         0.015,   # smooth subsonic Euler (multi-channel)
    "swe_2d":           0.002,   # linearized gravity waves (analytic GT)
    "allen_cahn_2d":    0.020,   # phase-field coarsening (Geneva & Zabaras 2022)
    "ns_hre_2d":        0.070,   # Li et al. 2020 Re=1000 (FNO Table 4)
}

# ── results.tsv helpers ───────────────────────────────────────────────────────

def load_results(benchmark: Optional[str] = None) -> list[dict]:
    """Return all rows from results.json, optionally filtered by benchmark.
    
    Each row has val_l2_rel coerced to float (nan on parse failure).
    """
    import json
    rows: list[dict] = []
    if not RESULTS_FILE.exists():
        return rows
    try:
        with open(RESULTS_FILE) as f:
            rows = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {RESULTS_FILE}: {e}")
        return []

    processed_rows = []
    for row in rows:
        try:
            row["val_l2_rel"] = float(row.get("val_l2_rel", float("nan")))
        except (ValueError, TypeError):
            row["val_l2_rel"] = float("nan")
        
        if benchmark and row.get("benchmark") != benchmark:
            continue
        processed_rows.append(row)
    return processed_rows


def best_per_benchmark(rows: list[dict]) -> dict[str, float]:
    """Return lowest val_l2_rel among 'keep' rows, keyed by benchmark."""
    best: dict[str, float] = {}
    for row in rows:
        if row.get("status") == "keep" and not math.isnan(row["val_l2_rel"]):
            bm = row["benchmark"]
            if bm not in best or row["val_l2_rel"] < best[bm]:
                best[bm] = row["val_l2_rel"]
    return best


def done_names() -> set[str]:
    """Return set of experiment names already recorded in results.tsv."""
    names: set[str] = set()
    rows = load_results()
    for row in rows:
        desc = row.get("description", "")
        names.add(desc.split()[0] if desc else "")
    return names
