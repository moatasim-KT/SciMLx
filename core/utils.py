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
    # 1D benchmarks
    "burgers_1d":       0.0031,  # GNOT (Hao et al. 2023) — tightest published
    "kdv_1d":           0.010,   # RFNO (estimated)
    "wave_1d":          0.005,   # FNO (estimated)
    "euler_1d":         0.003,   # estimated for smooth subsonic multi-channel
    "burgers_nu_001":   0.080,   # estimated for near-inviscid shocks
    # 2D benchmarks
    "darcy_2d":         0.0041,  # GNOT (Hao et al. 2023) — tightest published
    "ns_2d":            0.0128,  # Li et al. 2021 FNO Table 1
    "ns_hre_2d":        0.050,   # estimated Re=1000 target
    "swe_2d":           0.015,   # FNO baseline (estimated)
    "allen_cahn_2d":    0.080,   # estimated phase-field target
    "elasticity_2d":    0.010,   # estimated linear elasticity target
    "wavebench_2d":     0.015,   # WaveBench baseline (estimated)
    "pdebench_2d":      0.005,   # PDEBench baseline (estimated)
    "mhd_2d":           0.050,   # estimated MHD target
    "multiphysics_2d":  0.200,   # estimated multi-physics baseline
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


# ── DuckDB-powered fast queries ───────────────────────────────────────────────

def query_results(sql: str) -> list[dict]:
    """Run an arbitrary SQL query against results.json via DuckDB.

    The table is exposed as `results`.  Example:
        query_results("SELECT benchmark, MIN(val_l2_rel) FROM results GROUP BY benchmark")

    Falls back gracefully if duckdb is not installed (returns empty list with a warning).
    """
    if not RESULTS_FILE.exists():
        return []
    try:
        import duckdb
        con = duckdb.connect()
        con.execute(f"CREATE VIEW results AS SELECT * FROM read_json_auto('{RESULTS_FILE}')")
        rows = con.execute(sql).fetchall()
        cols = [d[0] for d in con.description]
        return [dict(zip(cols, row)) for row in rows]
    except ImportError:
        print("[utils] duckdb not installed — falling back to load_results()")
        return []
    except Exception as e:
        print(f"[utils] DuckDB query failed: {e}")
        return []


def best_per_benchmark_sql() -> dict[str, float]:
    """DuckDB-powered replacement for best_per_benchmark(load_results()).

    ~10-100x faster than the Python loop on large results.json files.
    Falls back to the pure-Python path if duckdb is unavailable.
    """
    rows = query_results(
        "SELECT benchmark, MIN(CAST(val_l2_rel AS DOUBLE)) AS best "
        "FROM results WHERE status = 'keep' GROUP BY benchmark"
    )
    if rows:
        return {r["benchmark"]: float(r["best"]) for r in rows if r["best"] is not None}
    # fallback
    return best_per_benchmark(load_results())
