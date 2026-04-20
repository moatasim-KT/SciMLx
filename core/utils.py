"""Shared utilities for the SciML autoresearch loop.

Centralises path constants, SOTA targets, and results.tsv helpers that were
previously duplicated across analyze.py, auto_suggest.py, autorun.py, and viz.py.
"""

import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Optional

from filelock import FileLock

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

# Cross-process file lock for results.json — prevents concurrent autorun corruption
RESULTS_LOCK = FileLock(str(RESULTS_FILE) + ".lock", timeout=30)

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

    Acquires the cross-process lock so reads are never concurrent with writes.
    Deduplicates by id (self-healing after any historical race corruption).
    Each row has val_l2_rel coerced to float (nan on parse failure).
    """
    import json
    if not RESULTS_FILE.exists():
        return []
    try:
        with RESULTS_LOCK:
            rows: list[dict] = json.loads(RESULTS_FILE.read_text())
    except Exception as e:
        print(f"Warning: Could not load {RESULTS_FILE}: {e}")
        return []

    # Deduplicate by id — self-heals any entries doubled by past races
    seen: set = set()
    deduped: list[dict] = []
    for row in rows:
        rid = row.get("id")
        if rid not in seen:
            seen.add(rid)
            deduped.append(row)

    processed: list[dict] = []
    for row in deduped:
        try:
            row["val_l2_rel"] = float(row.get("val_l2_rel", float("nan")))
        except (ValueError, TypeError):
            row["val_l2_rel"] = float("nan")
        if benchmark and row.get("benchmark") != benchmark:
            continue
        processed.append(row)
    return processed


def append_result(row: dict) -> None:
    """Safely append one result row to results.json.

    Guarantees:
    - Exclusive cross-process lock (no concurrent writer can interleave)
    - Re-reads from disk under lock (never writes stale in-memory state)
    - Deduplicates by id (idempotent — safe to call twice for same result)
    - Atomic rename: a crash mid-write never leaves a truncated file
    """
    import json
    with RESULTS_LOCK:
        if RESULTS_FILE.exists():
            try:
                rows: list[dict] = json.loads(RESULTS_FILE.read_text())
            except Exception:
                rows = []
        else:
            rows = []

        existing_ids = {r.get("id") for r in rows}
        if row.get("id") not in existing_ids:
            rows.append(row)

        # Write to a sibling tmp file, then atomically rename
        with tempfile.NamedTemporaryFile(
            mode="w", dir=RESULTS_FILE.parent,
            suffix=".tmp", delete=False
        ) as tf:
            json.dump(rows, tf, indent=2)
            tmp_path = tf.name
        os.replace(tmp_path, RESULTS_FILE)


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


def best_per_benchmark_sql(threshold: int = 500) -> dict[str, float]:
    """Return best val_l2_rel per benchmark, using DuckDB above `threshold` rows.

    DuckDB connection setup costs ~50ms, so for small result sets the Python
    loop is faster.  Above `threshold` experiments DuckDB is typically 5-20x
    faster.  Falls back to the pure-Python path if duckdb is unavailable or
    the result set is below threshold.
    """
    # Use Python path for small files (avoids ~50ms DuckDB startup overhead)
    rows = load_results()
    if len(rows) < threshold:
        return best_per_benchmark(rows)

    sql_rows = query_results(
        "SELECT benchmark, MIN(CAST(val_l2_rel AS DOUBLE)) AS best "
        "FROM results WHERE status = 'keep' GROUP BY benchmark"
    )
    if sql_rows:
        return {r["benchmark"]: float(r["best"]) for r in sql_rows if r["best"] is not None}
    return best_per_benchmark(rows)
