"""Shared utilities for the SciML autoresearch loop.

Centralises path constants, SOTA targets, and results helpers.

Results I/O — all code must use these functions, never write results.json directly:
  load_results(benchmark?)  — read from SQLite (primary) with JSON fallback
  append_result(row)        — write to SQLite + atomically export JSON
  best_per_benchmark(rows)  — pure-Python aggregation
  best_per_benchmark_sql()  — SQLite-native aggregation (always fast)
  query_results(sql)        — arbitrary SQL via SQLite or DuckDB on JSON
"""

import math
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT     = Path(__file__).parent.parent
RESULTS_FILE  = REPO_ROOT / "results.json"
LOGS_DIR      = REPO_ROOT / "logs"
TELEMETRY_DIR = LOGS_DIR / "telemetry"
SENTINEL_DIR  = LOGS_DIR / "sentinels"
FIGS_DIR      = REPO_ROOT / "figs"
PAPERS_DIR    = REPO_ROOT / "docs" / "papers"

for d in [LOGS_DIR, TELEMETRY_DIR, SENTINEL_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── SOTA targets ──────────────────────────────────────────────────────────────

SOTA: dict[str, float] = {
    "burgers_1d":       0.0031,
    "kdv_1d":           0.010,
    "wave_1d":          0.005,
    "euler_1d":         0.003,
    "burgers_nu_001":   0.080,
    "darcy_2d":         0.0041,
    "ns_2d":            0.0128,
    "ns_hre_2d":        0.050,
    "swe_2d":           0.015,
    "allen_cahn_2d":    0.080,
    "elasticity_2d":    0.010,
    "wavebench_2d":     0.015,
    "pdebench_2d":      0.005,
    "mhd_2d":           0.050,
    "multiphysics_2d":  0.200,
}

# ── Results I/O — always go through ResultsStore ──────────────────────────────

def load_results(benchmark: Optional[str] = None) -> list[dict]:
    """Return rows from the SQLite store, optionally filtered by benchmark.

    Falls back to the JSON file if the DB doesn't exist yet (first run or
    CI environment without results.db).  Each row has val_l2_rel as float.
    """
    try:
        from core.results_store import store
        return store.load(benchmark)
    except Exception:
        pass
    # JSON fallback (bootstrap / migration path)
    import json
    from filelock import FileLock
    lock = FileLock(str(RESULTS_FILE) + ".lock", timeout=30)
    if not RESULTS_FILE.exists():
        return []
    try:
        with lock:
            rows = json.loads(RESULTS_FILE.read_text())
    except Exception as e:
        print(f"Warning: Could not load {RESULTS_FILE}: {e}")
        return []
    seen: set = set()
    deduped = []
    for row in rows:
        rid = row.get("id")
        if rid not in seen:
            seen.add(rid)
            deduped.append(row)
    out = []
    for row in deduped:
        try:
            row["val_l2_rel"] = float(row.get("val_l2_rel", float("nan")))
        except (ValueError, TypeError):
            row["val_l2_rel"] = float("nan")
        if benchmark and row.get("benchmark") != benchmark:
            continue
        out.append(row)
    return out


def append_result(row: dict) -> None:
    """Write one result to SQLite and atomically re-export results.json."""
    from core.results_store import store
    store.append(row)


# ── Aggregation helpers ───────────────────────────────────────────────────────

def best_per_benchmark(rows: list[dict]) -> dict[str, float]:
    """Return lowest val_l2_rel among 'keep' rows, keyed by benchmark."""
    best: dict[str, float] = {}
    for row in rows:
        if row.get("status") == "keep" and not math.isnan(row["val_l2_rel"]):
            bm = row["benchmark"]
            if bm not in best or row["val_l2_rel"] < best[bm]:
                best[bm] = row["val_l2_rel"]
    return best


def best_per_benchmark_sql(threshold: int = 500) -> dict[str, float]:
    """Return best val_l2_rel per benchmark via SQLite (always fast).

    The threshold parameter is kept for API compatibility but ignored —
    SQLite is faster than Python at any scale and has no startup overhead.
    """
    try:
        from core.results_store import store
        return store.best_per_benchmark()
    except Exception:
        return best_per_benchmark(load_results())


def done_names() -> set[str]:
    """Return set of experiment names already recorded in results."""
    names: set[str] = set()
    for row in load_results():
        cfg_name = (row.get("config") or {}).get("name", "")
        if cfg_name:
            names.add(cfg_name)
        desc = row.get("description", "")
        if desc:
            names.add(desc.split()[0])
    return names


# ── Ad-hoc SQL queries ────────────────────────────────────────────────────────

def query_results(sql: str) -> list[dict]:
    """Run arbitrary SQL against the results table via SQLite.

    The table is named `results`.  Falls back to DuckDB on results.json
    if SQLite is unavailable for some reason.
    """
    try:
        from core.results_store import store
        return store.query(sql)
    except Exception:
        pass
    # DuckDB fallback on JSON
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
        print("[utils] duckdb not installed")
        return []
    except Exception as e:
        print(f"[utils] query_results failed: {e}")
        return []
