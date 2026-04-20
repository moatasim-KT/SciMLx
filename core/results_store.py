"""Concurrency-safe results store for the SciML autoresearch loop.

Architecture:
  - SQLite (WAL mode) is the primary store — handles concurrent readers/writers
    natively with no external locking needed for DB operations.
  - results.json is a human-readable export kept in sync after every write,
    protected by a FileLock + atomic rename so the JSON is never half-written.
  - All code must go through ResultsStore — never write results.json directly.

Usage:
    from core.results_store import store

    store.append(row_dict)           # insert one result (idempotent by id)
    rows = store.load()              # all rows as list[dict]
    rows = store.load("burgers_1d")  # filtered by benchmark
    store.export_json()              # explicit JSON resync (called after append)
"""

import json
import os
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Optional

from filelock import FileLock

from core.utils import REPO_ROOT

DB_FILE      = REPO_ROOT / "results.db"
JSON_FILE    = REPO_ROOT / "results.json"
_JSON_LOCK   = FileLock(str(JSON_FILE) + ".lock", timeout=30)

# SQLite connection pool — one connection per thread, WAL shared across processes
_local = threading.local()

def _conn() -> sqlite3.Connection:
    if not getattr(_local, "conn", None):
        con = sqlite3.connect(str(DB_FILE), timeout=30, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")   # safe with WAL, faster than FULL
        con.execute("PRAGMA foreign_keys=ON")
        _local.conn = con
    return _local.conn


def _ensure_schema() -> None:
    con = _conn()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS results (
            id          TEXT PRIMARY KEY,
            parent_id   TEXT,
            timestamp   INTEGER,
            benchmark   TEXT NOT NULL,
            model       TEXT NOT NULL,
            val_l2_rel  REAL,
            memory_gb   REAL,
            status      TEXT,
            description TEXT,
            git_commit  TEXT,
            config      TEXT,   -- JSON blob
            rationale   TEXT,
            conclusion  TEXT,
            diag        TEXT    -- JSON blob
        );
        CREATE INDEX IF NOT EXISTS idx_benchmark ON results(benchmark);
        CREATE INDEX IF NOT EXISTS idx_status    ON results(status);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON results(timestamp);
    """)
    con.commit()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    # Rename storage column back to the canonical key used everywhere
    d["commit"] = d.pop("git_commit", None)
    for blob_col in ("config", "diag"):
        if d.get(blob_col):
            try:
                d[blob_col] = json.loads(d[blob_col])
            except (json.JSONDecodeError, TypeError):
                d[blob_col] = {}
        else:
            d[blob_col] = {}
    try:
        d["val_l2_rel"] = float(d.get("val_l2_rel") or float("nan"))
    except (ValueError, TypeError):
        d["val_l2_rel"] = float("nan")
    return d


class ResultsStore:
    """Thread- and process-safe results store backed by SQLite WAL."""

    def __init__(self):
        _ensure_schema()

    # ── Read ──────────────────────────────────────────────────────────────────

    def load(self, benchmark: Optional[str] = None) -> list[dict]:
        """Return all result rows, optionally filtered by benchmark."""
        con = _conn()
        if benchmark:
            cur = con.execute(
                "SELECT * FROM results WHERE benchmark=? ORDER BY timestamp",
                (benchmark,),
            )
        else:
            cur = con.execute("SELECT * FROM results ORDER BY timestamp")
        return [_row_to_dict(r) for r in cur.fetchall()]

    def best_per_benchmark(self) -> dict[str, float]:
        """Return lowest val_l2_rel among kept rows, keyed by benchmark."""
        con = _conn()
        cur = con.execute(
            "SELECT benchmark, MIN(val_l2_rel) AS best "
            "FROM results WHERE status='keep' AND val_l2_rel IS NOT NULL "
            "GROUP BY benchmark"
        )
        return {r["benchmark"]: float(r["best"]) for r in cur.fetchall()}

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Run arbitrary SQL against the results table."""
        con = _conn()
        cur = con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── Write ─────────────────────────────────────────────────────────────────

    def append(self, row: dict) -> None:
        """Insert one result row (idempotent — silently skips duplicate ids).

        After inserting, atomically re-exports results.json under FileLock so
        the human-readable JSON stays in sync.
        """
        con = _conn()
        con.execute(
            """INSERT OR IGNORE INTO results
               (id, parent_id, timestamp, benchmark, model, val_l2_rel,
                memory_gb, status, description, git_commit, config,
                rationale, conclusion, diag)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row.get("id"),
                row.get("parent_id"),
                row.get("timestamp"),
                row.get("benchmark", ""),
                row.get("model", ""),
                row.get("val_l2_rel"),
                row.get("memory_gb"),
                row.get("status"),
                row.get("description"),
                row.get("commit"),
                json.dumps(row.get("config") or {}),
                row.get("rationale"),
                row.get("conclusion"),
                json.dumps(row.get("diag") or {}),
            ),
        )
        con.commit()
        self.export_json()

    # ── JSON export ───────────────────────────────────────────────────────────

    def export_json(self) -> None:
        """Write results.json from the DB under FileLock + atomic rename.

        Safe to call from multiple processes simultaneously — the lock ensures
        only one writer at a time and the rename is POSIX-atomic.
        """
        rows = self.load()
        with _JSON_LOCK:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=JSON_FILE.parent, suffix=".tmp", delete=False
            ) as tf:
                json.dump(rows, tf, indent=2)
                tmp = tf.name
            os.replace(tmp, JSON_FILE)

    # ── Migration ─────────────────────────────────────────────────────────────

    def import_json(self, path: Path = JSON_FILE) -> int:
        """Bulk-import rows from a JSON file, skipping any already in the DB.

        Returns the number of new rows inserted.
        """
        if not path.exists():
            return 0
        try:
            with _JSON_LOCK:
                rows = json.loads(path.read_text())
        except Exception as e:
            print(f"[ResultsStore] import_json failed to read {path}: {e}")
            return 0

        con = _conn()
        inserted = 0
        for row in rows:
            cur = con.execute(
                """INSERT OR IGNORE INTO results
                   (id, parent_id, timestamp, benchmark, model, val_l2_rel,
                    memory_gb, status, description, git_commit, config,
                    rationale, conclusion, diag)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row.get("id"),
                    row.get("parent_id"),
                    row.get("timestamp"),
                    row.get("benchmark", ""),
                    row.get("model", ""),
                    row.get("val_l2_rel"),
                    row.get("memory_gb"),
                    row.get("status"),
                    row.get("description"),
                    row.get("commit"),
                    json.dumps(row.get("config") or {}),
                    row.get("rationale"),
                    row.get("conclusion"),
                    json.dumps(row.get("diag") or {}),
                ),
            )
            inserted += cur.rowcount
        con.commit()
        return inserted


# Module-level singleton — import this everywhere
store = ResultsStore()
