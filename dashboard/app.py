"""Command Center API — FastAPI backend for the SciML Discovery Engine."""

import sys
from pathlib import Path

# Ensure repo root is on sys.path regardless of how this file is invoked
# (e.g. `python3 dashboard/app.py` or `uvicorn dashboard.app:app`)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
import math
from typing import List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.utils import FIGS_DIR, LOGS_DIR, REPO_ROOT, SOTA, TELEMETRY_DIR, SENTINEL_DIR

PAUSE_FILE = SENTINEL_DIR / ".autorun_pause"

app = FastAPI(title="SciML Command Center API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FIGS_DIR.exists():
    app.mount("/figs", StaticFiles(directory=str(FIGS_DIR)), name="figs")

if LOGS_DIR.exists():
    app.mount("/logs", StaticFiles(directory=str(LOGS_DIR)), name="logs")

UI_DIR = REPO_ROOT / "dashboard" / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")


@app.get("/")
def get_dashboard():
    from fastapi.responses import FileResponse
    return FileResponse(UI_DIR / "dashboard.html")


def _tracker():
    """Always return a fresh Tracker so new results.json writes are reflected."""
    from core.tracker import Tracker
    return Tracker()


def sanitize(data: Any) -> Any:
    """Recursively replace NaN/Inf with None for JSON compliance."""
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    elif isinstance(data, dict):
        return {k: sanitize(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize(v) for v in data]
    return data


# ── Experiment data ───────────────────────────────────────────────────────────

@app.get("/api/experiments")
def get_experiments():
    return sanitize(_tracker().get_lineage())


@app.get("/api/experiment/{exp_id}")
def get_experiment(exp_id: str):
    t = _tracker()
    exp = t.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    # inspect PNG is named with the inspect_id logged by train.py (stored in conclusion)
    inspect_id = (exp.get("conclusion") or "").strip()
    for candidate in [inspect_id, exp_id]:
        inspect_path = FIGS_DIR / f"inspect_{candidate}.png"
        if inspect_path.exists():
            exp["inspect_url"] = f"/figs/inspect_{candidate}.png"
            break
    return sanitize(exp)


@app.get("/api/lineage")
def get_lineage():
    exps = _tracker().get_lineage()
    nodes, links = [], []
    for e in exps:
        nodes.append({
            "id": e["id"],
            "label": f"{e['model']} ({e['val_l2_rel']:.4f})",
            "status": e["status"],
            "benchmark": e["benchmark"],
        })
        if e.get("parent_id"):
            links.append({"source": e["parent_id"], "target": e["id"]})
    return {"nodes": nodes, "links": links}


TELEMETRY_FILE = TELEMETRY_DIR / ".vram_telemetry"

def _read_telemetry_files() -> list[dict]:
    """Read all .vram_telemetry* files and return a list of live experiment dicts.

    Each dict has: experiment, vram_active_mb, vram_peak_mb, progress,
    remaining_s, step, loss, loss_history.  Stale files (>60s old) are skipped.
    Falls back to the legacy .vram_telemetry file for backwards compatibility.
    """
    import time as _time
    now = _time.time()
    results = []
    seen_experiments = set()

    # Collect all telemetry files (named + legacy)
    telemetry_files = sorted(TELEMETRY_DIR.glob(".vram_telemetry*"))

    for path in telemetry_files:
        try:
            age_s = now - path.stat().st_mtime
            if age_s > 60:
                continue
            data = json.loads(path.read_text())
        except Exception:
            continue
        exp_name = data.get("experiment", "")
        if exp_name in seen_experiments:
            continue
        seen_experiments.add(exp_name)
        results.append({
            "experiment":     exp_name,
            "vram_active_mb": data.get("vram_active_mb", 0.0),
            "vram_peak_mb":   data.get("vram_peak_mb",   0.0),
            "progress":       data.get("progress"),
            "remaining_s":    data.get("remaining_s"),
            "step":           data.get("step"),
            "loss":           data.get("loss"),
            "loss_history":   data.get("loss_history"),
        })
    return results


@app.get("/api/status")
def get_status():
    t = _tracker()
    exps = t.experiments
    active_runs = _read_telemetry_files()

    # Aggregate totals across all active experiments
    vram_active_mb = sum(r["vram_active_mb"] for r in active_runs)
    vram_peak_mb   = max((r["vram_peak_mb"] for r in active_runs), default=0.0)

    # For backwards-compat single-experiment fields: use the most-progressed run
    primary = max(active_runs, key=lambda r: r.get("progress") or 0) if active_runs else {}

    return {
        # Single-experiment backwards-compat fields (most-progressed run)
        "vram_active_mb":    vram_active_mb,
        "vram_peak_mb":      vram_peak_mb,
        "progress":          primary.get("progress"),
        "remaining_s":       primary.get("remaining_s"),
        "step":              primary.get("step"),
        "loss":              primary.get("loss"),
        "loss_history":      primary.get("loss_history"),
        # Per-experiment breakdown (new field — dashboard uses this for multi-run display)
        "active_runs":       active_runs,
        "experiments_count": len(exps),
        "last_updated":      exps[-1]["timestamp"] if exps else 0,
        "paused":            PAUSE_FILE.exists(),
    }


@app.get("/api/sota")
def get_sota():
    """Return SOTA targets and our current best per benchmark."""
    from core.utils import best_per_benchmark
    exps = _tracker().get_lineage()
    our_best = best_per_benchmark(exps)
    result = {}
    for bm, target in SOTA.items():
        best = our_best.get(bm)
        result[bm] = {
            "sota": target,
            "our_best": best,
            "ratio": round(best / target, 3) if best else None,
            "beats_sota": best is not None and best < target,
        }
    return result


# ── Log viewer ────────────────────────────────────────────────────────────────

@app.get("/api/logs/{exp_name}")
def get_log(exp_name: str, tail: int = 300):
    """Return last N lines of an experiment log file."""
    log_path = LOGS_DIR / f"{exp_name}.log"
    if not log_path.exists():
        # Try without extension in case name is stored differently
        candidates = list(LOGS_DIR.glob(f"{exp_name}*"))
        if not candidates:
            raise HTTPException(status_code=404, detail=f"No log found for {exp_name!r}")
        log_path = candidates[0]
    content = log_path.read_text(errors="replace")
    all_lines = content.splitlines()
    return {
        "name": exp_name,
        "total_lines": len(all_lines),
        "lines": all_lines[-tail:],
        "path": str(log_path.name),
    }


@app.get("/api/logs")
def list_logs():
    """List all available log files."""
    if not LOGS_DIR.exists():
        return []
    return sorted(p.stem for p in LOGS_DIR.glob("*.log"))


# ── Diagnostics ───────────────────────────────────────────────────────────────

@app.get("/api/diagnostics/{exp_name}")
def get_diagnostics(exp_name: str):
    """Return diagnostics for a completed or in-progress experiment.

    Sources (merged in priority order):
    1. Live telemetry file (.vram_telemetry_<name>) — grad_norm, loss_history
    2. Log file (logs/<name>.log)  — grad_norm_max, spectral bias, val_history
    3. results.json entry — diag snapshot stored at run end

    Returns a JSON object with keys:
        grad_norm_current, grad_norm_max, loss_history,
        low_freq_error, high_freq_error, val_history,
        vram_active_mb, vram_peak_mb, progress, status
    """
    from core.diagnostics import parse_log_file

    result: dict = {
        "exp_name": exp_name,
        "grad_norm_current": None,
        "grad_norm_max": None,
        "loss_history": [],
        "val_history": [],
        "low_freq_error": None,
        "high_freq_error": None,
        "vram_active_mb": None,
        "vram_peak_mb": None,
        "progress": None,
        "status": "unknown",
    }

    # 1. Live telemetry (in-progress run)
    slug = exp_name.replace("/", "_").replace(" ", "_")
    telemetry_path = TELEMETRY_DIR / f".vram_telemetry_{slug}"
    if not telemetry_path.exists():
        telemetry_path = TELEMETRY_DIR / ".vram_telemetry"
    if telemetry_path.exists():
        try:
            telem = json.loads(telemetry_path.read_text())
            result["grad_norm_current"] = telem.get("grad_norm")
            result["grad_norm_max"]     = telem.get("max_grad_norm")
            result["loss_history"]      = telem.get("loss_history", [])
            result["vram_active_mb"]    = telem.get("vram_active_mb")
            result["vram_peak_mb"]      = telem.get("vram_peak_mb")
            result["progress"]          = telem.get("progress")
            result["status"]            = "running"
        except Exception:
            pass

    # 2. Completed log file
    log_path = LOGS_DIR / f"{exp_name}.log"
    if log_path.exists():
        try:
            parsed = parse_log_file(log_path)
            diag = parsed.get("diag", {})
            if diag.get("diag_grad_norm_max") is not None:
                result["grad_norm_max"]  = diag["diag_grad_norm_max"]
            if diag.get("diag_low_freq_error") is not None:
                result["low_freq_error"] = diag["diag_low_freq_error"]
            if diag.get("diag_high_freq_error") is not None:
                result["high_freq_error"] = diag["diag_high_freq_error"]
            if parsed.get("val") is not None:
                result["val_history"].append(parsed["val"])
                result["status"] = "completed"
        except Exception:
            pass

    # 3. results.json snapshot
    try:
        from core.utils import load_results
        all_results = load_results()
        entry = next(
            (e for e in reversed(all_results)
             if (e.get("description") or "").startswith(exp_name)
             or e.get("config", {}).get("name") == exp_name),
            None
        )
        if entry:
            snap = entry.get("diag") or {}
            if snap.get("diag_low_freq_error") and result["low_freq_error"] is None:
                result["low_freq_error"] = snap["diag_low_freq_error"]
            if snap.get("diag_high_freq_error") and result["high_freq_error"] is None:
                result["high_freq_error"] = snap["diag_high_freq_error"]
    except Exception:
        pass

    return result


# ── Queue ─────────────────────────────────────────────────────────────────────

@app.get("/api/queue")
def get_queue():
    """Return pending experiments sorted by effective priority (overrides applied)."""
    from core.loader import get_experiments
    from core.utils import done_names
    done = done_names()

    # Load dashboard priority overrides (written by POST /api/priority)
    overrides_path = SENTINEL_DIR / ".priority_overrides.json"
    overrides: dict = {}
    if overrides_path.exists():
        try:
            overrides = json.loads(overrides_path.read_text())
        except Exception:
            pass

    pending = [
        {
            "name": e.name,
            "benchmark": e.benchmark,
            "model": e.model,
            "priority": overrides.get(e.name, e.priority),
            "hidden_dim": e.hidden_dim,
            "n_layers": e.n_layers,
            "n_modes": e.n_modes,
            "budget_s": e.budget_s,
            "rationale": e.rationale,
        }
        for e in get_experiments()
        if e.name not in done
    ]
    pending.sort(key=lambda x: x["priority"])
    return {"pending": len(pending), "experiments": pending}


@app.get("/api/active")
def get_active():
    """Return names of experiments that are currently running."""
    import time
    active = []
    
    # Priority 1: Check for explicit signaling file
    active_file = SENTINEL_DIR / ".active_experiment"
    if active_file.exists():
        try:
            name = active_file.read_text().strip()
            if name:
                active.append(name)
        except Exception:
            pass

    # Priority 2: Check for recently modified logs (fallback/backup)
    if LOGS_DIR.exists():
        now = time.time()
        for log_path in LOGS_DIR.glob("*.log"):
            # Increased threshold to 120s for cases where mtime updates are slow
            if now - log_path.stat().st_mtime < 120:
                # Skip the autorun meta-logs
                if log_path.stem.startswith("autorun"):
                    continue
                name = log_path.stem
                if name not in active:
                    active.append(name)
                    
    return {"active": active}


@app.post("/api/kill/{name}")
def kill_experiment(name: str):
    """Request termination of a running experiment by writing a sentinel file.
    autorun.py polls for this file every 2s and calls proc.terminate() when found."""
    kill_file = SENTINEL_DIR / f".kill_{name}"
    kill_file.touch()
    return {"status": "kill_requested", "name": name,
            "message": "Sentinel written — experiment will stop within ~2s."}


# ── Control endpoints ─────────────────────────────────────────────────────────

@app.post("/api/pause")
def pause_autorun():
    PAUSE_FILE.touch()
    return {"status": "paused"}


@app.post("/api/resume")
def resume_autorun():
    PAUSE_FILE.unlink(missing_ok=True)
    return {"status": "running"}


@app.get("/api/pause")
def get_pause_status():
    return {"paused": PAUSE_FILE.exists()}


class PriorityUpdate(BaseModel):
    name: str
    priority: int


@app.post("/api/priority")
def update_priority(update: PriorityUpdate):
    overrides_path = SENTINEL_DIR / ".priority_overrides.json"
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    overrides[update.name] = update.priority
    overrides_path.write_text(json.dumps(overrides, indent=2))
    return {"status": "ok", "name": update.name, "new_priority": update.priority}


class ExperimentInject(BaseModel):
    name: str
    benchmark: str
    model: str
    hidden_dim: int
    n_layers: int
    n_modes: int = 16
    priority: int = 1
    rationale: str = ""


@app.post("/api/inject")
def inject_experiment(exp: ExperimentInject):
    injections_path = SENTINEL_DIR / ".injected_experiments.json"
    injections = json.loads(injections_path.read_text()) if injections_path.exists() else []
    injections.append(exp.dict())
    injections_path.write_text(json.dumps(injections, indent=2))
    return {"status": "queued", "name": exp.name}


# ── Architecture visualizations ──────────────────────────────────────────────

ARCH_DIR = FIGS_DIR / "arch"

@app.get("/api/arch-map")
def get_arch_map():
    """Return mapping of registry model key → image URL for every available PNG."""
    if not ARCH_DIR.exists():
        return {}
    return {
        p.stem: f"/figs/arch/{p.name}"
        for p in sorted(ARCH_DIR.glob("*.png"))
    }


# ── Model Registry ────────────────────────────────────────────────────────────

@app.get("/api/model-registry")
def get_model_registry():
    """Return all model versions grouped by benchmark.

    Merges two sources:
      1. model_registry.json  — explicitly registered versions with checkpoint paths
      2. results.json         — best "keep" result per (benchmark, model) for any
                                benchmark not yet in the registry (no checkpoint needed)
    """
    result: dict = {}

    # ── Source 1: model_registry.json ────────────────────────────────────────
    try:
        from core.model_versioning import get_registry
        reg = get_registry()
        for v in reg._versions:
            bm = v.benchmark
            if bm not in result:
                result[bm] = {"versions": []}
            result[bm]["versions"].append({
                "version_id":    v.version_id,
                "benchmark":     v.benchmark,
                "model":         v.model,
                "exp_name":      v.exp_name,
                "val_l2_rel":    v.val_l2_rel,
                "ckpt_path":     v.ckpt_path,
                "timestamp":     v.timestamp,
                "is_champion":   v.is_champion,
                "exists":        v.exists(),
                "mlflow_run_id": v.mlflow_run_id,
                "config":        v.config,
                "source":        "registry",
            })
    except Exception:
        pass

    # ── Source 2: results.json fallback ──────────────────────────────────────
    # For every (benchmark, model) pair, surface the best "keep" result even
    # when no checkpoint was saved (so the Models tab is never empty).
    try:
        from core.utils import RESULTS_FILE
        import time as _time
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                all_results = json.load(f)

            # Build set of (benchmark, model) already covered by registry
            covered = {
                (v["benchmark"], v["model"])
                for bm_data in result.values()
                for v in bm_data["versions"]
            }

            # Best "keep" per (benchmark, model) not already in registry
            best: dict = {}
            for e in all_results:
                if e.get("status") != "keep":
                    continue
                val = e.get("val_l2_rel")
                if not val or val <= 0 or val >= 10.0:
                    continue
                key = (e["benchmark"], e["model"])
                if key in covered:
                    continue
                if key not in best or val < best[key]["val_l2_rel"]:
                    best[key] = e

            for (bm, model), e in sorted(best.items()):
                cfg      = e.get("config") or {}
                exp_name = cfg.get("name") or e.get("description", "").split()[0]
                if bm not in result:
                    result[bm] = {"versions": []}
                result[bm]["versions"].append({
                    "version_id":    e["id"],
                    "benchmark":     bm,
                    "model":         model,
                    "exp_name":      exp_name,
                    "val_l2_rel":    e["val_l2_rel"],
                    "ckpt_path":     None,
                    "timestamp":     e.get("timestamp", 0),
                    "is_champion":   True,   # best of its kind for this benchmark
                    "exists":        False,
                    "mlflow_run_id": None,
                    "config":        cfg,
                    "source":        "results",
                })
    except Exception:
        pass

    # Sort versions within each benchmark; mark the overall champion
    for bm_data in result.values():
        vs = bm_data["versions"]
        vs.sort(key=lambda x: x["val_l2_rel"])
        # Ensure only the single best version is marked champion
        champion_set = False
        for v in vs:
            if not champion_set and v["is_champion"]:
                champion_set = True
            elif v["source"] == "results":
                # results-derived entries are champion only if nothing better exists
                v["is_champion"] = not champion_set

    return sanitize(result)


@app.get("/api/model-registry/{benchmark}")
def get_model_registry_benchmark(benchmark: str):
    """Return all registered versions for one benchmark."""
    full = get_model_registry()
    data = full.get(benchmark)
    if not data:
        raise HTTPException(status_code=404, detail=f"No versions for {benchmark!r}")
    return data


# ── Reflection ───────────────────────────────────────────────────────────────

@app.get("/api/reflection/{benchmark}")
def get_reflection(benchmark: str):
    """Return the latest reflection for a benchmark (what worked, what didn't, next steps)."""
    from core.closed_loop_reasoner import generate_reflection, load_reflections
    # Try cached first
    cached = load_reflections().get(benchmark)
    if cached:
        return sanitize(cached)
    # Generate fresh
    try:
        return sanitize(generate_reflection(benchmark))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reflections")
def get_all_reflections():
    """Return all saved reflections keyed by benchmark."""
    from core.closed_loop_reasoner import load_reflections
    return sanitize(load_reflections())


@app.post("/api/reflection/{benchmark}/refresh")
def refresh_reflection(benchmark: str):
    """Force-regenerate reflection for a benchmark from current results."""
    from core.closed_loop_reasoner import generate_reflection
    try:
        return sanitize(generate_reflection(benchmark))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── MLflow runs ───────────────────────────────────────────────────────────────

@app.get("/api/mlflow/runs")
def get_mlflow_runs(benchmark: Optional[str] = None, limit: int = 200):
    """Return MLflow run summaries, optionally filtered by benchmark."""
    try:
        import mlflow
        from core.mlflow_integration import _TRACKING_URI
        mlflow.set_tracking_uri(_TRACKING_URI)
        client = mlflow.MlflowClient()
        experiments = client.search_experiments()
        runs: list = []
        for exp in experiments:
            if benchmark and exp.name != benchmark:
                continue
            for r in client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["metrics.val_l2_rel ASC"],
                max_results=limit,
            ):
                runs.append({
                    "run_id":           r.info.run_id,
                    "benchmark":        exp.name,
                    "exp_name":         r.data.tags.get("exp_name", r.info.run_name or ""),
                    "model":            r.data.tags.get("model", ""),
                    "val_l2_rel":       r.data.metrics.get("val_l2_rel"),
                    "training_seconds": r.data.metrics.get("training_seconds"),
                    "peak_vram_mb":     r.data.metrics.get("peak_vram_mb"),
                    "num_steps":        r.data.metrics.get("num_steps"),
                    "diag_high_freq":   r.data.metrics.get("diag_high_freq_error"),
                    "diag_low_freq":    r.data.metrics.get("diag_low_freq_error"),
                    "start_time":       r.info.start_time,
                    "params":           dict(r.data.params),
                })
        runs.sort(key=lambda x: (x["benchmark"], x.get("val_l2_rel") or 999))
        return sanitize({"runs": runs, "total": len(runs)})
    except Exception as e:
        return {"runs": [], "total": 0, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
