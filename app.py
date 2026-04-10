"""Command Center API — FastAPI backend for the SciML Discovery Engine."""

import json
import math
from pathlib import Path
from typing import List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils import FIGS_DIR, LOGS_DIR, REPO_ROOT, SOTA

PAUSE_FILE = REPO_ROOT / ".autorun_pause"

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

UI_DIR = REPO_ROOT / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")


@app.get("/")
def get_dashboard():
    from fastapi.responses import FileResponse
    return FileResponse(UI_DIR / "dashboard.html")


def _tracker():
    """Always return a fresh Tracker so new results.json writes are reflected."""
    from tracker import Tracker
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


TELEMETRY_FILE = REPO_ROOT / ".vram_telemetry"

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
    telemetry_files = sorted(REPO_ROOT.glob(".vram_telemetry*"))

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
    from utils import best_per_benchmark
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


# ── Queue ─────────────────────────────────────────────────────────────────────

@app.get("/api/queue")
def get_queue():
    """Return pending experiments sorted by priority."""
    from experiments import get_experiments
    from utils import done_names
    done = done_names()
    pending = [
        {
            "name": e.name,
            "benchmark": e.benchmark,
            "model": e.model,
            "priority": e.priority,
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
    active_file = REPO_ROOT / ".active_experiment"
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
    kill_file = REPO_ROOT / f".kill_{name}"
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
    overrides_path = REPO_ROOT / ".priority_overrides.json"
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
    injections_path = REPO_ROOT / ".injected_experiments.json"
    injections = json.loads(injections_path.read_text()) if injections_path.exists() else []
    injections.append(exp.dict())
    injections_path.write_text(json.dumps(injections, indent=2))
    return {"status": "queued", "name": exp.name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
