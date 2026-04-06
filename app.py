"""Command Center API — FastAPI backend for the SciML Discovery Engine."""

import json
from pathlib import Path
from typing import List, Optional

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


def _tracker():
    """Always return a fresh Tracker so new results.json writes are reflected."""
    from tracker import Tracker
    return Tracker()


# ── Experiment data ───────────────────────────────────────────────────────────

@app.get("/api/experiments")
def get_experiments():
    return _tracker().get_lineage()


@app.get("/api/experiment/{exp_id}")
def get_experiment(exp_id: str):
    t = _tracker()
    exp = t.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    inspect_path = FIGS_DIR / f"inspect_{exp_id}.png"
    if inspect_path.exists():
        exp["inspect_url"] = f"/figs/inspect_{exp_id}.png"
    return exp


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


@app.get("/api/status")
def get_status():
    import mlx.core as mx
    t = _tracker()
    exps = t.experiments
    return {
        "vram_peak_mb": mx.get_peak_memory() / 1024 / 1024,
        "experiments_count": len(exps),
        "last_updated": exps[-1]["timestamp"] if exps else 0,
        "paused": PAUSE_FILE.exists(),
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
