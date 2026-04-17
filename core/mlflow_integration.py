"""MLflow integration for SciML AutoResearch.

Each training run is logged as an MLflow run, grouped by benchmark as the
MLflow experiment name.  This sits *alongside* results.json — it does not
replace the SSoT Tracker.

Usage (called automatically from train.py):
    from core.mlflow_integration import log_run, log_step_metric
    run_id = log_run(benchmark, model, exp_name, params, metrics, artifact_paths)

The MLflow UI can be launched with:
    uv run mlflow ui --backend-store-uri file://./mlruns
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import mlflow
    import mlflow.artifacts
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

from core.utils import REPO_ROOT

MLRUNS_DIR = REPO_ROOT / "mlruns"
# MLflow 3.x recommends SQLite over the (deprecated) file:// store.
_DB_PATH = REPO_ROOT / "mlflow.db"
_TRACKING_URI = f"sqlite:///{_DB_PATH}"

# Module-level active run handle (for step-level logging during training)
_active_run: Optional[Any] = None


def _ensure_available() -> bool:
    if not _MLFLOW_AVAILABLE:
        print("[MLflow] mlflow not installed — logging skipped. Run: uv sync")
        return False
    return True


# ── End-of-run logging ────────────────────────────────────────────────────────

def log_run(
    benchmark: str,
    model: str,
    exp_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifact_paths: Optional[List[str]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Log a completed training run to MLflow.

    Args:
        benchmark:      Benchmark name (becomes the MLflow experiment name).
        model:          Model class name (e.g. "FNO", "TFNO").
        exp_name:       Unique experiment name from experiments.yaml.
        params:         Hyperparameter dict (logged as MLflow params).
        metrics:        Final metric dict (val_l2_rel, training_seconds, …).
        artifact_paths: Optional list of file paths to upload as artifacts.
        tags:           Optional extra string tags.

    Returns:
        MLflow run_id string, or None if logging was skipped.
    """
    if not _ensure_available():
        return None

    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment(benchmark)  # one MLflow experiment per benchmark

    with mlflow.start_run(run_name=exp_name) as run:
        # Core tags
        mlflow.set_tag("benchmark", benchmark)
        mlflow.set_tag("model", model)
        mlflow.set_tag("exp_name", exp_name)
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, str(v))

        # Hyperparameters
        mlflow.log_params(params)

        # Final metrics
        mlflow.log_metrics(metrics)

        # Artifacts: checkpoint file + log file
        if artifact_paths:
            for p in artifact_paths:
                path = Path(p)
                if path.exists():
                    mlflow.log_artifact(str(path))

        run_id = run.info.run_id

    return run_id


# ── Step-level logging (called from Trainer during training loop) ─────────────

def start_run(benchmark: str, exp_name: str, params: Dict[str, Any]) -> Optional[str]:
    """Open an MLflow run at training start for step-level metric logging.

    Returns the run_id, or None if MLflow is unavailable.
    Stores the active run in module state so log_step() can use it.
    """
    global _active_run
    if not _ensure_available():
        return None

    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment(benchmark)
    _active_run = mlflow.start_run(run_name=exp_name)
    mlflow.set_tag("benchmark", benchmark)
    mlflow.set_tag("exp_name", exp_name)
    mlflow.log_params(params)
    return _active_run.info.run_id


def log_step_metric(key: str, value: float, step: int) -> None:
    """Log a single metric at a given step (e.g. mid-run val_l2_rel)."""
    if not _MLFLOW_AVAILABLE or _active_run is None:
        return
    mlflow.log_metric(key, value, step=step)


def end_run(
    metrics: Dict[str, float],
    artifact_paths: Optional[List[str]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """Finalise the active run with end-of-training metrics and artifacts."""
    global _active_run
    if not _MLFLOW_AVAILABLE or _active_run is None:
        return

    mlflow.log_metrics(metrics)
    if tags:
        for k, v in tags.items():
            mlflow.set_tag(k, str(v))
    if artifact_paths:
        for p in artifact_paths:
            path = Path(p)
            if path.exists():
                mlflow.log_artifact(str(path))

    mlflow.end_run()
    _active_run = None


# ── Convenience: read back best run for a benchmark ──────────────────────────

def get_best_run(benchmark: str) -> Optional[Dict[str, Any]]:
    """Return the MLflow run with the lowest val_l2_rel for a benchmark."""
    if not _ensure_available():
        return None

    mlflow.set_tracking_uri(_TRACKING_URI)
    client = mlflow.MlflowClient()
    try:
        exp = client.get_experiment_by_name(benchmark)
        if exp is None:
            return None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.val_l2_rel ASC"],
            max_results=1,
        )
        if not runs:
            return None
        r = runs[0]
        return {
            "run_id":      r.info.run_id,
            "exp_name":    r.data.tags.get("exp_name", ""),
            "val_l2_rel":  r.data.metrics.get("val_l2_rel"),
            "params":      r.data.params,
            "metrics":     r.data.metrics,
        }
    except Exception as e:
        print(f"[MLflow] get_best_run failed: {e}")
        return None
