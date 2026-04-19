"""Autonomous SciML experiment runner.

Iterates through the experiment queue in experiments.yaml, skips anything
already recorded in results.json, runs each via subprocess, and logs results.

Usage:
    uv run autorun.py                          # run all pending experiments
    uv run autorun.py --max 5                  # stop after 5 experiments
    uv run autorun.py --benchmark burgers_1d   # filter by benchmark
    uv run autorun.py --model FNO              # filter by model type
    uv run autorun.py --priority 3             # only experiments with priority ≤ 3
    uv run autorun.py --dry-run                # print plan without running
    uv run autorun.py --commit                 # git-commit good results automatically

Logs are written to logs/<name>.log.
results.json is updated after every experiment.
"""

import argparse
import dataclasses
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.diagnostics import parse_log_file, check_early_stop_condition, get_fix_strategies
from core.loader import ExperimentConfig, get_experiments
from core.utils import REPO_ROOT, RESULTS_FILE, LOGS_DIR, SENTINEL_DIR, load_results, done_names, best_per_benchmark
from core.tracker import Tracker

TIMEOUT_S = 7200  # 2 hours per experiment (accommodates heavy 2D data-gen)

TRAJECTORIES_FILE = LOGS_DIR / "trajectories.jsonl"


def _write_trajectory(entry: dict) -> None:
    """Append one JSON line to logs/trajectories.jsonl (RL replay buffer).

    Never raises — a logging failure must not crash the runner.
    Fields written: timestamp, benchmark, state, hypothesis, action,
    expected_outcome, outcome (null while running), diag_snapshot.
    """
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
        with open(TRAJECTORIES_FILE, "a") as _tf:
            _tf.write(json.dumps(entry) + "\n")
    except Exception:
        pass

# Early-stop: kill if any mid-run val (logged by trainer every 10%) exceeds
# baseline x this multiplier at >=30% progress - catches disasters early.
EARLY_STOP_MULTIPLIER: float = 50.0

# Max number of retry strategies to attempt before discarding an experiment.
# Strategy r1 = smart crash fix, r2 = halve model, r3 = minimal viable config.
MAX_RETRY_STRATEGIES: int = 3

# Adaptive retry: if a completed run has val > baseline x this, the config is
# clearly not working - query HypothesisEngine for a smarter config and retry once.
POOR_RESULT_MULTIPLIER: float = 3.0

# Memory budget for parallel scheduling (M1 8GB unified memory)
TOTAL_MEMORY_MB = 7500  # leave ~500MB headroom
# Estimated memory per experiment type (conservative)
MEMORY_ESTIMATE_1D_MB = 2500   # burgers, kdv, wave, euler
MEMORY_ESTIMATE_2D_MB = 5000   # darcy, ns, swe, allen_cahn, ns_hre

BENCHMARKS_2D = {
    # Must stay in sync with the protected set in core/research_plugins.py build()
    "darcy_2d", "ns_2d", "swe_2d", "allen_cahn_2d", "ns_hre_2d", "mhd_2d",
    "elasticity_2d", "wavebench_2d", "pdebench_2d", "multiphysics_2d",
    "radiative_2d",
}

# Minimum training budgets (seconds).  Shorter experiments are upgraded to
# these floors so every run gets enough steps to meaningfully converge.
BUDGET_FLOOR_1D = 1800   # 30 minutes
BUDGET_FLOOR_2D = 3600   # 60 minutes


def _apply_budget_floor(exp: "ExperimentConfig") -> "ExperimentConfig":
    """Return a copy of exp with budget_s raised to the per-dimension floor if needed."""
    floor = BUDGET_FLOOR_2D if exp.benchmark in BENCHMARKS_2D else BUDGET_FLOOR_1D
    if exp.budget_s < floor:
        # Debug print for TASK-005 audit
        print(f"  [AUDIT] Upgrading {exp.name} budget: {exp.budget_s}s -> {floor}s")
        return dataclasses.replace(exp, budget_s=floor)
    return exp

# File written by POST /api/inject — autorun polls this between experiments
INJECTIONS_FILE = SENTINEL_DIR / ".injected_experiments.json"
PAUSE_FILE      = SENTINEL_DIR / ".autorun_pause"
ACTIVE_FILE     = SENTINEL_DIR / ".active_experiment"

def _is_benchmark_stuck(
    benchmark: str,
    results: list,
    window: int = 5,
    threshold: float = 0.02,
) -> bool:
    """Return True if the last `window` kept results show <threshold relative improvement.

    Used to detect plateaus and deprioritise benchmarks where no architecture
    has made progress recently (e.g. burgers_1d plateaued at 0.147 for 11 runs).
    """
    kept = sorted(
        [r for r in results
         if r.get("benchmark") == benchmark
         and r.get("status") == "keep"
         and r.get("val_l2_rel") is not None],
        key=lambda r: r.get("timestamp", ""),
    )
    if len(kept) < window:
        return False
    recent_vals = [r["val_l2_rel"] for r in kept[-window:]]
    best = min(recent_vals)
    worst = max(recent_vals)
    if worst < 1e-9:
        return False
    relative_spread = (worst - best) / worst
    return relative_spread < threshold


def _pick_highest_gap_benchmark() -> str:
    """Return the benchmark with the largest ratio of current_best / SOTA target.

    Falls back to 'burgers_1d' if SOTA targets are unavailable.
    """
    try:
        from core.utils import SOTA
        baselines = get_baselines()
        worst_bm, worst_ratio = "burgers_1d", 0.0
        for bm, sota in SOTA.items():
            current = baselines.get(bm, float("inf"))
            if current < float("inf") and sota > 0:
                ratio = current / sota
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    worst_bm = bm
        return worst_bm
    except Exception:
        return "burgers_1d"


def _run_hpo_batch(benchmark: str, args) -> int:
    """Run HPO suggestions for `benchmark`. Returns number of experiments run.

    Uses OptunaHPO (TPE + MedianPruner) when optuna is available,
    falling back to BayesianHPO (GP + EI) otherwise.
    """
    import time as _t2
    try:
        try:
            from core.hpo import OptunaHPO
            hpo = OptunaHPO(benchmark)
        except Exception:
            from core.hpo import BayesianHPO as _HPO
            hpo = _HPO(benchmark)
        hpo.load_history()
        hpo_configs = hpo.suggest_top(n=3)
        done = load_done_names()
        ran = 0
        for cfg in hpo_configs:
            ts = int(_t2.time())
            exp_name = (
                f"hpo_{benchmark}_{cfg.get('model','FNO').lower()}"
                f"_h{cfg.get('hidden_dim',64)}_l{cfg.get('n_layers',4)}"
                f"_m{cfg.get('n_modes',16)}_{ts}"
            )
            if exp_name in done:
                continue
            _budget = BUDGET_FLOOR_2D if benchmark in BENCHMARKS_2D else BUDGET_FLOOR_1D
            exp = ExperimentConfig(
                name=exp_name,
                benchmark=benchmark,
                model=cfg.get("model", "FNO"),
                hidden_dim=int(cfg.get("hidden_dim", 64)),
                n_layers=int(cfg.get("n_layers", 4)),
                n_modes=int(cfg.get("n_modes", 16)),
                lr=float(cfg.get("lr", 1e-3)),
                budget_s=_budget,
                priority=2,
                rationale=f"BayesianHPO suggestion (EI acquisition) for {benchmark}",
            )
            _write_trajectory({
                "benchmark": benchmark,
                "state": "queue_empty",
                "hypothesis": "BayesianHPO_suggestion",
                "action": f"queue {exp.name}",
                "expected_outcome": "val < current_best",
            })
            _log = LOGS_DIR / f"{exp.name}.log"
            _bl = get_baselines().get(benchmark, float("inf"))
            run_experiment(exp, _log, baseline=_bl)
            ran += 1
            _t2.sleep(1)
        return ran
    except Exception as _e:
        print(f"  [HPO] BayesianHPO failed for {benchmark}: {_e}")
        return 0


def estimate_memory_mb(exp: ExperimentConfig) -> int:
    """Estimate peak memory usage for an experiment."""
    base = MEMORY_ESTIMATE_2D_MB if exp.benchmark in BENCHMARKS_2D else MEMORY_ESTIMATE_1D_MB
    # Scale with model size
    scale = (exp.hidden_dim / 64) * (exp.n_layers / 4)
    return int(base * min(scale, 2.0))

tracker = Tracker()
_tracker_lock = threading.Lock()
_baselines_lock = threading.Lock()
_active_lock = threading.Lock()
_print_lock = threading.Lock()
_active_experiments: set[str] = set()

_worker_prefix: threading.local = threading.local()

# ── Auto-loop state (persist across recursive main() calls) ──────────────────
_AUTO_START_TIME: float = 0.0
_AUTO_EXP_COUNT:  int   = 0

def wprint(*args, **kwargs):
    """Thread-safe print that prefixes output with the current worker's experiment name."""
    prefix = getattr(_worker_prefix, "name", None)
    with _print_lock:
        if prefix:
            print(f"[{prefix}]", *args, **kwargs)
        else:
            print(*args, **kwargs)

# ── Results I/O ───────────────────────────────────────────────────────────────

def load_done_names() -> set[str]:
    return done_names()


def get_baselines() -> dict[str, float]:
    return best_per_benchmark(load_results())


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def is_arch_compatible(e1: ExperimentConfig, e2: ExperimentConfig) -> bool:
    """Return True if two configs share the same architecture (can load weights)."""
    return (
        e1.model == e2.model and
        e1.hidden_dim == e2.hidden_dim and
        e1.n_layers == e2.n_layers and
        e1.n_modes == e2.n_modes and
        e1.n_head == e2.n_head and
        e1.slice_num == e2.slice_num and
        e1.n_levels == e2.n_levels
    )


def smart_fix(exp: ExperimentConfig, log_path: Path, results: dict) -> Optional[tuple]:
    """Analyse crash log and compose ALL applicable fixes into a single retry config.
    Uses centralized fix strategy logic from diagnostics.py.
    """
    crash_type = results.get("crash_type")
    if not crash_type:
        return None

    if crash_type == "IncompatibleDimensions":
        wprint("  DIAGNOSIS: 1D model given 2D input — incompatible pairing, skipping.")
        return None

    # ── Get list of applicable fixes for this specific crash ────────────────
    # get_fix_strategies maps crash_type strings to field overrides
    fixes = get_fix_strategies(crash_type, vars(exp))
    
    if not fixes:
        return None

    # Merge overrides from all fixes (later ones in list win)
    merged_kwargs = {}
    for desc, field_overrides in fixes:
        merged_kwargs.update(field_overrides)

    full_desc = " + ".join(d for d, _ in fixes)
    wprint(f"  DIAGNOSIS: {len(fixes)} fix(es) composed from {crash_type}: {full_desc}")
    
    # Return as (description, new_config)
    return full_desc, dataclasses.replace(exp, name=f"{exp.name}_r1", **merged_kwargs)


def _build_r2(exp: ExperimentConfig) -> ExperimentConfig:
    """Strategy r2: halve model capacity + lr/10 + tighter clip."""
    return dataclasses.replace(
        exp,
        name=f"{exp.name}_r2",
        hidden_dim=max(16, exp.hidden_dim // 2),
        n_layers=max(1, exp.n_layers // 2),
        n_modes=max(4, exp.n_modes // 2),
        lr=round(exp.lr / 10, 8),
        grad_clip=3.0,
    )


def _build_r3(exp: ExperimentConfig) -> ExperimentConfig:
    """Strategy r3: minimal viable config — last resort before discarding."""
    return dataclasses.replace(
        exp,
        name=f"{exp.name}_r3",
        hidden_dim=32,
        n_layers=2,
        n_modes=min(8, exp.n_modes),
        lr=1e-4,
        grad_clip=1.0,
        batch_size=max(16, min(32, exp.batch_size)),
    )


def _build_adapt(exp: ExperimentConfig, suggestion: dict) -> ExperimentConfig:
    """Build an adaptive retry config from a HypothesisEngine suggestion dict.

    Applies model/arch fields from the suggestion while preserving benchmark,
    budget_s, and other non-arch fields from the original experiment.
    """
    return dataclasses.replace(
        exp,
        name=f"{exp.name}_adapt",
        model=suggestion.get("model", exp.model),
        hidden_dim=suggestion.get("hidden_dim", exp.hidden_dim),
        n_layers=suggestion.get("n_layers", exp.n_layers),
        n_modes=suggestion.get("n_modes", exp.n_modes),
        loss_type=suggestion.get("loss_type", exp.loss_type),
        rationale=suggestion.get("rationale", "HypothesisEngine adaptive retry"),
    )


def poll_injections(pending: list, done_names_set: set) -> list:
    """Read .injected_experiments.json, prepend new experiments to pending, clear the file.

    Called between experiments in the run loop so injected configs are picked up
    without restarting autorun. Returns a (possibly extended) pending list.
    Returns the original list unchanged if the file is absent or malformed.
    """
    if not INJECTIONS_FILE.exists():
        return pending
    try:
        raw = json.loads(INJECTIONS_FILE.read_text())
    except Exception:
        return pending

    if not isinstance(raw, list) or not raw:
        INJECTIONS_FILE.unlink(missing_ok=True)
        return pending

    import dataclasses as _dc
    new_exps = []
    skipped  = []
    for item in raw:
        name = item.get("name", "")
        if not name or name in done_names_set:
            skipped.append(name or "<unnamed>")
            continue
        if any(e.name == name for e in pending):
            skipped.append(name)  # already queued
            continue
        # Build ExperimentConfig from the injected dict
        # Use defaults for fields not supplied by the inject payload
        try:
            bm = item["benchmark"]
            default_budget = BUDGET_FLOOR_2D if bm in BENCHMARKS_2D else BUDGET_FLOOR_1D
            exp = _apply_budget_floor(ExperimentConfig(
                name=name,
                benchmark=bm,
                model=item["model"],
                hidden_dim=int(item.get("hidden_dim", 64)),
                n_layers=int(item.get("n_layers", 4)),
                n_modes=int(item.get("n_modes", 16)),
                budget_s=int(item.get("budget_s", default_budget)),
                priority=int(item.get("priority", 1)),
                rationale=item.get("rationale", "injected via /api/inject"),
            ))
            new_exps.append(exp)
        except Exception as e:
            print(f"  [inject] Skipping malformed entry {name!r}: {e}")

    INJECTIONS_FILE.unlink(missing_ok=True)

    if new_exps:
        names = ", ".join(e.name for e in new_exps)
        print(f"\n  [inject] +{len(new_exps)} experiment(s) from /api/inject: {names}")
        # Prepend so injected experiments run before the rest of the queue
        return new_exps + pending
    if skipped:
        print(f"  [inject] {len(skipped)} injected experiment(s) already done/queued — skipping")
    return pending


def run_experiment(exp: ExperimentConfig, log_path: Path,
                   baseline: float = float("inf")) -> dict:
    """Run one experiment. Returns dict of results.

    baseline: current best val_l2_rel for the benchmark. If a mid-run
    validation line in the log exceeds baseline x EARLY_STOP_MULTIPLIER at
    >=30% progress, the process is terminated early (crash_type='EarlyStop').
    """
    cmd = ["uv", "run", "train.py", *exp.to_cli_args()]

    wprint(f"  CMD: {' '.join(cmd)}")
    wprint(f"  LOG: {log_path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with _active_lock:
        _active_experiments.add(exp.name)
        ACTIVE_FILE.write_text(", ".join(sorted(_active_experiments)))
    try:
        t0 = time.time()
        try:
            with open(log_path, "w") as log_f:
                import os
                env = os.environ.copy()
                env["PYTHONPATH"] = "."
                env["PYTHONUNBUFFERED"] = "1"
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=REPO_ROOT,
                    env=env,
                )
            try:
                kill_file = SENTINEL_DIR / f".kill_{exp.name}"
                deadline = time.time() + TIMEOUT_S
                last_size = 0
                while time.time() < deadline:
                    if kill_file.exists():
                        proc.terminate()
                        proc.wait()
                        kill_file.unlink(missing_ok=True)
                        wprint(f"  KILLED by dashboard request")
                        return {"val": None, "mem_mb": 0.0, "diag": {}, "inspect_id": None,
                                "crash_type": "Killed"}
                    
                    # Watch for budget extensions in the log
                    try:
                        if log_path.exists():
                            curr_size = log_path.stat().st_size
                            if curr_size > last_size:
                                with open(log_path, "r") as f:
                                    f.seek(last_size)
                                    new_content = f.read()
                                    if "[Dynamic Budget]" in new_content:
                                        import re
                                        matches = re.findall(r"Extending budget by (\d+)s to (\d+)s", new_content)
                                        if matches:
                                            ext_s, total_s = map(int, matches[-1])
                                            # Push deadline by the extension amount
                                            deadline += ext_s
                                            wprint(f"  Watchdog extended by {ext_s}s (new budget: {total_s}s)")
                                last_size = curr_size
                    except Exception as e:
                        pass

                    # Early-stop: check mid-run val from trainer log
                    bad_val = check_early_stop_condition(log_path, baseline, EARLY_STOP_MULTIPLIER)
                    if bad_val is not None:
                        proc.terminate()
                        proc.wait()
                        thresh = baseline * EARLY_STOP_MULTIPLIER
                        wprint(f"  EARLY STOP: mid-run val={bad_val:.4f} "
                               f"> {thresh:.4f} ({baseline:.4f}x{EARLY_STOP_MULTIPLIER})")
                        wprint(f"  Saving compute — will retry with adjusted config.")
                        return {"val": None, "mem_mb": 0.0, "diag": {}, "inspect_id": None,
                                "crash_type": "EarlyStop"}
                    if proc.poll() is not None:
                        break
                    time.sleep(2)
                else:
                    proc.kill()
                    proc.wait()
                    wprint(f"  TIMEOUT after {TIMEOUT_S}s — process killed")
                    return {"val": None, "mem_mb": 0.0, "diag": {}, "inspect_id": None}
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                wprint(f"  TIMEOUT after {TIMEOUT_S}s — process killed")
                return {"val": None, "mem_mb": 0.0, "diag": {}, "inspect_id": None}
            elapsed = time.time() - t0
            wprint(f"  Finished in {elapsed:.0f}s  (exit code {proc.returncode})")

            if proc.returncode != 0:
                wprint(f"  Non-zero exit — checking log for details...")
                with open(log_path) as f:
                    tail = f.readlines()[-10:]
                with _print_lock:
                    for line in tail:
                        print(f"    {line}", end="")
                return parse_log_file(log_path)

        except Exception as e:
            wprint(f"  ERROR: {e}")
            return {"val": None, "mem_mb": 0.0, "diag": {}, "inspect_id": None}

        return parse_log_file(log_path)
    finally:
        with _active_lock:
            _active_experiments.discard(exp.name)
            if _active_experiments:
                ACTIVE_FILE.write_text(", ".join(sorted(_active_experiments)))
            else:
                ACTIVE_FILE.unlink(missing_ok=True)


# ── Git integration ───────────────────────────────────────────────────────────

def git_commit_result(exp: ExperimentConfig, val: float) -> None:
    """Stage results.json and create a commit noting the new best result."""
    try:
        subprocess.run(["git", "add", "results.json"], cwd=REPO_ROOT, check=True)
        msg = (f"result: {exp.benchmark} {exp.model} val_l2_rel={val:.6f}\n\n"
               f"{exp.name}: {exp.short()}\n"
               f"Rationale: {exp.rationale}")
        subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, check=True)
        print(f"  Committed results.json")
    except subprocess.CalledProcessError as e:
        print(f"  Git commit failed: {e}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Autonomous SciML experiment runner")
    p.add_argument("--max",         type=int,  default=None,
                   help="Maximum number of experiments to run")
    p.add_argument("--benchmark",   default=None,
                   help="Filter to this benchmark only")
    p.add_argument("--model",       default=None,
                   help="Filter to this model type only")
    p.add_argument("--priority",    type=int, default=10,
                   help="Only run experiments with priority ≤ this value")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print the plan without running anything")
    p.add_argument("--commit",      action="store_true",
                   help="Git-commit results.json after each improved result")
    p.add_argument("--force",       action="store_true",
                   help="Re-run experiments already in results.json")
    p.add_argument("--auto",        action="store_true",
                   help="Fully autonomous mode: run all pending, then invoke "
                        "agent_loop to generate next experiments and loop")
    p.add_argument("--max-auto-experiments", type=int, default=None,
                   help="In --auto mode, stop after N total experiments across all iterations")
    p.add_argument("--max-auto-time",        type=int, default=None,
                   help="In --auto mode, stop after N total wall-clock seconds")
    p.add_argument("--suggest-after", action="store_true",
                   help="Print auto_suggest report after all experiments complete")
    p.add_argument("--workers",     type=int, default=1,
                   help="Number of parallel experiments (default 1). "
                        "Use 2 for 1D-only queues on M1 8GB; keep 1 for 2D experiments.")
    args = p.parse_args()

    # Build queue — load without priority filter so overrides can promote
    # experiments that were originally below the --priority threshold.
    queue = get_experiments(args.benchmark, args.model)
    done  = set() if args.force else load_done_names()

    # Apply dashboard priority overrides (POST /api/priority) before filtering.
    # This ensures an experiment overridden to priority=1 is included even if
    # its experiments.yaml priority was above the --priority threshold.
    _overrides_path = SENTINEL_DIR / ".priority_overrides.json"
    _overrides: dict = {}
    if _overrides_path.exists():
        try:
            _overrides = json.loads(_overrides_path.read_text())
        except Exception as _ov_err:
            print(f"  [priority] Warning: could not read overrides: {_ov_err}")

    if _overrides:
        queue = [
            dataclasses.replace(e, priority=_overrides[e.name])
            if e.name in _overrides else e
            for e in queue
        ]
        print(f"  [priority] Applied {len(_overrides)} dashboard override(s).")

    # Now filter by effective priority and done set, then sort.
    # Apply budget floors so every experiment gets at least 30 min (1D) or 60 min (2D).
    pending = [
        _apply_budget_floor(e) for e in queue
        if e.name not in done and e.priority <= args.priority
    ]
    # Check for plateaued benchmarks and warn (deprioritise by bumping effective priority)
    _all_results = load_results()
    _stuck_benchmarks: set[str] = set()
    for _bm in {e.benchmark for e in pending}:
        if _is_benchmark_stuck(_bm, _all_results):
            _stuck_benchmarks.add(_bm)
            print(f"  [PLATEAU WARNING] {_bm} — last 5 kept results show <2% improvement. "
                  f"Deprioritising in queue.")

    def _effective_priority(e: ExperimentConfig) -> int:
        # Bump priority by 2 for stuck benchmarks so other benchmarks go first
        return e.priority + (2 if e.benchmark in _stuck_benchmarks else 0)

    pending.sort(key=_effective_priority)

    if args.max:
        pending = pending[: args.max]

    baselines = get_baselines()

    # ── Dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\nDry-run: {len(pending)} pending experiments "
              f"(~{len(pending) * 7 / 60:.1f} hours)\n")
        print(f"{'#':>3}  {'P':>2}  {'Name':<30}  {'Config':<45}  "
              f"{'Benchmark'}")
        print("─" * 100)
        for i, e in enumerate(pending, 1):
            print(f"{i:>3}  {e.priority:>2}  {e.name:<30}  {e.short():<45}  "
                  f"{e.benchmark}")
        return

    # ── Run loop ──────────────────────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print(f"  SciML Autorun  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {len(pending)} experiments pending  |  "
          f"~{len(pending) * 7 / 60:.1f} hours estimated")
    print(f"  Baselines: {baselines}")
    print(f"{'━'*70}\n")

    n_improved = 0
    n_crashed  = 0
    workers    = args.workers

    # Warn if workers > 1 with 2D benchmarks
    has_2d = any(e.benchmark in BENCHMARKS_2D for e in pending)
    if workers > 1 and has_2d:
        print(f"  WARNING: --workers {workers} with 2D benchmarks may OOM on 8GB M1.")
        print(f"           2D experiments will be serialised automatically.")

    def run_one(i: int, exp: ExperimentConfig) -> tuple:
        """Run one experiment + retries.  Returns (exp, results, val, mem_gb, status).

        Crash/early-stop retry chain (skipped for ImportError/Killed/incompatible):
          r1  smart_fix()  - context-aware from crash log (lr, batch, modes)
          r2  _build_r2()  - halve hidden_dim/n_layers/n_modes + lr x 0.1
          r3  _build_r3()  - minimal viable: h=32 l=2 m<=8 lr=1e-4
        Discards only after all three fail.

        Poor-result adaptive retry (triggered when completed val > baseline x POOR_RESULT_MULTIPLIER):
          _adapt  _build_adapt()  — HypothesisEngine.suggest_intervention() picks a
                                    better model/arch for the benchmark based on history.
          Skipped for experiments that are themselves retries (_r1/_r2/_r3/_adapt).
        """
        _worker_prefix.name = exp.name  # prefix all wprint() calls with experiment name

        if not args.force and exp.name in load_done_names():
            wprint(f"\n[{i}/{len(pending)}]  already done, skipping")
            return exp, None, None, 0.0, "skip"

        with _print_lock:
            print(f"\n{'─'*70}")
            print(f"[{i}/{len(pending)}]  {exp.name}")
            print(f"  Config   : {exp.short()}")
            print(f"  Benchmark: {exp.benchmark}")
            if exp.rationale:
                print(f"  Rationale: {exp.rationale}")
            with _baselines_lock:
                baseline_val = baselines.get(exp.benchmark, float("inf"))
            print(f"  Current best ({exp.benchmark}): "
                  f"{baseline_val:.6f}" if baseline_val < float("inf")
                  else f"  No baseline yet for {exp.benchmark}")

        log_path = LOGS_DIR / f"{exp.name}.log"

        # ── Trajectory log: record action before run starts ───────────────────
        _traj_state = (
            f"{exp.benchmark} | best={baseline_val:.6f}"
            if baseline_val < float("inf")
            else f"{exp.benchmark} | no baseline yet"
        )
        _write_trajectory({
            "benchmark":        exp.benchmark,
            "model":            exp.model,
            "exp_name":         exp.name,
            "state":            _traj_state,
            "hypothesis":       exp.rationale or "—",
            "action":           f"started {exp.name}: {exp.short()}",
            "expected_outcome": exp.expected or None,
            "outcome":          None,
            "diag_snapshot":    {},
        })

        results  = run_experiment(exp, log_path, baseline=baseline_val)
        val, mem_gb = results["val"], results["mem_mb"] / 1024.0

        # ── Multi-strategy retry loop ─────────────────────────────────────────
        # Skip retries for unfixable failure modes
        _SKIP_RETRY_TYPES = {"ImportError", "NoOutput", "Killed"}
        try:
            _log_lower = log_path.read_text().lower()
        except Exception:
            _log_lower = ""
        _incompatible = "too many values to unpack" in _log_lower

        if val is None and results.get("crash_type") not in _SKIP_RETRY_TYPES and not _incompatible:
            # Build strategy chain: r1 = smart crash fix, r2 = halve model, r3 = minimal
            _strategies: list[tuple[str, ExperimentConfig]] = []

            _fix = smart_fix(exp, log_path, results)
            if _fix is not None:
                _fix_desc, _r1_exp = _fix
                _strategies.append((_fix_desc, _r1_exp))  # already named _r1 by smart_fix

            _strategies.append((
                f"half-model: h{exp.hidden_dim}->{max(16, exp.hidden_dim//2)} "
                f"l{exp.n_layers}->{max(1, exp.n_layers//2)} lr x 0.1",
                _build_r2(exp),
            ))
            _strategies.append((
                "minimal: hidden=32 layers=2 modes≤8 lr=1e-4",
                _build_r3(exp),
            ))

            wprint(f"  {len(_strategies)} retry strategies queued "
                   f"(r1=smart-fix, r2=half-model, r3=minimal)")

            _done_set = load_done_names()
            _retry_tag = ""   # tracks which strategy succeeded (for conclusion field)
            for _attempt_n, (_strat_desc, _retry_exp) in enumerate(_strategies, 1):
                if _retry_exp.name in _done_set:
                    wprint(f"  [r{_attempt_n}] {_retry_exp.name} already in results — skipping")
                    continue

                wprint(f"\n  ── Retry {_attempt_n}/{len(_strategies)} [{_retry_exp.name}]: {_strat_desc}")
                # Add resumption if architecture matches and original might have saved a checkpoint
                if is_arch_compatible(exp, _retry_exp):
                    ckpt_path = REPO_ROOT / "checkpoints" / f"{exp.name}_best.npz"
                    if ckpt_path.exists():
                        _retry_exp.resume_from = exp.name
                        wprint(f"     Resuming from parent checkpoint: {ckpt_path.name}")

                _retry_log = LOGS_DIR / f"{_retry_exp.name}.log"
                with _baselines_lock:
                    _bl_now = baselines.get(exp.benchmark, float("inf"))
                _retry_res = run_experiment(_retry_exp, _retry_log, baseline=_bl_now)

                if _retry_res["val"] is not None:
                    wprint(f"  Retry r{_attempt_n} succeeded: val={_retry_res['val']:.6f}")
                    results   = _retry_res
                    val       = _retry_res["val"]
                    mem_gb    = _retry_res["mem_mb"] / 1024.0
                    _retry_tag = f"[succeeded via r{_attempt_n}: {_strat_desc}]"
                    break
                else:
                    wprint(f"  Retry r{_attempt_n} also failed "
                           f"[{_retry_res.get('crash_type', '?')}]")
            else:
                wprint(f"  All {len(_strategies)} retry strategies exhausted — discarding.")

            # Append retry outcome to results conclusion for DAG inspector
            if _retry_tag:
                results["_retry_tag"] = _retry_tag

        with _baselines_lock:
            baseline_val = baselines.get(exp.benchmark, float("inf"))

        # ── Adaptive retry on poor-but-completed result ───────────────────────
        # If the run finished (no crash) but val >> baseline, the config is not
        # working for this benchmark. Query HypothesisEngine for a smarter config
        # and run one adaptive retry before giving up.
        # Guard: skip if this experiment is itself already a retry/adapt variant.
        _retry_suffixes = ("_r1", "_r2", "_r3", "_adapt", "_retry")
        _is_retry = any(exp.name.endswith(s) for s in _retry_suffixes)
        _is_poor = (
            val is not None
            and baseline_val < float("inf")
            and val > baseline_val * POOR_RESULT_MULTIPLIER
            and not _is_retry
        )
        if _is_poor:
            wprint(
                f"  POOR RESULT: val={val:.4f} > {baseline_val:.4f}x{POOR_RESULT_MULTIPLIER:.0f} "
                f"- querying HypothesisEngine for adaptive config..."
            )
            try:
                from core.hypothesis import HypothesisEngine
                _engine = HypothesisEngine()
                _diag = results.get("diag") or {}
                _suggestion = _engine.suggest_intervention(exp.benchmark, val, _diag)
                _adapt_exp = _build_adapt(exp, _suggestion)
                _done_set = load_done_names()
                if _adapt_exp.name in _done_set:
                    wprint(f"  [adapt] {_adapt_exp.name} already in results — skipping")
                else:
                    wprint(
                        f"  ── Adaptive retry [{_adapt_exp.name}]: "
                        f"{_adapt_exp.model} h={_adapt_exp.hidden_dim} "
                        f"l={_adapt_exp.n_layers} m={_adapt_exp.n_modes}"
                    )
                    # Add resumption if architecture matches
                    if is_arch_compatible(exp, _adapt_exp):
                        ckpt_path = REPO_ROOT / "checkpoints" / f"{exp.name}_best.npz"
                        if ckpt_path.exists():
                            _adapt_exp.resume_from = exp.name
                            wprint(f"     Resuming from original checkpoint: {ckpt_path.name}")

                    wprint(f"     {_suggestion.get('rationale', '')[:140]}")
                    _adapt_log = LOGS_DIR / f"{_adapt_exp.name}.log"
                    with _baselines_lock:
                        _bl_now = baselines.get(exp.benchmark, float("inf"))
                    _adapt_res = run_experiment(_adapt_exp, _adapt_log, baseline=_bl_now)
                    _adapt_val = _adapt_res.get("val")
                    if _adapt_val is not None:
                        wprint(f"  Adaptive retry: val={_adapt_val:.6f}  "
                               f"({'improved' if _adapt_val < val else 'no improvement vs original'})")
                        if _adapt_val < val:
                            results = _adapt_res
                            val     = _adapt_val
                            mem_gb  = _adapt_res["mem_mb"] / 1024.0
                    else:
                        wprint(f"  Adaptive retry crashed [{_adapt_res.get('crash_type', '?')}]")
            except Exception as _adapt_err:
                wprint(f"  [adapt] HypothesisEngine error: {_adapt_err}")
            # Re-read baseline after adaptive retry (may have been updated by parallel workers)
            with _baselines_lock:
                baseline_val = baselines.get(exp.benchmark, float("inf"))

        _critique = ""
        if val is None:
            status = "crash"
            crash_type = results.get("crash_type", "Unknown")
            wprint(f"  RESULT: CRASH  [{crash_type}]")
            _critique = f"Architecture discovery halted by {crash_type}."
        else:
            improved = val < baseline_val
            status   = "keep" if improved else "discard"
            delta    = (baseline_val - val) / baseline_val * 100 if baseline_val < float("inf") else 0
            marker   = f"↑ NEW BEST  (+{delta:.1f}%)" if improved else f"↓ no improvement"
            wprint(f"  RESULT: val_l2_rel = {val:.6f}   {marker}")

            try:
                from core.hypothesis import HypothesisEngine
                _reflection = HypothesisEngine().suggest_intervention(exp.benchmark, val, results.get("diag", {}))
                _critique = _reflection.get("rationale", "No specific critique.")
                if "paper_ref" in _reflection:
                    _critique += f" [Grounding: {_reflection['paper_ref']}]"
                wprint(f"  CRITIQUE: {_critique[:120]}...")
            except Exception as _refl_err:
                _critique = f"Reflection error: {_refl_err}"

            if improved:
                with _baselines_lock:
                    baselines[exp.benchmark] = val

        commit = current_commit()
        parent_name = getattr(exp, "parent_name", "") or ""
        with _tracker_lock:
            tracker.log_experiment(
                benchmark=exp.benchmark,
                model=exp.model,
                val_l2_rel=val if val is not None else 1.0,
                memory_gb=mem_gb,
                status=status,
                description=f"{exp.name} {exp.short()}",
                commit=commit,
                parent_name=parent_name or None,
                config=vars(exp),
                rationale=exp.rationale,
                conclusion=(
                    (f"crash:{results['crash_type']} " if results.get("crash_type") else "")
                    + (results.get("inspect_id") or "")
                    + (" " + results.get("_retry_tag", "") if results.get("_retry_tag") else "")
                    + f" | Critique: {_critique}"
                ),
                diag=results.get("diag", {}),
            )
        wprint(f"  Logged to results.json [status={status}]")

        # ── Trajectory log: record outcome after run completes ────────────────
        _write_trajectory({
            "benchmark":        exp.benchmark,
            "model":            exp.model,
            "exp_name":         exp.name,
            "state":            _traj_state,
            "hypothesis":       exp.rationale or "—",
            "action":           f"completed {exp.name}: {exp.short()}",
            "expected_outcome": exp.expected or None,
            "critique":         _critique,
            "outcome": (
                f"val_l2_rel={val:.6f} status={status}"
                if val is not None
                else f"crash:{results.get('crash_type', 'Unknown')}"
            ),
            "diag_snapshot":    results.get("diag", {}),
        })

        if args.commit and status == "keep":
            git_commit_result(exp, val)

        # ── Brain Distillation: Evolve RESEARCH_BRAIN.md ──────────────────────
        try:
            from core.brain_distiller import distill
            distill()
        except Exception as _distill_err:
            wprint(f"  [brain] Distillation failed: {_distill_err}")

        return exp, results, val, mem_gb, status

    # ── Memory-aware parallel scheduler ──────────────────────────────────────
    # 2D experiments are serialised (they use too much memory to overlap safely)
    # 1D experiments can run up to `workers` at a time
    mem_sem     = threading.Semaphore(workers)

    def mem_aware_run(i: int, exp: ExperimentConfig):
        est = estimate_memory_mb(exp)
        # For 2D: always serialise by acquiring all slots
        slots = workers if exp.benchmark in BENCHMARKS_2D else 1
        for _ in range(slots):
            mem_sem.acquire()
        try:
            return run_one(i, exp)
        finally:
            for _ in range(slots):
                mem_sem.release()

    if workers == 1:
        # Simple sequential path — no thread overhead
        # Poll for injected experiments between each run
        i = 1
        while i <= len(pending):
            exp = pending[i - 1]
            _, _, _, _, status = run_one(i, exp)
            if status == "crash":
                n_crashed += 1
            elif status == "keep":
                n_improved += 1
            
            # Check for pause
            while PAUSE_FILE.exists():
                print("\n  [pause] .autorun_pause found — waiting…", end="\r")
                time.sleep(5)
            
            # Check for newly injected experiments before moving on
            pending = poll_injections(pending, load_done_names())
            i += 1
    else:
        print(f"\n  Parallel mode: up to {workers} workers  "
              f"(2D experiments serialised for memory safety)")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(mem_aware_run, i, exp): exp
                       for i, exp in enumerate(pending, 1)}
            for fut in as_completed(futures):
                _, _, _, _, status = fut.result()
                if status == "crash":
                    n_crashed += 1
                elif status == "keep":
                    n_improved += 1
                # Pick up injected experiments and submit them to the pool
                injected = poll_injections([], load_done_names())
                for new_exp in injected:
                    idx = len(futures) + 1
                    futures[pool.submit(mem_aware_run, idx, new_exp)] = new_exp

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print(f"Autorun complete: {len(pending)} run  |  "
          f"{n_improved} improved  |  {n_crashed} crashed")
    print(f"Final baselines:")
    for bm, val in sorted(baselines.items()):
        print(f"  {bm:<25}  {val:.6f}")
    print(f"{'━'*70}\n")

    # ── Auto-suggest next steps ───────────────────────────────────────────────
    if args.suggest_after or args.auto:
        print("\n─── auto_suggest.py output ───")
        try:
            result = subprocess.run(
                ["uv", "run", "auto_suggest.py", "--top", "6"]
                + (["--benchmark", args.benchmark] if args.benchmark else []),
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=60
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr[:500])
        except Exception as e:
            print(f"  auto_suggest failed: {e}")

    # ── Autonomous loop (iterative — no recursion) ────────────────────────────
    if not args.auto:
        return

    import time as _time

    auto_start   = _time.time()
    auto_exp_cnt = len(pending)
    idle_rounds  = 0          # consecutive rounds with no new experiments run
    MAX_IDLE     = 3          # stop after 3 idle rounds (all HPO suggestions exhausted)

    while True:
        # ── Time / experiment count guards ───────────────────────────────────
        if args.max_auto_time:
            elapsed = _time.time() - auto_start
            if elapsed >= args.max_auto_time:
                print(f"\n  [auto] --max-auto-time {args.max_auto_time}s reached "
                      f"({elapsed:.0f}s elapsed). Stopping.")
                break

        if args.max_auto_experiments and auto_exp_cnt >= args.max_auto_experiments:
            print(f"\n  [auto] --max-auto-experiments {args.max_auto_experiments} reached "
                  f"({auto_exp_cnt} run). Stopping.")
            break

        # ── Check for pending work in experiments.yaml ───────────────────────
        # Re-import so experiments added by auto_suggest --generate are picked up
        from importlib import reload as _reload
        import core.loader as _loader_mod
        _reload(_loader_mod)

        _done_now = load_done_names()
        _remaining = [
            _apply_budget_floor(e)
            for e in _loader_mod.get_experiments(args.benchmark, args.model)
            if e.name not in _done_now and e.priority <= args.priority
        ]

        # Apply plateau-aware priority bump (re-use logic from main)
        _all_res = load_results()
        _stuck_bms: set = set()
        for _bm in {e.benchmark for e in _remaining}:
            if _is_benchmark_stuck(_bm, _all_res):
                _stuck_bms.add(_bm)
        _remaining.sort(key=lambda e: e.priority + (2 if e.benchmark in _stuck_bms else 0))

        if _remaining:
            idle_rounds = 0
            print(f"\n  [auto] {len(_remaining)} experiments pending — running next batch…")
            # Run the next batch (up to --max experiments if set)
            batch = _remaining
            if args.max:
                remaining_quota = args.max - auto_exp_cnt
                if remaining_quota <= 0:
                    print(f"  [auto] --max {args.max} experiments reached. Stopping.")
                    break
                batch = batch[:remaining_quota]

            for _i, _exp in enumerate(batch, 1):
                _worker_prefix.name = _exp.name
                _, _, _, _, _status = run_one(_i, _exp)
                auto_exp_cnt += 1
                if _status == "crash":
                    n_crashed += 1
                elif _status == "keep":
                    n_improved += 1

                # Honour pause between experiments
                while PAUSE_FILE.exists():
                    print("\n  [pause] .autorun_pause found — waiting…", end="\r")
                    time.sleep(5)

                # Pick up any dashboard-injected experiments
                batch = poll_injections(batch, load_done_names())

            continue  # loop back to check for more pending

        # ── Queue exhausted: generate new experiments via auto_suggest ────────
        print(f"\n  [auto] Queue exhausted (idle_rounds={idle_rounds}) — "
              f"running auto_suggest --generate…")
        _generated = 0
        try:
            _suggest_cmd = (
                ["uv", "run", "auto_suggest.py", "--generate", "--write-yaml", "--top", "5"]
                + (["--benchmark", args.benchmark] if args.benchmark else [])
            )
            _res = subprocess.run(
                _suggest_cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60
            )
            if _res.stdout:
                print(_res.stdout[:2000])
            # Count how many lines were added to experiments.yaml
            _new_remaining = [
                e for e in _loader_mod.get_experiments(args.benchmark, args.model)
                if e.name not in load_done_names() and e.priority <= args.priority
            ]
            _generated = len(_new_remaining)
        except Exception as _sg_err:
            print(f"  auto_suggest --generate failed: {_sg_err}")

        if _generated > 0:
            idle_rounds = 0
            print(f"  [auto] auto_suggest generated {_generated} new experiments — continuing…")
            continue

        # ── HPO fallback: Bayesian optimisation for highest-gap benchmark ─────
        _hpo_bm = args.benchmark or _pick_highest_gap_benchmark()
        print(f"\n  [auto] auto_suggest generated nothing — "
              f"falling back to BayesianHPO for {_hpo_bm}…")
        _hpo_ran = _run_hpo_batch(_hpo_bm, args)
        if _hpo_ran:
            idle_rounds = 0
            auto_exp_cnt += _hpo_ran
            continue

        idle_rounds += 1
        if idle_rounds >= MAX_IDLE:
            print(f"\n  [auto] {MAX_IDLE} consecutive idle rounds — no experiments to generate. "
                  f"Stopping autonomous loop.")
            print("  → Run manually: uv run auto_suggest.py --generate")
            break

        print(f"  [auto] Idle round {idle_rounds}/{MAX_IDLE} — sleeping 60s before retrying…")
        time.sleep(60)


if __name__ == "__main__":
    main()
