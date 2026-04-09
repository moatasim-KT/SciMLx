"""Autonomous SciML experiment runner.

Iterates through the experiment queue in experiments.py, skips anything
already recorded in results.tsv, runs each via subprocess, and logs results.

Usage:
    uv run autorun.py                          # run all pending experiments
    uv run autorun.py --max 5                  # stop after 5 experiments
    uv run autorun.py --benchmark burgers_1d   # filter by benchmark
    uv run autorun.py --model FNO              # filter by model type
    uv run autorun.py --priority 3             # only experiments with priority ≤ 3
    uv run autorun.py --dry-run                # print plan without running
    uv run autorun.py --commit                 # git-commit good results automatically

Logs are written to logs/<name>.log.
results.tsv is updated after every experiment.
"""

import argparse
import dataclasses
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from experiments import ExperimentConfig, get_experiments
from utils import REPO_ROOT, RESULTS_FILE, LOGS_DIR, load_results, done_names, best_per_benchmark
from tracker import Tracker

TIMEOUT_S = 1500  # 25 min per experiment (20-min budget + data/compile overhead)

# Memory budget for parallel scheduling (M1 8GB unified memory)
TOTAL_MEMORY_MB = 7500  # leave ~500MB headroom
# Estimated memory per experiment type (conservative)
MEMORY_ESTIMATE_1D_MB = 2500   # burgers, kdv, wave, euler
MEMORY_ESTIMATE_2D_MB = 5000   # darcy, ns, swe, allen_cahn, ns_hre

BENCHMARKS_2D = {"darcy_2d_fix", "ns_2d_fix", "swe_2d", "allen_cahn_2d", "ns_hre_2d", "darcy_2d"}

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


def parse_log(log_path: Path) -> dict:
    """Extract metrics, diagnostics, and crash type from a train.py log file."""
    results = {
        "val": None,
        "mem_mb": 0.0,
        "diag": {},
        "inspect_id": None,
        "crash_type": None,
    }
    try:
        content = log_path.read_text()
        for line in content.splitlines():
            if line.startswith("val_l2_rel:"):
                results["val"] = float(line.split(":")[1].strip())
            elif line.startswith("peak_vram_mb:"):
                results["mem_mb"] = float(line.split(":")[1].strip())
            elif line.startswith("diag_"):
                key = line.split(":")[0].strip()
                val = float(line.split(":")[1].strip())
                results["diag"][key] = val
            elif line.startswith("inspect_id:"):
                results["inspect_id"] = line.split(":", 1)[1].strip()

        # Classify crash type if no val_l2_rel found
        if results["val"] is None:
            lower = content.lower()
            if "vram limit exceeded" in lower:
                results["crash_type"] = "VRAMLimit"
            elif "out of memory" in lower or ("memory" in lower and "error" in lower):
                results["crash_type"] = "OOM"
            elif "nan" in lower or "inf" in lower or "diverged" in lower:
                results["crash_type"] = "NaN/Inf"
            elif "importerror" in lower or "modulenotfounderror" in lower:
                results["crash_type"] = "ImportError"
            elif "valueerror" in lower or "assertionerror" in lower or "runtimeerror" in lower:
                results["crash_type"] = "ValueError"
            elif "timeout" in lower or "timed out" in lower:
                results["crash_type"] = "Timeout"
            elif "traceback" in lower or "error" in lower:
                results["crash_type"] = "UnknownError"
            else:
                results["crash_type"] = "NoOutput"
    except Exception:
        pass
    return results


def smart_fix(exp: ExperimentConfig, log_path: Path, results: dict) -> Optional[tuple]:
    """Analyse crash log and return (fix_description, fixed_ExperimentConfig) or None.

    Priority order of diagnoses:
      1. 1D model on 2D data (shape unpack)  → unfixable, skip
      2. Modes broadcast mismatch            → halve n_modes
      3. OOM                                 → halve batch_size
      4. NaN / Inf                           → lr÷10, grad_clip→5
      5. Timeout                             → halve hidden_dim + n_layers
      6. Generic ValueError / RuntimeError   → halve n_modes first
      7. Unknown                             → halve hidden_dim + n_layers
    """
    try:
        content = log_path.read_text()
    except Exception:
        content = ""
    lower = content.lower()

    def _replace(**kw):
        return dataclasses.replace(exp, name=f"{exp.name}_retry", **kw)

    # ── 1. 1D model receiving 2D input ───────────────────────────────────────
    if "too many values to unpack" in lower:
        wprint("  DIAGNOSIS: 1D model given 2D input — incompatible pairing, skipping.")
        return None

    # ── 2. Fourier modes too wide for grid ───────────────────────────────────
    if "broadcast_shapes" in lower or (
        "cannot be broadcast" in lower and ("modes" in lower or "shapes" in lower)
    ):
        new_modes = max(4, exp.n_modes // 2)
        desc = f"modes {exp.n_modes}→{new_modes} (broadcast shape error)"
        wprint(f"  DIAGNOSIS: modes too large for grid. Fix: {desc}")
        return desc, _replace(n_modes=new_modes)

    # ── 3a. VRAM soft limit exceeded (our own guard) ─────────────────────────
    if "vram limit exceeded" in lower:
        new_hidden = max(32, exp.hidden_dim // 2)
        new_layers = max(2, exp.n_layers // 2)
        new_batch  = max(8, exp.batch_size // 2)
        desc = (f"h {exp.hidden_dim}→{new_hidden}, l {exp.n_layers}→{new_layers}, "
                f"batch {exp.batch_size}→{new_batch} (VRAM limit)")
        wprint(f"  DIAGNOSIS: VRAM limit exceeded. Fix: {desc}")
        return desc, _replace(hidden_dim=new_hidden, n_layers=new_layers, batch_size=new_batch)

    # ── 3b. Out of memory (Metal OOM) ────────────────────────────────────────
    if "out of memory" in lower or ("memory" in lower and "alloc" in lower):
        new_batch = max(8, exp.batch_size // 2)
        desc = f"batch_size {exp.batch_size}→{new_batch} (OOM — preserve hidden_dim)"
        wprint(f"  DIAGNOSIS: OOM. Fix: {desc}")
        return desc, _replace(batch_size=new_batch)

    # ── 4. NaN / Inf divergence ──────────────────────────────────────────────
    if "nan" in lower or ("inf" in lower and "loss" in lower):
        new_lr = round(exp.lr / 10, 8)
        desc = f"lr {exp.lr:.1e}→{new_lr:.1e}, grad_clip 1.0→5.0 (NaN/Inf)"
        wprint(f"  DIAGNOSIS: numerical instability. Fix: {desc}")
        return desc, _replace(lr=new_lr, grad_clip=5.0)

    # ── 5. Timeout (ran too long, no result) ─────────────────────────────────
    if results.get("crash_type") == "Timeout":
        new_hidden = max(32, exp.hidden_dim // 2)
        new_layers = max(2, exp.n_layers // 2)
        desc = f"h {exp.hidden_dim}→{new_hidden}, l {exp.n_layers}→{new_layers} (timeout — smaller model)"
        wprint(f"  DIAGNOSIS: timeout. Fix: {desc}")
        return desc, _replace(hidden_dim=new_hidden, n_layers=new_layers)

    # ── 6. ValueError / RuntimeError — try modes first ───────────────────────
    if "valueerror" in lower or "runtimeerror" in lower or "assertionerror" in lower:
        for line in reversed(content.splitlines()):
            if "error:" in line.lower() or "Error" in line:
                wprint(f"  DIAGNOSIS: {line.strip()}")
                break
        new_modes = max(4, exp.n_modes // 2)
        desc = f"modes {exp.n_modes}→{new_modes} (ValueError — reduce spectral width)"
        wprint(f"  Fix: {desc}")
        return desc, _replace(n_modes=new_modes)

    # ── 7. Unknown — generic size reduction ──────────────────────────────────
    new_hidden = max(32, exp.hidden_dim // 2)
    new_layers = max(2, exp.n_layers // 2)
    desc = f"h {exp.hidden_dim}→{new_hidden}, l {exp.n_layers}→{new_layers} (unknown error)"
    wprint(f"  DIAGNOSIS: unknown crash. Fix: {desc}")
    return desc, _replace(hidden_dim=new_hidden, n_layers=new_layers)


def run_experiment(exp: ExperimentConfig, log_path: Path) -> dict:
    """Run one experiment. Returns dict of results."""
    cmd = ["uv", "run", "train.py"] + exp.to_cli_args()

    wprint(f"  CMD: {' '.join(cmd)}")
    wprint(f"  LOG: {log_path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    active_file = REPO_ROOT / ".active_experiment"
    with _active_lock:
        _active_experiments.add(exp.name)
        active_file.write_text(", ".join(sorted(_active_experiments)))
    try:
        t0 = time.time()
        try:
            with open(log_path, "w") as log_f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=REPO_ROOT,
                )
            try:
                proc.wait(timeout=TIMEOUT_S)
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
                return {"val": None, "mem_mb": 0.0, "diag": {}, "inspect_id": None}

        except Exception as e:
            wprint(f"  ERROR: {e}")
            return {"val": None, "mem_mb": 0.0, "diag": {}, "inspect_id": None}

        return parse_log(log_path)
    finally:
        with _active_lock:
            _active_experiments.discard(exp.name)
            if _active_experiments:
                active_file.write_text(", ".join(sorted(_active_experiments)))
            else:
                active_file.unlink(missing_ok=True)


# ── Git integration ───────────────────────────────────────────────────────────

def git_commit_result(exp: ExperimentConfig, val: float) -> None:
    """Stage results.tsv and create a commit noting the new best result."""
    try:
        subprocess.run(["git", "add", "results.tsv"], cwd=REPO_ROOT, check=True)
        msg = (f"result: {exp.benchmark} {exp.model} val_l2_rel={val:.6f}\n\n"
               f"{exp.name}: {exp.short()}\n"
               f"Rationale: {exp.rationale}")
        subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, check=True)
        print(f"  Committed results.tsv")
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
                   help="Git-commit results.tsv after each improved result")
    p.add_argument("--force",       action="store_true",
                   help="Re-run experiments already in results.tsv")
    p.add_argument("--auto",        action="store_true",
                   help="Fully autonomous mode: run all pending, then call "
                        "auto_suggest.py to print next steps and loop indefinitely")
    p.add_argument("--suggest-after", action="store_true",
                   help="Print auto_suggest report after all experiments complete")
    p.add_argument("--workers",     type=int, default=1,
                   help="Number of parallel experiments (default 1). "
                        "Use 2 for 1D-only queues on M1 8GB; keep 1 for 2D experiments.")
    args = p.parse_args()

    # Build queue
    queue   = get_experiments(args.benchmark, args.model, args.priority)
    done    = set() if args.force else load_done_names()
    pending = [e for e in queue if e.name not in done]

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
        """Run one experiment + optional retry. Returns (exp, results, val, mem_gb, status)."""
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
        results  = run_experiment(exp, log_path)
        val, mem_gb = results["val"], results["mem_mb"] / 1024.0

        # ── Smart crash recovery ──────────────────────────────────────────────
        if val is None and results.get("crash_type") not in ("ImportError", "NoOutput"):
            fix = smart_fix(exp, log_path, results)
            if fix is None:
                wprint("  No auto-fix available — skipping retry.")
            else:
                fix_desc, retry_exp = fix
                wprint(f"  Retrying with targeted fix: {fix_desc}")
                retry_log = LOGS_DIR / f"{retry_exp.name}.log"
                retry_res = run_experiment(retry_exp, retry_log)
                if retry_res["val"] is not None:
                    wprint(f"  Retry succeeded: val_l2_rel={retry_res['val']:.6f}")
                    results = retry_res
                    val = retry_res["val"]
                    mem_gb = retry_res["mem_mb"] / 1024.0

        with _baselines_lock:
            baseline_val = baselines.get(exp.benchmark, float("inf"))

        if val is None:
            status = "crash"
            crash_type = results.get("crash_type", "Unknown")
            wprint(f"  RESULT: CRASH  [{crash_type}]")
        else:
            improved = val < baseline_val
            status   = "keep" if improved else "discard"
            delta    = (baseline_val - val) / baseline_val * 100 if baseline_val < float("inf") else 0
            marker   = f"↑ NEW BEST  (+{delta:.1f}%)" if improved else f"↓ no improvement"
            wprint(f"  RESULT: val_l2_rel = {val:.6f}   {marker}")
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
                conclusion=(f"crash:{results['crash_type']} " if results.get("crash_type") else "") + (results.get("inspect_id") or ""),
                diag=results.get("diag", {}),
            )
        wprint(f"  Logged to results.json and results.tsv  [status={status}]")

        if args.commit and status == "keep":
            git_commit_result(exp, val)

        return exp, results, val, mem_gb, status

    # ── Memory-aware parallel scheduler ──────────────────────────────────────
    # 2D experiments are serialised (they use too much memory to overlap safely)
    # 1D experiments can run up to `workers` at a time
    mem_budget  = TOTAL_MEMORY_MB
    mem_in_use  = 0
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
        for i, exp in enumerate(pending, 1):
            _, results, val, mem_gb, status = run_one(i, exp)
            if status == "crash":
                n_crashed += 1
            elif status == "keep":
                n_improved += 1
    else:
        print(f"\n  Parallel mode: up to {workers} workers  "
              f"(2D experiments serialised for memory safety)")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(mem_aware_run, i, exp): exp
                       for i, exp in enumerate(pending, 1)}
            for fut in as_completed(futures):
                _, results, val, mem_gb, status = fut.result()
                if status == "crash":
                    n_crashed += 1
                elif status == "keep":
                    n_improved += 1

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
                ["uv", "run", "auto_suggest.py",
                 "--benchmark", args.benchmark or "burgers_1d",
                 "--top", "6"],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr[:500])
        except Exception as e:
            print(f"  auto_suggest failed: {e}")

    # ── Autonomous loop ───────────────────────────────────────────────────────
    if args.auto:
        # Check if there are still pending experiments in the queue
        remaining = [e for e in get_experiments(args.benchmark, args.model, args.priority)
                     if e.name not in load_done_names()]
        if remaining:
            print(f"\n  {len(remaining)} experiments still in queue — continuing…")
            # Recurse by re-entering main (restart the loop)
            import sys
            # Pass same flags but without --auto to avoid infinite recursion on crash
            new_argv = [a for a in sys.argv[1:] if a != "--auto"] + ["--auto"]
            sys.argv[1:] = new_argv
            main()
        else:
            print("\n  Queue exhausted. Autonomous loop complete.")
            print("  → Add new experiments via 'uv run auto_suggest.py --generate'")
            print("  → Or manually edit experiments.py and re-run autorun.py --auto")


if __name__ == "__main__":
    main()
