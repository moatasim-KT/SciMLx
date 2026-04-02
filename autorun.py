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
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from experiments import ExperimentConfig, get_experiments

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).parent
RESULTS_FILE = REPO_ROOT / "results.tsv"
LOGS_DIR     = REPO_ROOT / "logs"
TIMEOUT_S    = 720   # 12 min per experiment (5-min budget + data/compile overhead)

TSV_HEADER   = ["commit", "benchmark", "model", "val_l2_rel",
                "memory_gb", "status", "description"]


# ── Results I/O ───────────────────────────────────────────────────────────────

def load_done_names() -> set[str]:
    """Return set of experiment names already in results.tsv description field."""
    done: set[str] = set()
    if not RESULTS_FILE.exists():
        return done
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            desc = row.get("description", "")
            # Experiment names are stored as the first token in description
            done.add(desc.split()[0] if desc else "")
    return done


def get_baselines() -> dict[str, float]:
    """Return current best val_l2_rel per benchmark from results.tsv."""
    best: dict[str, float] = {}
    if not RESULTS_FILE.exists():
        return best
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("status") in ("keep",):
                bm  = row.get("benchmark", "")
                val = float(row.get("val_l2_rel", "1e9"))
                if bm not in best or val < best[bm]:
                    best[bm] = val
    return best


def append_result(exp: ExperimentConfig, val_l2_rel: Optional[float],
                  status: str, memory_gb: float, commit: str) -> None:
    """Append one row to results.tsv."""
    val_str = f"{val_l2_rel:.6f}" if val_l2_rel is not None else "N/A"
    row     = [commit, exp.benchmark, exp.model, val_str,
               f"{memory_gb:.2f}", status,
               f"{exp.name} {exp.short()}"]

    write_header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a") as f:
        if write_header:
            f.write("\t".join(TSV_HEADER) + "\n")
        f.write("\t".join(row) + "\n")


# ── Subprocess helpers ────────────────────────────────────────────────────────

def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def parse_log(log_path: Path) -> tuple[Optional[float], float]:
    """Extract (val_l2_rel, memory_gb) from a train.py log file.
    Returns (None, 0.0) if the run crashed.
    """
    val    = None
    mem_mb = 0.0
    try:
        with open(log_path) as f:
            for line in f:
                if line.startswith("val_l2_rel:"):
                    val = float(line.split(":")[1].strip())
                elif line.startswith("peak_vram_mb:"):
                    mem_mb = float(line.split(":")[1].strip())
    except Exception:
        pass
    return val, mem_mb / 1024.0


def run_experiment(exp: ExperimentConfig, log_path: Path) -> tuple[Optional[float], float]:
    """Run one experiment. Returns (val_l2_rel or None, memory_gb)."""
    cmd = ["uv", "run", "train.py"] + exp.to_cli_args()

    print(f"  CMD: {' '.join(cmd)}")
    print(f"  LOG: {log_path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
                timeout=TIMEOUT_S,
            )
        elapsed = time.time() - t0
        print(f"  Finished in {elapsed:.0f}s  (exit code {proc.returncode})")

        if proc.returncode != 0:
            print(f"  Non-zero exit — checking log for details...")
            # Print last 10 lines of log for quick diagnosis
            with open(log_path) as f:
                tail = f.readlines()[-10:]
            for line in tail:
                print(f"    {line}", end="")
            return None, 0.0

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {TIMEOUT_S}s")
        return None, 0.0
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, 0.0

    return parse_log(log_path)


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

    for i, exp in enumerate(pending, 1):
        # Re-read results.tsv before each run — guards against duplicate runs
        # when multiple sessions overlap or a previous session was interrupted
        # mid-write.
        if not args.force and exp.name in load_done_names():
            print(f"\n[{i}/{len(pending)}]  {exp.name}  — already in results.tsv, skipping")
            continue

        print(f"\n{'─'*70}")
        print(f"[{i}/{len(pending)}]  {exp.name}")
        print(f"  Config   : {exp.short()}")
        print(f"  Benchmark: {exp.benchmark}")
        if exp.rationale:
            print(f"  Rationale: {exp.rationale}")
        baseline_val = baselines.get(exp.benchmark, float("inf"))
        print(f"  Current best ({exp.benchmark}): "
              f"{baseline_val:.6f}" if baseline_val < float("inf")
              else f"  No baseline yet for {exp.benchmark}")

        log_path = LOGS_DIR / f"{exp.name}.log"
        val, mem_gb = run_experiment(exp, log_path)

        if val is None:
            status = "crash"
            n_crashed += 1
            print(f"  RESULT: CRASH")
        else:
            improved = val < baseline_val
            status   = "keep" if improved else "discard"
            delta    = (baseline_val - val) / baseline_val * 100 if baseline_val < float("inf") else 0
            marker   = f"↑ NEW BEST  (+{delta:.1f}%)" if improved else f"↓ no improvement"
            print(f"  RESULT: val_l2_rel = {val:.6f}   {marker}")

            if improved:
                baselines[exp.benchmark] = val
                n_improved += 1

        commit = current_commit()
        append_result(exp, val, status, mem_gb, commit)
        print(f"  Logged to results.tsv  [status={status}]")

        if args.commit and status == "keep":
            git_commit_result(exp, val)

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
