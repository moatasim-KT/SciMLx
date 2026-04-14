"""Autonomous Agent Orchestration Loop for SciML Research.

This is ONE of TWO supported orchestration modes — choose whichever fits your
workflow:

  MODE A · External agent (program.md)          ← original, always supported
  ─────────────────────────────────────────────
  An external AI agent (Claude Code, GPT-4, etc.) reads program.md and drives
  the research loop manually: edits train.py / experiments.py, calls autorun.py,
  interprets results, forms hypotheses, and commits improvements.

  Best for: interactive research, novel architecture ideas, steering by
  intuition, or when you want full human-visible control over every decision.

  MODE B · agent_loop.py (this file)             ← optional in-process loop
  ─────────────────────────────────────────────
  A fully automated in-process loop that replaces the need for an external
  agent to interpret results. It reads tracker.analyze_lineage(), calls
  HypothesisEngine, runs Bayesian HPO, and appends new ExperimentConfigs to
  experiments.py — all without human intervention.

  Best for: overnight runs, saturating the queue automatically after an
  external agent session, or scaling up experiment throughput.

Both modes share the same infrastructure (experiments.py queue, results.json,
autorun.py runner) and can be used interchangeably or together.

Steps performed in Mode B:
  1. Analyses current state via tracker.analyze_lineage()
  2. Identifies failure patterns via hypothesis.HypothesisEngine
  3. Generates next experiments via auto_suggest + bayesian_hpo
  4. Writes new ExperimentConfig entries to experiments.py (gated)
  5. Optionally triggers autorun.py for the next batch

Usage:
    uv run agent_loop.py --dry-run          # plan without writing
    uv run agent_loop.py --benchmark kdv_1d # focus on one benchmark
    uv run agent_loop.py --run              # generate + immediately run top-3
    uv run agent_loop.py --top 5            # generate top-N new configs

The loop checks .autorun_pause before each action and respects it.
See program.md for the external-agent (Mode A) workflow.
"""

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

from core.utils import REPO_ROOT

# ── Imports (lazy to avoid MLX startup cost when just planning) ───────────────

def _load_tracker():
    from core.tracker import Tracker
    return Tracker()

def _load_hypothesis():
    from core.hypothesis import HypothesisEngine
    return HypothesisEngine()

def _load_hpo(benchmark: str, model: str = "FNO"):
    from core.hpo import BayesianHPO
    hpo = BayesianHPO(benchmark, model)
    hpo.load_history()
    return hpo

# ── State checks ──────────────────────────────────────────────────────────────

def is_paused() -> bool:
    return (REPO_ROOT / ".autorun_pause").exists()


def current_sota_gaps() -> dict[str, float]:
    """Return ratio (our_best / sota) for each benchmark. <1 means we beat SOTA."""
    from core.utils import SOTA, load_results, best_per_benchmark
    rows = load_results()
    best = best_per_benchmark(rows)
    gaps = {}
    for bm, sota in SOTA.items():
        our = best.get(bm)
        if our:
            gaps[bm] = round(our / sota, 3)
    return gaps


def pending_count() -> int:
    from core.loader import get_experiments
    from core.utils import done_names
    done = done_names()
    return sum(1 for e in get_experiments() if e.name not in done)

# ── Core analysis ─────────────────────────────────────────────────────────────

def analyse_state(benchmark: Optional[str] = None) -> dict:
    """Full state analysis: lineage + hypothesis + SOTA gaps."""
    tracker   = _load_tracker()
    engine    = _load_hypothesis()
    analysis  = tracker.analyze_lineage(benchmark)
    gaps      = current_sota_gaps()
    pending   = pending_count()

    # Identify highest-priority benchmarks (farthest from SOTA with fewest runs)
    summaries = analysis.get("benchmark_summaries", {})
    priority_queue = []
    for bm, s in summaries.items():
        gap   = gaps.get(bm, float("inf"))
        runs  = s.get("n_experiments", 0)
        score = gap * max(1, 10 - runs)   # high gap + few runs = highest priority
        priority_queue.append((score, bm, gap, runs))
    priority_queue.sort(reverse=True)

    # Hypothesis-driven next steps for each benchmark
    interventions = {}
    for _, bm, gap, runs in priority_queue[:5]:
        bm_report = engine.analyze_benchmark(bm)
        best_val  = bm_report.get("best_val", float("inf"))
        interv    = engine.suggest_intervention(bm, best_val)
        if interv:
            interventions[bm] = interv

    return {
        "analysis":         analysis,
        "sota_gaps":        gaps,
        "pending_experiments": pending,
        "priority_order":   [(bm, gap, runs) for _, bm, gap, runs in priority_queue],
        "interventions":    interventions,
    }


# ── Config generation ─────────────────────────────────────────────────────────

def _name_from_config(benchmark: str, model: str, cfg: dict) -> str:
    h = cfg.get("hidden_dim", 64)
    l = cfg.get("n_layers",   4)
    m = cfg.get("n_modes",    16)
    return f"agent_{model.lower()}_{benchmark[:5]}_h{h}_l{l}_m{m}"


def generate_new_configs(state: dict, top_n: int = 5, no_hpo: bool = False) -> list[dict]:
    """Use Bayesian HPO + hypothesis interventions to propose new ExperimentConfigs."""
    from core.utils import done_names
    done = done_names()
    configs = []

    # Source 1: hypothesis-driven interventions (highest confidence)
    for bm, interv in state["interventions"].items():
        model = interv.get("model", "FNO")
        cfg = {
            "hidden_dim": interv.get("hidden_dim", 128),
            "n_layers":   interv.get("n_layers",   8),
            "n_modes":    interv.get("n_modes",    24),
            "lr":         interv.get("lr",         1e-3),
        }
        loss = interv.get("loss_type", "l2_rel")
        name = _name_from_config(bm, model, cfg)
        if name in done:
            continue
        is_2d = "2d" in bm
        configs.append({
            "name":       name,
            "benchmark":  bm,
            "model":      model,
            "hidden_dim": cfg["hidden_dim"],
            "n_layers":   cfg["n_layers"],
            "n_modes":    cfg["n_modes"],
            "loss_type":  loss,
            "budget_s":   480 if is_2d else 300,
            "priority":   1,
            "rationale":  interv.get("rationale", "Hypothesis-engine suggestion"),
            "paper_ref":  interv.get("paper_ref", ""),
            "source":     "hypothesis",
        })

    # Source 2: Bayesian HPO suggestions for top-priority benchmarks
    if no_hpo:
        pass  # skipped via --no-hpo flag
    for bm, gap, runs in ([] if no_hpo else state["priority_order"][:3]):
        if gap < 1.0:
            continue  # already beating SOTA, deprioritise
        try:
            hpo = _load_hpo(bm)
            if len(hpo.y) < 2:
                continue  # not enough data for meaningful GP
            for _ in range(2):
                cfg = hpo.ask()
                name = _name_from_config(bm, "FNO", cfg)
                if name in done:
                    continue
                is_2d = "2d" in bm
                configs.append({
                    "name":       name,
                    "benchmark":  bm,
                    "model":      "FNO",
                    "hidden_dim": cfg["hidden_dim"],
                    "n_layers":   cfg["n_layers"],
                    "n_modes":    cfg["n_modes"],
                    "loss_type":  "l2_rel",
                    "budget_s":   480 if is_2d else 300,
                    "priority":   2,
                    "rationale":  f"Bayesian HPO suggestion (GP-EI, {len(hpo.y)} obs on {bm})",
                    "source":     "bayesian_hpo",
                })
        except Exception:
            pass

    # Deduplicate by name, cap at top_n
    seen = set()
    unique = []
    for c in configs:
        if c["name"] not in seen and c["name"] not in done:
            seen.add(c["name"])
            unique.append(c)
        if len(unique) >= top_n:
            break

    return unique


# ── Code generation ───────────────────────────────────────────────────────────

def _config_to_code(cfg: dict) -> str:
    """Render an ExperimentConfig(...) code block."""
    lines = [
        f"    ExperimentConfig(",
        f"        name={cfg['name']!r},",
        f"        benchmark={cfg['benchmark']!r},",
        f"        model={cfg['model']!r},",
        f"        hidden_dim={cfg['hidden_dim']}, n_layers={cfg['n_layers']}, n_modes={cfg['n_modes']},",
    ]
    if cfg.get("loss_type", "l2_rel") != "l2_rel":
        lines.append(f"        loss_type={cfg['loss_type']!r},")
    if cfg.get("budget_s", 300) != 300:
        lines.append(f"        budget_s={cfg['budget_s']},")
    lines.append(f"        priority={cfg['priority']},")
    if cfg.get("rationale"):
        rat = textwrap.shorten(cfg["rationale"], width=90)
        lines.append(f"        rationale={rat!r},")
    if cfg.get("paper_ref"):
        lines.append(f"        paper_ref={cfg['paper_ref']!r},")
    lines.append(f"    ),")
    return "\n".join(lines)


def append_configs_to_experiments(configs: list[dict]) -> int:
    """Append new ExperimentConfig entries to experiments.py (gated: smoke-test first)."""
    if not configs:
        return 0

    exp_path = REPO_ROOT / "experiments.py"
    content  = exp_path.read_text()

    # Validate: each config must have required fields and unique name
    from core.utils import done_names
    done = done_names()
    to_add = []
    for cfg in configs:
        if not all(k in cfg for k in ("name", "benchmark", "model", "hidden_dim", "n_layers")):
            print(f"  SKIP {cfg.get('name','?')} — missing required fields")
            continue
        if cfg["name"] in done:
            print(f"  SKIP {cfg['name']} — already in results")
            continue
        if f'name={cfg["name"]!r}' in content:
            print(f"  SKIP {cfg['name']} — already in experiments.py")
            continue
        to_add.append(cfg)

    if not to_add:
        return 0

    # Build insertion block
    block = "\n    # ── Agent-generated experiments (" + \
            __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M") + \
            ") ──────────────────────────────\n"
    for cfg in to_add:
        block += _config_to_code(cfg) + "\n"

    # Insert before the closing `]` of EXPERIMENTS
    insertion_point = content.rfind("\n]")
    if insertion_point == -1:
        print("  ERROR: could not find EXPERIMENTS closing ] in experiments.py")
        return 0

    new_content = content[:insertion_point] + block + content[insertion_point:]
    exp_path.write_text(new_content)
    print(f"  Appended {len(to_add)} new configs to experiments.py")
    return len(to_add)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_state_report(state: dict) -> None:
    gaps    = state["sota_gaps"]
    pending = state["pending_experiments"]

    print(f"\n{'═'*65}")
    print(f"  SciML Agent Loop — State Report")
    print(f"{'═'*65}")
    print(f"\n  Pending experiments in queue: {pending}")

    print(f"\n  SOTA gaps (our_best / sota — <1.0 means beating SOTA):")
    for bm, gap in sorted(gaps.items(), key=lambda x: -x[1]):
        bar  = "█" * min(20, int(gap * 5)) if gap > 0.1 else ""
        flag = " ← BEATS SOTA" if gap < 1.0 else ""
        print(f"    {bm:<25}  {gap:>6.2f}x  {bar}{flag}")

    order = state.get("priority_order", [])
    if order:
        print(f"\n  Research priority order:")
        for i, (bm, gap, runs) in enumerate(order[:5], 1):
            print(f"    {i}. {bm:<25}  gap={gap:.2f}x  runs={runs}")

    interventions = state.get("interventions", {})
    if interventions:
        print(f"\n  Hypothesis-engine interventions:")
        for bm, interv in interventions.items():
            model = interv.get("model", "FNO")
            rat   = textwrap.shorten(interv.get("rationale", ""), width=65)
            print(f"    {bm}: → {model}  {rat}")

    print()


def print_config_proposals(configs: list[dict]) -> None:
    if not configs:
        print("  No new configs generated.")
        return
    print(f"\n  Generated {len(configs)} new ExperimentConfig proposals:\n")
    for cfg in configs:
        print(f"  [{cfg['source']}] {cfg['name']}")
        print(f"    {cfg['benchmark']} / {cfg['model']} "
              f"h={cfg['hidden_dim']} l={cfg['n_layers']} m={cfg['n_modes']}")
        rat = textwrap.shorten(cfg.get("rationale", ""), width=70)
        print(f"    {rat}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="SciML Agent Orchestration Loop")
    p.add_argument("--benchmark", default=None,
                   help="Focus on a specific benchmark")
    p.add_argument("--top", type=int, default=5,
                   help="Number of new configs to generate")
    p.add_argument("--dry-run", action="store_true",
                   help="Analyse and plan without writing to experiments.py")
    p.add_argument("--run", action="store_true",
                   help="After generating configs, immediately run top-3 via autorun.py")
    p.add_argument("--no-hpo", action="store_true",
                   help="Skip Bayesian HPO suggestions (faster startup)")
    args = p.parse_args()

    if is_paused():
        print("  Loop is paused (.autorun_pause exists). Remove it to resume.")
        sys.exit(0)

    print("Analysing current state...")
    state = analyse_state(args.benchmark)
    print_state_report(state)

    print("Generating new experiment proposals...")
    configs = generate_new_configs(state, top_n=args.top, no_hpo=args.no_hpo)
    print_config_proposals(configs)

    if args.dry_run:
        print("  [dry-run] Not writing to experiments.py.")
        print("  Proposed ExperimentConfig snippets:\n")
        for cfg in configs:
            print(_config_to_code(cfg))
            print()
        return

    n_added = append_configs_to_experiments(configs)
    if n_added == 0:
        print("  Nothing new to add — queue already covers all suggestions.")
        return

    if args.run:
        print(f"\nLaunching autorun.py for top-3 new experiments...")
        subprocess.run(
            ["uv", "run", "autorun.py", "--max", "3",
             "--priority", "1", "--commit"],
            cwd=REPO_ROOT,
        )


if __name__ == "__main__":
    main()
