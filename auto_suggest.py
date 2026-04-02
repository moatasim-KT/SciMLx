"""Autonomous next-experiment suggester for the SciML research loop.

Combines three sources of information to generate prioritised experiment plans:
  1. results.tsv   — what we've already tried and what worked
  2. experiments.py — what's in the queue and what's pending
  3. papers/*.yaml  — what the literature says to try next

Usage:
    uv run auto_suggest.py                    # full suggestion report
    uv run auto_suggest.py --benchmark burgers_1d
    uv run auto_suggest.py --top 10           # top N suggestions
    uv run auto_suggest.py --generate         # output ready-to-run experiment configs
    uv run auto_suggest.py --gaps             # SOTA gap analysis from papers

The output is actionable:
  - "Run now" suggestions with specific CLI commands
  - ExperimentConfig snippets to paste into experiments.py
  - Literature-backed rationales for each suggestion

Design principle: This tool should remove the need for human reasoning about
what to try next.  An agent can call `uv run auto_suggest.py` and immediately
get the next experiments to run, drawn from both empirical findings and papers.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

RESULTS_FILE = Path(__file__).parent / "results.tsv"
PAPERS_DIR   = Path(__file__).parent / "papers"

# ── Empirical heuristics (from accumulated results) ───────────────────────────

# Known bad ideas — skip suggesting these again
BLACKLIST = {
    "pino",          # Physics loss at endpoint → fundamentally broken
    "wno_burgers",   # Wrong inductive bias for periodic Burgers
    "l=10_fno",      # FNO step-time-limited at l=10 without residuals
    "l=12_fno",      # FNO collapses at l=12
    "h=256_fno",     # Width doesn't help — step-time-limited
}

# Known good patterns (from empirical findings)
KNOWN_WINS = {
    "burgers_1d": {
        "best_modes":  24,
        "best_hidden": 128,
        "best_layers": 8,
        "best_model":  "FNO",
        "best_val":    0.155287,
        "best_config": "fno_h128_m24_l8",
        "key_findings": [
            "m=24 beats m=16 by 11% and m=32 catastrophically",
            "h=128 beats h=64 and h=256 (step-time-limited)",
            "l=8 is depth sweet spot for FNO; l=10+ degrades",
            "lr=1e-3 optimal; bs=32 better than bs=16",
            "Pre-LN residuals (RFNO) needed to unlock l>8",
            "PINO fundamentally broken for endpoint-only training",
            "WNO wrong inductive bias for periodic Burgers",
        ],
    }
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_results(benchmark: Optional[str] = None) -> list[dict]:
    rows = []
    if not RESULTS_FILE.exists():
        return rows
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                row["val_l2_rel"] = float(row["val_l2_rel"])
            except (ValueError, KeyError):
                row["val_l2_rel"] = float("nan")
            if benchmark and row.get("benchmark") != benchmark:
                continue
            rows.append(row)
    return rows


def _done_names() -> set[str]:
    done = set()
    if not RESULTS_FILE.exists():
        return done
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            desc = row.get("description", "")
            done.add(desc.split()[0] if desc else "")
    return done


def _best_per_benchmark(rows: list[dict]) -> dict[str, float]:
    best: dict[str, float] = {}
    for row in rows:
        if row.get("status") == "keep" and not math.isnan(row["val_l2_rel"]):
            bm = row["benchmark"]
            if bm not in best or row["val_l2_rel"] < best[bm]:
                best[bm] = row["val_l2_rel"]
    return best


# ── Suggestion generation ─────────────────────────────────────────────────────

class Suggestion:
    def __init__(self, name: str, benchmark: str, cli: str,
                 rationale: str, expected: str, priority: int,
                 source: str):
        self.name      = name
        self.benchmark = benchmark
        self.cli       = cli
        self.rationale = rationale
        self.expected  = expected
        self.priority  = priority
        self.source    = source   # "empirical" | "paper:<id>" | "ablation"

    def __repr__(self) -> str:
        return f"Suggestion({self.name}, p={self.priority})"


def _generate_empirical_suggestions(rows: list[dict],
                                     benchmark: str) -> list[Suggestion]:
    """Suggestions based purely on empirical findings in results.tsv."""
    done  = _done_names()
    best  = _best_per_benchmark(rows).get(benchmark, float("inf"))
    suggs = []

    wins = KNOWN_WINS.get(benchmark, {})
    bm   = wins.get("best_modes",  24)
    bh   = wins.get("best_hidden", 128)
    bl   = wins.get("best_layers", 8)

    # 1. Try RFNO if not yet run (session 6 is running — this covers post-session)
    for l in [8, 10, 12]:
        name = f"rfno_h{bh}_m{bm}_l{l}"
        if name not in done:
            suggs.append(Suggestion(
                name=name, benchmark=benchmark,
                cli=f"uv run train.py --model RFNO --hidden {bh} --layers {l} "
                    f"--modes {bm} --benchmark {benchmark}",
                rationale=f"RFNO l={l}: Pre-LN residuals unlock depth > 8 (FNO degraded here)",
                expected="~0.13–0.15",
                priority=1, source="empirical",
            ))

    # 2. AFNO at best config
    for l in [8, 10]:
        name = f"afno_h{bh}_m{bm}_l{l}"
        if name not in done:
            suggs.append(Suggestion(
                name=name, benchmark=benchmark,
                cli=f"uv run train.py --model AFNO --hidden {bh} --layers {l} "
                    f"--modes {bm} --benchmark {benchmark}",
                rationale=f"AFNO: non-linear Fourier mixing + softshrink sparsity (Guibas 2022)",
                expected="~0.12–0.15",
                priority=1, source="paper:afno-2022",
            ))

    # 3. H1 loss on best FNO config
    name = f"fno_h{bh}_m{bm}_l{bl}_h1"
    if name not in done:
        suggs.append(Suggestion(
            name=name, benchmark=benchmark,
            cli=f"uv run train.py --model FNO --hidden {bh} --layers {bl} "
                f"--modes {bm} --loss h1 --h1_alpha 0.1 --benchmark {benchmark}",
            rationale="H1 Sobolev loss: penalises gradient errors → targets shock fronts directly",
            expected=f"~{best * 0.88:.4f}–{best * 1.02:.4f}",
            priority=1, source="paper:h1-sobolev-loss",
        ))

    # 4. Best RFNO + H1 combination
    name = f"rfno_h{bh}_m{bm}_l8_h1"
    if name not in done:
        suggs.append(Suggestion(
            name=name, benchmark=benchmark,
            cli=f"uv run train.py --model RFNO --hidden {bh} --layers 8 "
                f"--modes {bm} --loss h1 --h1_alpha 0.1 --benchmark {benchmark}",
            rationale="RFNO + H1: compound architecture + loss improvements",
            expected=f"~{best * 0.80:.4f}–{best * 0.92:.4f}",
            priority=2, source="empirical+paper",
        ))

    # 5. KdV baseline (new benchmark, high value)
    if benchmark == "burgers_1d":
        name = "fno_kdv_h128_m24_l8"
        if name not in done:
            suggs.append(Suggestion(
                name=name, benchmark="kdv_1d",
                cli=f"uv run train.py --model FNO --hidden 128 --layers 8 "
                    f"--modes 24 --benchmark kdv_1d",
                rationale="Establish KdV baseline. New benchmark → immediate novel result.",
                expected="~0.02–0.08",
                priority=2, source="paper:ffno-2023",
            ))

    return suggs


def _generate_paper_suggestions(rows: list[dict],
                                  benchmark: str) -> list[Suggestion]:
    """Suggestions from paper registry (pending papers)."""
    suggs = []
    done  = _done_names()

    if not PAPERS_DIR.exists():
        return suggs

    import glob
    import re

    for yaml_path in sorted(PAPERS_DIR.glob("*.yaml")):
        try:
            with open(yaml_path) as f:
                content = f.read()
        except Exception:
            continue

        # Check if status is pending
        if "status: pending" not in content:
            continue

        # Extract suggested experiments
        paper_id = re.search(r"^id:\s*(.+)", content, re.MULTILINE)
        paper_id = paper_id.group(1).strip() if paper_id else "unknown"

        exp_blocks = re.findall(r"- name:\s*(\S+).*?rationale:\s*(.*?)(?=\n\s*-|\Z)",
                                content, re.DOTALL)
        for name, rationale in exp_blocks:
            name = name.strip()
            if name in done:
                continue
            # Parse model from name
            model = "FNO"
            for m in ["AFNO", "RFNO", "FFNO", "UNO", "WNO"]:
                if m.lower() in name.lower():
                    model = m
                    break
            suggs.append(Suggestion(
                name=name, benchmark=benchmark,
                cli=f"uv run train.py --model {model} --benchmark {benchmark} "
                    f"# [from paper {paper_id} — fill in hyperparams]",
                rationale=rationale.strip()[:120],
                expected="see paper",
                priority=3, source=f"paper:{paper_id}",
            ))

    return suggs


def _rank_suggestions(suggs: list[Suggestion]) -> list[Suggestion]:
    """Sort by priority, then by estimated impact."""
    return sorted(suggs, key=lambda s: (s.priority, s.name))


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_findings(rows: list[dict], benchmark: str) -> None:
    wins = KNOWN_WINS.get(benchmark, {})
    findings = wins.get("key_findings", [])
    if not findings:
        return
    print(f"\n  Key empirical findings for {benchmark}:")
    for f in findings:
        print(f"    • {f}")


def _sota_gap(rows: list[dict], benchmark: str) -> None:
    SOTA_TARGETS = {
        "burgers_1d":       0.0149,
        "darcy_2d":         0.0108,
        "navier_stokes_2d": 0.0128,
        "kdv_1d":           0.010,
        "wave_1d":          0.005,
    }
    best = _best_per_benchmark(rows)
    sota = SOTA_TARGETS.get(benchmark)
    our  = best.get(benchmark)
    if sota and our:
        gap = our / sota
        print(f"\n  SOTA gap ({benchmark}):  "
              f"our={our:.6f}  sota={sota:.4f}  gap={gap:.1f}×")
        # Estimate experiments needed at 10% improvement/run
        improvements_needed = math.log(gap) / math.log(1 / 0.90)
        print(f"  At 10%/run → ~{improvements_needed:.0f} more successful experiments to SOTA")


def report(benchmark: Optional[str], top_n: int) -> None:
    all_rows = _load_results()
    benchmarks = [benchmark] if benchmark else sorted({r["benchmark"] for r in all_rows})

    print(f"\n{'═'*70}")
    print("  SciML AutoSuggest — next experiment recommendations")
    print(f"{'═'*70}")

    for bm in benchmarks:
        rows = [r for r in all_rows if r["benchmark"] == bm]
        if not rows and bm not in ("kdv_1d", "wave_1d"):
            continue

        print(f"\n── {bm} {'─'*(60-len(bm))}")

        _sota_gap(all_rows, bm)
        _print_findings(all_rows, bm)

        # Generate suggestions
        emp_suggs   = _generate_empirical_suggestions(all_rows, bm)
        paper_suggs = _generate_paper_suggestions(all_rows, bm)
        all_suggs   = _rank_suggestions(emp_suggs + paper_suggs)

        if not all_suggs:
            print(f"\n  No new suggestions — all known experiments have been run!")
            continue

        print(f"\n  Top {min(top_n, len(all_suggs))} suggestions (of {len(all_suggs)} total):")
        for i, s in enumerate(all_suggs[:top_n], 1):
            print(f"\n  [{i}] {s.name}  (priority={s.priority}, source={s.source})")
            print(f"      Rationale: {s.rationale[:80]}")
            print(f"      Expected:  {s.expected}")
            print(f"      CLI:       {s.cli}")

    # Final: what to run right now
    done = _done_names()
    all_suggs = []
    for bm in (benchmarks if benchmark else ["burgers_1d"]):
        rows = _load_results(bm)
        all_suggs += _generate_empirical_suggestions(rows, bm)
    all_suggs = _rank_suggestions(all_suggs)

    if all_suggs:
        top = all_suggs[0]
        print(f"\n{'═'*70}")
        print(f"  ▶ Run this next:")
        print(f"    {top.cli}")
        print(f"{'═'*70}\n")


def generate_config_snippets(benchmark: str, top_n: int = 5) -> None:
    """Print ExperimentConfig Python snippets ready to paste into experiments.py."""
    rows  = _load_results(benchmark)
    done  = _done_names()
    wins  = KNOWN_WINS.get(benchmark, {})
    best  = _best_per_benchmark(rows).get(benchmark, 1.0)
    bm    = wins.get("best_modes",  24)
    bh    = wins.get("best_hidden", 128)
    bl    = wins.get("best_layers", 8)

    snippets = []

    configs = [
        ("afno_h128_m24_l8",  "AFNO",  128, 8,  24, "l2_rel", 0.1, 1,
         "AFNO at best FNO config. Non-linear Fourier mixing + softshrink."),
        ("afno_h128_m24_l10", "AFNO",  128, 10, 24, "l2_rel", 0.1, 1,
         "AFNO deeper — Pre-LN residuals should unlock l=10."),
        ("fno_h128_m24_l8_h1","FNO",   128, 8,  24, "h1",     0.1, 1,
         "H1 Sobolev loss on best FNO. Targets shock gradient errors."),
        ("rfno_h128_m24_l10", "RFNO",  128, 10, 24, "l2_rel", 0.1, 1,
         "RFNO l=10 — FNO degraded here; residuals should unlock it."),
        ("afno_h128_m24_l8_h1","AFNO", 128, 8,  24, "h1",     0.1, 2,
         "AFNO + H1: compound architecture+loss innovations."),
    ]

    print(f"\n# ── Auto-generated experiment configs for {benchmark} ──────")
    print(f"# Generated by auto_suggest.py | current best = {best:.6f}\n")

    for name, model, h, l, m, loss, alpha, pri, rat in configs[:top_n]:
        if name in done:
            print(f"# SKIP: {name} — already in results.tsv")
            continue
        loss_arg = f'loss_type="{loss}", h1_alpha={alpha},' if loss != "l2_rel" else ""
        print(f"    ExperimentConfig(")
        print(f"        name=\"{name}\",")
        print(f"        benchmark=\"{benchmark}\", model=\"{model}\",")
        print(f"        hidden_dim={h}, n_layers={l}, n_modes={m},")
        if loss_arg:
            print(f"        {loss_arg}")
        print(f"        priority={pri},")
        print(f"        rationale=\"{rat}\",")
        print(f"    ),")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Autonomous experiment suggester")
    p.add_argument("--benchmark", default=None)
    p.add_argument("--top",       type=int, default=8)
    p.add_argument("--generate",  action="store_true",
                   help="Output ExperimentConfig code snippets")
    p.add_argument("--gaps",      action="store_true",
                   help="Just show SOTA gap analysis")
    args = p.parse_args()

    if args.generate:
        bm = args.benchmark or "burgers_1d"
        generate_config_snippets(bm, top_n=args.top)
    elif args.gaps:
        rows = _load_results()
        benchmarks = [args.benchmark] if args.benchmark else [
            "burgers_1d", "darcy_2d", "kdv_1d", "wave_1d"
        ]
        for bm in benchmarks:
            _sota_gap(rows, bm)
    else:
        report(args.benchmark, args.top)


if __name__ == "__main__":
    main()
