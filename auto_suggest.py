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
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

from core.utils import PAPERS_DIR, SOTA as _SOTA_TARGETS, load_results, best_per_benchmark as _best_per_bm, done_names as _done_names_fn

# Related benchmark groups for cross-benchmark transfer
# When a config wins on benchmark A, also suggest it on its relatives
BENCHMARK_RELATIVES = {
    "burgers_1d":   ["kdv_1d", "wave_1d"],
    "kdv_1d":       ["burgers_1d", "wave_1d"],
    "wave_1d":      ["burgers_1d", "kdv_1d"],
    "darcy_2d": ["ns_2d", "swe_2d", "allen_cahn_2d"],
    "ns_2d":    ["darcy_2d", "swe_2d"],
    "swe_2d":       ["darcy_2d", "ns_2d"],
    "allen_cahn_2d":["darcy_2d"],
    "euler_1d":     ["burgers_1d", "kdv_1d"],
}

# ── Empirical heuristics (from accumulated results) ───────────────────────────

# Known bad ideas — skip suggesting these again
BLACKLIST = {
    "pino",              # Physics loss at endpoint → fundamentally broken (original variant)
    "wno_burgers",       # Wrong inductive bias for periodic Burgers
    "l=10_fno",          # FNO step-time-limited at l=10 without residuals
    "l=12_fno",          # FNO collapses at l=12
    "h=256_fno",         # Width doesn't help — step-time-limited
    "afno_burgers",      # AFNO consistently 0.50-0.72 on Burgers — wrong inductive bias
    "h=128_2d_budget",   # 2D models with h=128 only get 2 steps in 5-min budget
    # "darcy_2d" is now consolidated
    "ns_2d",  # Original NS-2D has broken ICs — use ns_2d
}

# Known good patterns (from empirical findings — also updated dynamically from results.json)
_KNOWN_WINS_HARDCODED = {
    "burgers_1d": {
        "best_modes":  24,
        "best_hidden": 128,
        "best_layers": 8,
        "best_model":  "FNO",
        "best_val":    0.146759,
        "best_config": "fno_h128_m24_l8_aug",
        "key_findings": [
            "m=24 beats m=16 by 11% and m=32 catastrophically",
            "h=128 beats h=64 and h=256 (step-time-limited)",
            "l=8 is depth sweet spot for FNO; l=10+ degrades",
            "lr=1e-3 optimal; bs=32 better than bs=16",
            "Augmentation helps: +0.6% improvement over no-aug",
            "Pre-LN residuals (RFNO) needed to unlock l>8",
            "PINO fundamentally broken for endpoint-only training",
            "WNO wrong inductive bias for periodic Burgers",
            "AFNO weak on Burgers: 0.50-0.72 regardless of depth",
            "FFNO moderate: 0.24-0.35, worse than FNO",
        ],
    },
    "kdv_1d": {
        "best_modes":  24,
        "best_hidden": 128,
        "best_layers": 8,
        "best_model":  "RFNO",
        "best_val":    0.002023,
        "best_config": "rfno_kdv_h128_m24_l8",
        "key_findings": [
            "RFNO beats FNO on KdV: 0.002023 vs 0.002374 (soliton dynamics benefit from stability)",
            "Wide models (h=256) hurt: 0.0061 — fewer training steps",
            "l=8 is sweet spot; l=10: 0.002625, l=12: 0.003297",
            "m=24 beats m=32: 0.002023 vs 0.002394",
            "FFNO weak on KdV: 0.005267",
            "Beats SOTA by 5x: 0.002023 vs ~0.010",
        ],
    },
    "wave_1d": {
        "best_modes":  16,
        "best_hidden": 64,
        "best_layers": 4,
        "best_model":  "FNO",
        "best_val":    0.000992,
        "best_config": "fno_wave_h64_m16_l4",
        "key_findings": [
            "SMALLER model wins: FNO h=64 l=4 m=16 beats RFNO h=128 l=8 m=24",
            "Mechanism: smaller model gets 4x more training steps in 5-min budget",
            "Wave 1D is easy for Fourier methods; more steps > bigger model",
            "Beats SOTA by 5x: 0.000992 vs ~0.005",
            "h=128 l=8: 0.002098 (RFNO) — still very good but not best",
        ],
    },
    "darcy_2d": {
        "best_modes":  8,
        "best_hidden": 32,
        "best_layers": 4,
        "best_model":  "FNO",
        "best_val":    0.146942,
        "best_config": "fno_darcy2d_fix_h32_m8_l4",
        "key_findings": [
            "h=32 too small (0.1469) — needs more capacity",
            "h=128 l=8 m=24 ran out of time (only 2 steps in 5-min budget for 2D)",
            "2D models need extended budget (~480s) to train meaningfully",
            "Consolidated Darcy benchmark with corrected Richardson solver",
            "FNO2D auto-routed from FNO for 2D benchmarks",
        ],
    },
    "ns_2d": {
        "best_modes":  8,
        "best_hidden": 32,
        "best_layers": 4,
        "best_model":  "FNO",
        "best_val":    0.015166,
        "best_config": "fno_ns2d_fix_h32_m8_l4",
        "key_findings": [
            "Only 1 run so far — baseline FNO h=32 l=4 m=8",
            "1.2x from SOTA (0.0152 vs 0.0128) — very close, likely beatable",
            "Use ns_2d only — original ns_2d has broken ICs (CFL>60, NaN)",
            "2D models need extended budget (~480s)",
            "Wave 1D finding suggests smaller model + more steps may help",
        ],
    },
}

# ── Dynamic KNOWN_WINS: computed from results.json, falls back to hardcoded ───

_known_wins_cache: dict | None = None

def _compute_known_wins() -> dict:
    """Build KNOWN_WINS from actual results.json, merging with hardcoded fallback."""
    from core.utils import REPO_ROOT
    results_path = REPO_ROOT / "results.json"
    computed: dict = {}
    if results_path.exists():
        try:
            data = json.loads(results_path.read_text())
            by_bm: dict[str, list] = {}
            for e in data:
                if e.get("status") == "keep" and e.get("val_l2_rel"):
                    by_bm.setdefault(e["benchmark"], []).append(e)
            for bm, entries in by_bm.items():
                best = min(entries, key=lambda e: e["val_l2_rel"])
                cfg  = best.get("config") or {}
                hard = _KNOWN_WINS_HARDCODED.get(bm, {})
                computed[bm] = {
                    "best_modes":  cfg.get("n_modes",    hard.get("best_modes",  16)),
                    "best_hidden": cfg.get("hidden_dim", hard.get("best_hidden", 64)),
                    "best_layers": cfg.get("n_layers",   hard.get("best_layers",  4)),
                    "best_model":  best.get("model",     hard.get("best_model",  "FNO")),
                    "best_val":    best["val_l2_rel"],
                    "best_config": (best.get("description") or "").split()[0],
                    "key_findings": hard.get("key_findings", []),
                }
        except Exception:
            pass
    # Hardcoded values fill in any benchmark not yet in results.json
    merged = dict(_KNOWN_WINS_HARDCODED)
    merged.update(computed)
    return merged

def _get_known_wins() -> dict:
    """Return KNOWN_WINS, computing once per process and caching."""
    global _known_wins_cache
    if _known_wins_cache is None:
        _known_wins_cache = _compute_known_wins()
    return _known_wins_cache

# Module-level alias — refreshed on each top-level import (safe for CLI use)
KNOWN_WINS: dict = {}  # populated by _refresh_known_wins() below

def _refresh_known_wins() -> None:
    global KNOWN_WINS
    KNOWN_WINS = _get_known_wins()

_refresh_known_wins()


# Aliases to shared utils (kept as module-level names for call-site clarity)
_load_results        = load_results
_done_names          = _done_names_fn
_best_per_benchmark  = _best_per_bm


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

    wins = (_get_known_wins()).get(benchmark, {})
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


def _load_diag_from_results() -> dict[str, dict]:
    """Load spectral diagnostics from results.json diag fields.

    Returns: {exp_name_prefix: {"diag_high_freq_error": float, ...}}
    """
    from core.utils import REPO_ROOT
    results_path = REPO_ROOT / "results.json"
    diag_map: dict[str, dict] = {}
    if not results_path.exists():
        return diag_map
    try:
        with open(results_path) as f:
            data = json.load(f)
        for e in data:
            d = e.get("diag") or {}
            if d:
                name = e.get("description", "").split()[0]
                diag_map[name] = d
    except Exception:
        pass
    return diag_map


def _generate_diagnostic_suggestions(benchmark: str) -> list[Suggestion]:
    """Suggest loss/mode changes based on spectral bias diagnostics."""
    suggs = []
    done  = _done_names()
    diag_map = _load_diag_from_results()
    wins = (_get_known_wins()).get(benchmark, {})
    bh = wins.get("best_hidden", 128)
    bm = wins.get("best_modes", 24)
    bl = wins.get("best_layers", 8)

    # Find experiments for this benchmark with high spectral bias
    high_freq_experiments = [
        (name, d) for name, d in diag_map.items()
        if d.get("diag_high_freq_error", 0) > 0.3
    ]

    if high_freq_experiments:
        # H1 loss targets high-frequency errors directly
        name = f"fno_h{bh}_m{bm}_l{bl}_h1_diag"
        if name not in done:
            suggs.append(Suggestion(
                name=name, benchmark=benchmark,
                cli=f"uv run train.py --model FNO --hidden {bh} --layers {bl} "
                    f"--modes {bm} --loss h1 --h1_alpha 0.1 --benchmark {benchmark}",
                rationale=f"Spectral bias detected (high_freq_error>0.3 in {len(high_freq_experiments)} runs). "
                          f"H1 Sobolev loss directly penalises derivative errors at high frequencies.",
                expected="5-15% improvement on high-freq errors",
                priority=1, source="diagnostic:spectral_bias",
            ))
        # More modes to cover the under-sampled frequency range
        more_modes = min(bm + 8, 32)
        name = f"fno_h{bh}_m{more_modes}_l{bl}_diag"
        if name not in done:
            suggs.append(Suggestion(
                name=name, benchmark=benchmark,
                cli=f"uv run train.py --model FNO --hidden {bh} --layers {bl} "
                    f"--modes {more_modes} --benchmark {benchmark}",
                rationale=f"Spectral bias: increasing modes {bm}→{more_modes} to capture missing high-freq components.",
                expected="3-10% improvement",
                priority=2, source="diagnostic:spectral_bias",
            ))

    return suggs


def _generate_transfer_suggestions(benchmark: str) -> list[Suggestion]:
    """When a config wins on a related benchmark, suggest it here too."""
    suggs = []
    done  = _done_names()
    relatives = BENCHMARK_RELATIVES.get(benchmark, [])

    for rel_bm in relatives:
        rel_wins = (_get_known_wins()).get(rel_bm, {})
        if not rel_wins or rel_wins.get("best_val", float("inf")) > 0.5:
            continue  # no useful result on the relative benchmark

        rel_model  = rel_wins.get("best_model", "FNO")
        rel_hidden = rel_wins.get("best_hidden", 128)
        rel_layers = rel_wins.get("best_layers", 8)
        rel_modes  = rel_wins.get("best_modes", 24)
        rel_val    = rel_wins.get("best_val", 1.0)

        name = f"{rel_model.lower()}_transfer_{rel_bm[:5]}_h{rel_hidden}_l{rel_layers}_m{rel_modes}_{benchmark[:5]}"
        if name in done:
            continue

        is_2d = benchmark.endswith("_2d") or benchmark.endswith("_2d_fix")
        budget_flag = " --budget 480" if is_2d else ""
        suggs.append(Suggestion(
            name=name, benchmark=benchmark,
            cli=f"uv run train.py --model {rel_model} --hidden {rel_hidden} "
                f"--layers {rel_layers} --modes {rel_modes} "
                f"--benchmark {benchmark}{budget_flag}",
            rationale=f"Cross-benchmark transfer: {rel_model} h={rel_hidden} l={rel_layers} m={rel_modes} "
                      f"achieved {rel_val:.4f} on {rel_bm}. Testing if this config transfers to {benchmark}.",
            expected=f"Unknown — {rel_bm} insight may transfer",
            priority=2, source=f"transfer:{rel_bm}",
        ))

    return suggs


def _rank_suggestions(suggs: list[Suggestion]) -> list[Suggestion]:
    """Sort by priority, then by estimated impact."""
    return sorted(suggs, key=lambda s: (s.priority, s.name))


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_findings(rows: list[dict], benchmark: str) -> None:
    wins = (_get_known_wins()).get(benchmark, {})
    findings = wins.get("key_findings", [])
    if not findings:
        return
    print(f"\n  Key empirical findings for {benchmark}:")
    for f in findings:
        print(f"    • {f}")


def _sota_gap(rows: list[dict], benchmark: str) -> None:
    best = _best_per_benchmark(rows)
    sota = _SOTA_TARGETS.get(benchmark)
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

        # Generate suggestions (empirical + paper + diagnostic + transfer)
        emp_suggs      = _generate_empirical_suggestions(all_rows, bm)
        paper_suggs    = _generate_paper_suggestions(all_rows, bm)
        diag_suggs     = _generate_diagnostic_suggestions(bm)
        transfer_suggs = _generate_transfer_suggestions(bm)
        all_suggs      = _rank_suggestions(emp_suggs + paper_suggs + diag_suggs + transfer_suggs)

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
    wins  = (_get_known_wins()).get(benchmark, {})
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
