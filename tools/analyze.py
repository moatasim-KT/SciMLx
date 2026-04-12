"""SciML results analyzer and next-experiment suggester.

Reads results.tsv and produces:
  1. Best-per-benchmark/model summary table
  2. Improvement trajectory over time
  3. Hyperparameter correlation hints
  4. Concrete suggestions for next experiments

Usage:
    uv run analyze.py                     # full report
    uv run analyze.py --benchmark burgers_1d
    uv run analyze.py --plot              # save convergence plot (requires matplotlib)
"""

import argparse
import math
from collections import defaultdict
from typing import Optional

from core.utils import RESULTS_FILE, LOGS_DIR, FIGS_DIR, SOTA, load_results


# ── Analysis helpers ──────────────────────────────────────────────────────────

def best_per_group(rows: list[dict]) -> dict[tuple, dict]:
    """Return best row per (benchmark, model) group."""
    best: dict[tuple, dict] = {}
    for row in rows:
        if row["status"] not in ("keep",):
            continue
        key = (row["benchmark"], row["model"])
        if key not in best or row["val_l2_rel"] < best[key]["val_l2_rel"]:
            best[key] = row
    return best


def improvement_series(rows: list[dict], benchmark: str) -> list[tuple[int, float]]:
    """Return (index, val_l2_rel) for 'keep' rows in order, for one benchmark."""
    series = []
    for i, row in enumerate(rows):
        if row["benchmark"] == benchmark and row["status"] == "keep":
            series.append((i + 1, row["val_l2_rel"]))
    return series


def parse_config_from_desc(description: str) -> dict:
    """Extract hyperparams embedded in description field (space-separated tokens)."""
    params = {}
    for token in description.split():
        if "=" in token:
            k, v = token.split("=", 1)
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v
    return params


def correlate_hyperparams(rows: list[dict], benchmark: str) -> dict[str, list]:
    """Group val_l2_rel by hyperparameter value to spot trends."""
    from collections import defaultdict
    buckets: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["benchmark"] != benchmark:
            continue
        params = parse_config_from_desc(row.get("description", ""))
        val    = row["val_l2_rel"]
        if math.isnan(val):
            continue
        for k, v in params.items():
            buckets[k][str(v)].append(val)
    # Summarise: best (min) val per param value
    summary = {}
    for k, vdict in buckets.items():
        summary[k] = {v: min(vals) for v, vals in vdict.items()}
    return summary


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(rows: list[dict], benchmark_filter: Optional[str]) -> None:
    if not rows:
        print("No results found.")
        return

    benchmarks = sorted({r["benchmark"] for r in rows})
    if benchmark_filter:
        benchmarks = [b for b in benchmarks if b == benchmark_filter]

    for bm in benchmarks:
        sota = SOTA.get(bm, None)
        bm_rows = [r for r in rows if r["benchmark"] == bm]
        total = len(bm_rows)
        kept  = [r for r in bm_rows if r["status"] == "keep"]
        disc  = [r for r in bm_rows if r["status"] == "discard"]
        crash = [r for r in bm_rows if r["status"] == "crash"]

        print(f"\n{'═'*70}")
        print(f"  BENCHMARK: {bm}")
        print(f"  Total experiments: {total}  "
              f"(kept={len(kept)}, discarded={len(disc)}, crashed={len(crash)})")
        if sota:
            print(f"  SOTA target: {sota:.4f}")
        print(f"{'═'*70}")

        # Best per model
        bests = best_per_group(bm_rows)
        bm_bests = {k: v for k, v in bests.items() if k[0] == bm}
        if bm_bests:
            print(f"\n  Best per model type:")
            sorted_bests = sorted(bm_bests.items(), key=lambda x: x[1]["val_l2_rel"])
            for (_, model), row in sorted_bests:
                val  = row["val_l2_rel"]
                gap  = f"{val / sota:.1f}× from SOTA" if sota else ""
                print(f"    {model:<12}  {val:.6f}   {gap}   "
                      f"[ {row.get('description','')[:50]} ]")

        # Improvement trajectory
        series = improvement_series(bm_rows, bm)
        if len(series) > 1:
            first_val = series[0][1]
            last_val  = series[-1][1]
            pct       = (first_val - last_val) / first_val * 100
            print(f"\n  Improvement trajectory: "
                  f"{first_val:.6f} → {last_val:.6f}  ({pct:+.1f}%)")
            for idx, val in series:
                bar = "▓" * int(40 * (1 - val))
                print(f"    #{idx:>3}  {val:.6f}  {bar}")

        # All experiments table
        print(f"\n  All experiments (sorted by val_l2_rel):")
        sorted_rows = sorted(
            [r for r in bm_rows if not math.isnan(r["val_l2_rel"])],
            key=lambda r: r["val_l2_rel"]
        )
        print(f"    {'val_l2_rel':>12}  {'status':<8}  {'model':<12}  description")
        print(f"    {'─'*60}")
        for row in sorted_rows[:20]:
            print(f"    {row['val_l2_rel']:>12.6f}  {row['status']:<8}  "
                  f"{row['model']:<12}  {row.get('description','')[:40]}")

        # Suggestions
        print(f"\n  Suggestions for next experiments:")
        _print_suggestions(bm_rows, bm, sota)


def _print_suggestions(rows: list[dict], benchmark: str,
                        sota: Optional[float]) -> None:
    """Simple heuristic-based suggestions from results."""
    kept = [r for r in rows if r["status"] == "keep" and
            not math.isnan(r["val_l2_rel"])]
    if not kept:
        print("    No kept results yet — run autorun.py to start.")
        return

    best_row = min(kept, key=lambda r: r["val_l2_rel"])
    best_val = best_row["val_l2_rel"]
    best_desc = best_row.get("description", "")

    print(f"    Current best: {best_val:.6f}  [{best_desc[:50]}]")

    if sota:
        gap = best_val / sota
        if gap > 10:
            print(f"    Still {gap:.1f}× from SOTA. Focus on architectural improvements.")
            print(f"    → Try UNO (multi-scale) and PINO (physics loss) next.")
        elif gap > 3:
            print(f"    Getting closer ({gap:.1f}× from SOTA). "
                  "Fine-tune hyperparameters.")
            print(f"    → Try LR sweep (3e-4, 3e-3) and tighter grad clip.")
        else:
            print(f"    Near SOTA ({gap:.1f}× away). "
                  "Consider ensembling or longer training.")

    # Check if any model type has not been tried
    tried_models = {r["model"] for r in rows}
    for model in ["FNO", "UNO", "WNO", "DeepONet"]:
        if model not in tried_models:
            print(f"    → {model} not yet tried on {benchmark}.")

    # Check if PINO has been tried
    pino_rows = [r for r in rows if "pino" in r.get("description", "").lower()]
    if not pino_rows:
        print(f"    → Physics-informed loss (PINO) not yet explored.")
        print(f"       Try: uv run train.py --pino_lambda 0.01")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(rows: list[dict], benchmark: str) -> None:
    """Save val_l2_rel vs experiment index plot."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    bm_rows = [r for r in rows if r["benchmark"] == benchmark]
    if not bm_rows:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    sota = SOTA.get(benchmark)

    colors = {"keep": "green", "discard": "salmon", "crash": "gray"}
    for i, row in enumerate(bm_rows):
        if math.isnan(row["val_l2_rel"]):
            continue
        c = colors.get(row["status"], "black")
        ax.scatter(i + 1, row["val_l2_rel"], color=c, zorder=3, s=60)

    # Running best line
    best = float("inf")
    xs, ys = [], []
    for i, row in enumerate(bm_rows):
        v = row["val_l2_rel"]
        if not math.isnan(v) and v < best:
            best = v
            xs.append(i + 1)
            ys.append(best)
    if xs:
        ax.step(xs, ys, color="steelblue", linewidth=2, label="Running best")

    if sota:
        ax.axhline(sota, color="purple", linestyle="--",
                   linewidth=1.5, label=f"SOTA ({sota:.4f})")

    patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items()]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0][1:])
    ax.set_xlabel("Experiment index")
    ax.set_ylabel("val_l2_rel (lower is better)")
    ax.set_title(f"{benchmark} — experiment history")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    FIGS_DIR.mkdir(exist_ok=True)
    out = FIGS_DIR / f"analysis_{benchmark}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="SciML results analyzer")
    p.add_argument("--benchmark", default=None, help="Filter to one benchmark")
    p.add_argument("--plot",      action="store_true",
                   help="Save convergence plots with matplotlib")
    p.add_argument("--papers",    action="store_true",
                   help="Include paper registry gap report")
    args = p.parse_args()

    rows = load_results(args.benchmark)
    print_report(rows, args.benchmark)

    if args.papers:
        try:
            from tools.paper_registry import PaperRegistry
            reg = PaperRegistry()
            reg.gap_table()
        except Exception as e:
            print(f"\nPaper registry unavailable: {e}")

    if args.plot:
        benchmarks = ([args.benchmark] if args.benchmark
                      else sorted({r["benchmark"] for r in rows}))
        for bm in benchmarks:
            plot_results(rows, bm)


if __name__ == "__main__":
    main()
