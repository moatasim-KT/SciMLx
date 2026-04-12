"""Paper registry — loads all papers/*.yaml files and provides query APIs.

This module bridges the literature (papers/) with the experiment infrastructure
(experiments.py, autorun.py, analyze.py) to make the research loop aware of:

  1. What papers have been implemented and their expected improvements
  2. What our empirical results are vs paper claims
  3. What experiments to run next based on the literature

Usage:
    from paper_registry import PaperRegistry
    reg = PaperRegistry()
    reg.report()                   # full literature-vs-results report
    reg.suggest_next(results_tsv)  # suggest experiments from pending papers
    reg.gap_table()                # our results vs SOTA per benchmark

CLI:
    uv run paper_registry.py           # print full registry report
    uv run paper_registry.py --gaps    # print SOTA gap table
    uv run paper_registry.py --suggest # suggest next experiments
"""

import argparse
from pathlib import Path
from typing import Optional

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from core.utils import PAPERS_DIR, RESULTS_FILE, load_results as _load_rows


# ── YAML fallback (tiny parser for simple key: value files) ──────────────────

def _load_yaml_simple(path: Path) -> dict:
    """Minimal YAML loader — handles only what our paper files use.
    Falls back to this if PyYAML is not installed.
    """
    import re
    result = {}
    current_key = None
    multiline_value = []
    indent_level = 0

    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.rstrip()
        if not stripped or stripped.startswith("#"):
            if multiline_value is not None and current_key:
                multiline_value.append("")
            continue

        # Detect top-level key: value
        m = re.match(r"^(\w[\w_-]*):\s*(.*)", stripped)
        if m and not line.startswith(" "):
            if current_key and multiline_value:
                result[current_key] = " ".join(
                    s for s in multiline_value if s
                ).strip()
                multiline_value = []
            current_key = m.group(1)
            val = m.group(2).strip()
            if val == ">":
                multiline_value = []
            elif val.startswith('"') and val.endswith('"'):
                result[current_key] = val[1:-1]
                current_key = None
            elif val == "null":
                result[current_key] = None
                current_key = None
            elif val:
                result[current_key] = val
                current_key = None
        elif current_key and multiline_value is not None:
            multiline_value.append(stripped.strip())

    if current_key and multiline_value:
        result[current_key] = " ".join(s for s in multiline_value if s).strip()

    return result


def _load_paper(path: Path) -> dict:
    if _HAS_YAML:
        with open(path) as f:
            return yaml.safe_load(f)
    return _load_yaml_simple(path)


# ── PaperRegistry ────────────────────────────────────────────────────────────

class PaperRegistry:
    """Loads papers/*.yaml and provides query APIs."""

    def __init__(self):
        self.papers: list[dict] = []
        self._load_all()

    def _load_all(self) -> None:
        if not PAPERS_DIR.exists():
            return
        for path in sorted(PAPERS_DIR.glob("*.yaml")):
            try:
                paper = _load_paper(path)
                paper["_file"] = path.name
                if "id" not in paper:
                    paper["id"] = path.stem
                self.papers.append(paper)
            except Exception as e:
                print(f"Warning: could not load {path.name}: {e}")

    # ── Query methods ─────────────────────────────────────────────────────────

    def by_status(self, status: str) -> list[dict]:
        return [p for p in self.papers if p.get("status", "").startswith(status)]

    def by_model_class(self, model_class: str) -> list[dict]:
        return [p for p in self.papers if p.get("model_class") == model_class]

    def pending(self) -> list[dict]:
        """Papers with status='pending' — not yet implemented."""
        return [p for p in self.papers if p.get("status") == "pending"]

    def implemented(self) -> list[dict]:
        return [p for p in self.papers
                if str(p.get("status", "")).startswith("implemented")]

    def get(self, paper_id: str) -> Optional[dict]:
        for p in self.papers:
            if p.get("id") == paper_id:
                return p
        return None

    # ── Results integration ───────────────────────────────────────────────────

    @staticmethod
    def _load_results() -> dict[str, float]:
        """Return best val_l2_rel per benchmark from results.tsv."""
        from core.utils import best_per_benchmark
        return best_per_benchmark(_load_rows())

    def gap_table(self) -> None:
        """Print our results vs SOTA targets from paper registry."""
        best = self._load_results()

        print(f"\n{'═'*75}")
        print("  PAPER REGISTRY — SOTA Gap Report")
        print(f"{'═'*75}")
        print(f"  {'Benchmark':<20}  {'Paper SOTA':>12}  {'Our Best':>12}  "
              f"{'Gap':>7}  {'Paper'}")
        print(f"  {'─'*70}")

        # Collect all (benchmark, paper_sota, paper_id) tuples
        rows = []
        for paper in self.papers:
            bms = paper.get("benchmarks")
            if not isinstance(bms, dict):
                continue
            for bm, bm_data in bms.items():
                if not isinstance(bm_data, dict):
                    continue
                reported = bm_data.get("reported_val_l2_rel")
                if reported is None:
                    continue
                try:
                    sota = float(reported)
                except (TypeError, ValueError):
                    continue
                rows.append((bm, sota, paper.get("id", "?"), paper.get("model_class", "?")))

        # Deduplicate by benchmark: keep lowest SOTA per benchmark
        sota_by_bm: dict[str, tuple] = {}
        for bm, sota, pid, mc in rows:
            if bm not in sota_by_bm or sota < sota_by_bm[bm][0]:
                sota_by_bm[bm] = (sota, pid, mc)

        for bm in sorted(sota_by_bm):
            sota, pid, mc = sota_by_bm[bm]
            ours = best.get(bm)
            gap  = f"{ours / sota:.1f}×" if ours is not None else "N/A"
            ours_str = f"{ours:.6f}" if ours is not None else "N/A"
            print(f"  {bm:<20}  {sota:>12.4f}  {ours_str:>12}  {gap:>7}  {pid}")

        print()

    def report(self) -> None:
        """Full registry report: all papers with status and results."""
        best = self._load_results()

        print(f"\n{'═'*75}")
        print(f"  Paper Registry — {len(self.papers)} papers  "
              f"({len(self.implemented())} implemented, {len(self.pending())} pending)")
        print(f"{'═'*75}")

        for paper in sorted(self.papers, key=lambda p: p.get("year", 9999)):
            status = paper.get("status", "?")
            status_icon = {"implemented": "✓", "pending": "○", "failed": "✗",
                           "implemented_running": "⟳",
                           "partial": "◑"}.get(status, "?")
            print(f"\n  [{status_icon}] {paper.get('id','?')} — {paper.get('title','')[:55]}")
            print(f"       Venue: {paper.get('venue','?')} | "
                  f"Model: {paper.get('model_class','?')}")

            bms = paper.get("benchmarks")
            if isinstance(bms, dict):
                for bm, bm_data in bms.items():
                    if not isinstance(bm_data, dict):
                        continue
                    reported = bm_data.get("reported_val_l2_rel")
                    our_val  = best.get(bm)
                    our_str  = f"{our_val:.6f}" if our_val else "not run"
                    rep_str  = f"{reported:.4f}" if isinstance(reported, (int, float)) else str(reported)
                    gap_str  = ""
                    if our_val and isinstance(reported, (int, float)) and reported > 0:
                        gap_str = f"  ({our_val / reported:.1f}× from paper)"
                    print(f"       {bm:<18}  paper={rep_str:<10}  ours={our_str}{gap_str}")

        print()

    def suggest_next(self, top_n: int = 8) -> list[str]:
        """Generate prioritised list of next experiments from paper registry.

        Returns list of ExperimentConfig snippets ready to copy into experiments.py.
        """
        best = self._load_results()
        suggestions = []

        for paper in self.papers:
            if paper.get("status") != "pending":
                continue
            exps = paper.get("suggested_experiments", [])
            if not isinstance(exps, list):
                continue
            for exp in exps:
                if not isinstance(exp, dict):
                    continue
                suggestions.append({
                    "paper": paper.get("id"),
                    "name":  exp.get("name"),
                    "rationale": exp.get("rationale", ""),
                    "expected":  exp.get("expected", ""),
                })

        # Also suggest H1 loss experiments if h1_loss paper is loaded
        h1_paper = self.get("h1-sobolev-loss")
        if h1_paper:
            best_burgers = best.get("burgers_1d")
            if best_burgers is not None:
                suggestions.append({
                    "paper": "h1-sobolev-loss",
                    "name": "fno_h128_m24_l8_h1",
                    "rationale": f"H1 loss on best FNO config (current best {best_burgers:.4f}). "
                                 "U-FNO paper reports 10% improvement.",
                    "expected": f"~{best_burgers * 0.9:.4f}–{best_burgers * 1.05:.4f}",
                })
            suggestions.append({
                "paper": "h1-sobolev-loss",
                "name": "afno_h128_m24_l8_h1",
                "rationale": "AFNO + H1 loss — non-linear Fourier mixer + frequency-weighted error.",
                "expected": "~0.12–0.14 if AFNO + H1 compound",
            })

        # Print suggestions
        if suggestions:
            print(f"\n{'─'*70}")
            print(f"  Next experiments from paper registry ({len(suggestions)} suggestions):")
            print(f"{'─'*70}")
            for i, s in enumerate(suggestions[:top_n], 1):
                print(f"\n  [{i}] {s['name']}  (from: {s['paper']})")
                print(f"      Rationale: {s['rationale'][:80]}")
                if s['expected']:
                    print(f"      Expected:  {s['expected']}")

        return [s["name"] for s in suggestions[:top_n]]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="SciML paper registry")
    p.add_argument("--gaps",    action="store_true", help="Print SOTA gap table")
    p.add_argument("--suggest", action="store_true", help="Suggest next experiments")
    p.add_argument("--pending", action="store_true", help="List unimplemented papers")
    args = p.parse_args()

    reg = PaperRegistry()

    if args.gaps:
        reg.gap_table()
    elif args.suggest:
        reg.suggest_next()
    elif args.pending:
        pending = reg.pending()
        print(f"\n{len(pending)} pending papers (not yet implemented):")
        for p in pending:
            print(f"  {p['id']}: {p['title'][:60]}")
            exps = p.get("suggested_experiments", [])
            if isinstance(exps, list):
                for e in exps[:2]:
                    if isinstance(e, dict):
                        print(f"    → {e.get('name',''): <30}  {e.get('expected','')}")
    else:
        reg.report()
        reg.gap_table()


if __name__ == "__main__":
    main()
