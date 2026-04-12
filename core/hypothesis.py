"""Hypothesis Engine — Data-driven scientific reasoning for SciML experiment branching.

Reads results.json to identify failure patterns and suggests concrete interventions
backed by papers/*.yaml literature.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from core.utils import REPO_ROOT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_config_from_description(desc: str) -> dict:
    """Extract h/l/m values from legacy description strings like 'FNO  h=128  l=8  m=24'."""
    cfg = {}
    for key, field in [("h", "hidden_dim"), ("l", "n_layers"), ("m", "n_modes")]:
        m = re.search(rf"{key}=(\d+)", desc)
        if m:
            cfg[field] = int(m.group(1))
    return cfg


def _effective_config(exp: dict) -> dict:
    """Return config dict, falling back to description parsing for legacy entries."""
    cfg = dict(exp.get("config") or {})
    if not cfg:
        cfg = _parse_config_from_description(exp.get("description", ""))
    return cfg


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class HypothesisEngine:
    def __init__(
        self,
        results_path: str = None,
        papers_dir: str = None,
    ):
        if results_path is None: results_path = REPO_ROOT / "results.json"
        if papers_dir is None: papers_dir = REPO_ROOT / "docs" / "papers"
        results_path = Path(results_path)
        self.experiments: list[dict] = []
        if results_path.exists():
            with open(results_path) as f:
                self.experiments = json.load(f)

        self.papers: dict[str, dict] = {}
        papers_dir = Path(papers_dir)
        if papers_dir.is_dir():
            try:
                import yaml
                for p in papers_dir.glob("*.yaml"):
                    try:
                        with open(p) as f:
                            data = yaml.safe_load(f)
                        if data:
                            key = data.get("model_class") or p.stem
                            self.papers[key] = data
                    except Exception:
                        pass
            except ImportError:
                pass  # yaml not available

    # ------------------------------------------------------------------
    # 1. Benchmark history
    # ------------------------------------------------------------------

    def get_benchmark_history(self, benchmark: str) -> list:
        """Return all experiments for the given benchmark, sorted by timestamp."""
        exps = [e for e in self.experiments if e.get("benchmark") == benchmark]
        return sorted(exps, key=lambda e: e.get("timestamp", 0))

    # ------------------------------------------------------------------
    # 2. Failure mode detection
    # ------------------------------------------------------------------

    def detect_failure_mode(self, exp: dict) -> str:
        """Classify a single experiment's failure mode."""
        val = exp.get("val_l2_rel", 1.0)
        model = exp.get("model", "")
        mem = exp.get("memory_gb", 0.0)
        conclusion = (exp.get("conclusion") or "").lower()
        description = (exp.get("description") or "").lower()
        combined = conclusion + " " + description

        # Gradient collapse: val == 1.0, crash status, or early-stop mentions
        if val >= 1.0 or exp.get("status") == "crash":
            return "gradient_collapse"
        if any(kw in combined for kw in ("early stop", "diverged", "collapsed", "nan")):
            return "gradient_collapse"

        # Spectral bias: conclusion flags high-freq diagnostics, or high val with many modes
        if "diag_high_freq" in conclusion:
            m = re.search(r"diag_high_freq[^0-9]*([0-9.]+)", conclusion)
            if m and float(m.group(1)) > 0.3:
                return "spectral_bias"
        cfg = _effective_config(exp)
        n_modes = cfg.get("n_modes", 0)
        if val > 0.3 and n_modes >= 24:
            return "spectral_bias"

        # Wrong inductive bias: AFNO or WNO on periodic benchmarks with high error
        if model in ("AFNO", "WNO") and val > 0.4:
            return "wrong_inductive_bias"

        # Step-limited: large 2D model (high memory) with high error
        if val > 0.5 and mem > 1.0:
            return "step_limited"

        # Capacity-limited: small hidden dim, val close-but-not-matching best on benchmark
        hidden = cfg.get("hidden_dim", 128)
        if hidden < 64 and val > 0.15:
            return "capacity_limited"

        return "nominal"

    # ------------------------------------------------------------------
    # 3. Benchmark-level analysis
    # ------------------------------------------------------------------

    def analyze_benchmark(self, benchmark: str) -> dict:
        """Return a summary dict for a benchmark."""
        history = self.get_benchmark_history(benchmark)
        if not history:
            return {
                "benchmark": benchmark,
                "n_experiments": 0,
                "best_val": None,
                "best_config": {},
                "worst_model": None,
                "improvement_trend": "no_data",
                "dominant_failure": "no_data",
                "suggested_next": [],
            }

        valid = [e for e in history if e.get("val_l2_rel") is not None]
        best_exp = min(valid, key=lambda e: e.get("val_l2_rel", 1e9))
        best_val = best_exp.get("val_l2_rel")
        best_config = _effective_config(best_exp)
        best_config["model"] = best_exp.get("model")

        # Worst model (average val per model type)
        model_vals: dict[str, list] = defaultdict(list)
        for e in valid:
            model_vals[e.get("model", "unknown")].append(e.get("val_l2_rel", 1.0))
        worst_model = max(model_vals, key=lambda m: sum(model_vals[m]) / len(model_vals[m]))

        # Improvement trend: compare first third vs last third of timeline
        n = len(valid)
        if n < 3:
            trend = "no_data"
        else:
            first_avg = sum(e["val_l2_rel"] for e in valid[: n // 3]) / (n // 3)
            last_avg = sum(e["val_l2_rel"] for e in valid[-(n // 3) :]) / (n // 3)
            if last_avg < first_avg * 0.95:
                trend = "improving"
            elif last_avg > first_avg * 1.05:
                trend = "degrading"
            else:
                trend = "plateaued"

        # Dominant failure mode
        failure_counts: dict[str, int] = defaultdict(int)
        for e in valid:
            failure_counts[self.detect_failure_mode(e)] += 1
        dominant_failure = max(failure_counts, key=lambda k: failure_counts[k])

        # Suggested next experiments (simple heuristic)
        suggested = self._suggest_next(benchmark, best_val, best_config, valid)

        return {
            "benchmark": benchmark,
            "n_experiments": len(history),
            "best_val": best_val,
            "best_config": best_config,
            "worst_model": worst_model,
            "improvement_trend": trend,
            "dominant_failure": dominant_failure,
            "suggested_next": suggested,
        }

    def _suggest_next(
        self,
        benchmark: str,
        best_val: float,
        best_config: dict,
        all_exps: list,
    ) -> list[tuple[str, str]]:
        """Return a short list of (model, rationale) suggestions."""
        suggestions = []
        tried_models = {e.get("model") for e in all_exps}
        best_model = best_config.get("model", "FNO")
        n_layers = best_config.get("n_layers", 8)
        n_modes = best_config.get("n_modes", 24)

        # If RFNO not yet tried, recommend it for depth stability
        if "RFNO" not in tried_models:
            suggestions.append(("RFNO", "Pre-LN residuals unlock deeper stacks (l>=10)"))

        # If best is FNO and we're not near SOTA, try increasing depth by 2
        if best_model == "FNO" and best_val > 0.05 and n_layers < 10:
            suggestions.append(
                (best_model, f"Increment depth l={n_layers}→{n_layers+2} (step-budget permitting)")
            )

        # If best is RFNO and val is still high, try more modes
        if best_model == "RFNO" and best_val > 0.05 and n_modes < 32:
            suggestions.append(
                (best_model, f"Increment modes m={n_modes}→{n_modes+4} for finer spectral resolution")
            )

        # If plateau, try H1 loss
        if best_val < 0.25 and best_val > 0.01:
            suggestions.append(
                (best_model, "Apply H1 Sobolev loss (--loss h1) to target derivative errors")
            )

        # If no UNO tried for 2D benchmarks
        if "2d" in benchmark and "UNO" not in tried_models:
            suggestions.append(("UNO", "U-shaped encoder-decoder for multi-scale 2D features"))

        return suggestions[:4]  # cap at 4

    # ------------------------------------------------------------------
    # 4. Targeted intervention
    # ------------------------------------------------------------------

    def suggest_intervention(
        self,
        benchmark: str,
        current_val: float,
        diagnostics: dict | None = None,
    ) -> dict:
        """Return an ExperimentConfig-compatible dict with a concrete intervention."""
        diagnostics = diagnostics or {}
        history = self.get_benchmark_history(benchmark)
        valid = [e for e in history if e.get("val_l2_rel") is not None]

        if not valid:
            # No history at all: recommend sensible default
            return {
                "model": "FNO",
                "hidden_dim": 128,
                "n_layers": 8,
                "n_modes": 24,
                "loss_type": "l2_rel",
                "rationale": "No prior experiments found. Starting with FNO h=128 l=8 m=24 baseline.",
            }

        best_exp = min(valid, key=lambda e: e.get("val_l2_rel", 1e9))
        best_val = best_exp.get("val_l2_rel", 1.0)
        best_cfg = _effective_config(best_exp)
        best_model = best_exp.get("model", "FNO")
        n_layers = best_cfg.get("n_layers", 8)
        n_modes = best_cfg.get("n_modes", 24)
        hidden_dim = best_cfg.get("hidden_dim", 128)

        loss_type = "l2_rel"
        if diagnostics.get("diag_high_freq_error", 0) > 0.3:
            loss_type = "h1"

        # --- Decision tree based on current_val relative to best ---

        if best_val > 0 and current_val > 2 * best_val:
            # Likely misconfigured — replicate the best known config
            suggestion = {
                "model": best_model,
                "hidden_dim": hidden_dim,
                "n_layers": n_layers,
                "n_modes": n_modes,
                "loss_type": loss_type,
                "rationale": (
                    f"current_val ({current_val:.4f}) is >2x best ({best_val:.4f}). "
                    f"Replicating best known config: {best_model} h={hidden_dim} l={n_layers} m={n_modes}."
                ),
            }

        elif best_val > 0 and current_val > best_val and current_val < 1.1 * best_val:
            # Incremental improvement: try +1 layer or +4 modes
            new_layers = n_layers + 1
            suggestion = {
                "model": best_model,
                "hidden_dim": hidden_dim,
                "n_layers": new_layers,
                "n_modes": n_modes,
                "loss_type": loss_type,
                "rationale": (
                    f"current_val ({current_val:.4f}) marginally above best ({best_val:.4f}). "
                    f"Incrementing depth to l={new_layers} for incremental gain."
                ),
            }

        elif best_val > 0 and abs(current_val - best_val) / best_val < 0.05:
            # Within 5% of best: suggest a different model family
            tried_models = {e.get("model") for e in valid}
            alternatives = ["RFNO", "FNO", "FFNO", "UNO"]
            new_model = next((m for m in alternatives if m not in tried_models), "RFNO")
            suggestion = {
                "model": new_model,
                "hidden_dim": hidden_dim,
                "n_layers": n_layers,
                "n_modes": n_modes,
                "loss_type": loss_type,
                "rationale": (
                    f"current_val ({current_val:.4f}) within 5% of best ({best_val:.4f}). "
                    f"Exploring different model family: {new_model}."
                ),
            }

        else:
            # General case: increment modes
            new_modes = min(n_modes + 4, 48)
            suggestion = {
                "model": best_model,
                "hidden_dim": hidden_dim,
                "n_layers": n_layers,
                "n_modes": new_modes,
                "loss_type": loss_type,
                "rationale": (
                    f"Incrementing modes to m={new_modes} based on best config "
                    f"({best_model} h={hidden_dim} l={n_layers} m={n_modes}, val={best_val:.4f})."
                ),
            }

        # Cross-reference papers for the suggested model
        paper = self.papers.get(suggestion["model"])
        if paper:
            suggestion["paper_ref"] = paper.get("arxiv") or paper.get("id") or paper.get("title", "")

        return suggestion

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------

    def print_report(self, benchmark: str | None = None):
        """Print a readable analysis report."""
        benchmarks = (
            [benchmark]
            if benchmark
            else sorted({e.get("benchmark") for e in self.experiments if e.get("benchmark")})
        )

        print("=" * 70)
        print("HYPOTHESIS ENGINE — BENCHMARK ANALYSIS REPORT")
        print("=" * 70)
        print(f"Total experiments loaded: {len(self.experiments)}")
        print(f"Papers loaded: {len(self.papers)}")
        print()

        for bm in benchmarks:
            analysis = self.analyze_benchmark(bm)
            print(f"Benchmark : {analysis['benchmark']}")
            print(f"  Runs    : {analysis['n_experiments']}")
            if analysis["best_val"] is not None:
                print(f"  Best val: {analysis['best_val']:.6f}  (config: {analysis['best_config']})")
            else:
                print("  Best val: N/A")
            print(f"  Trend   : {analysis['improvement_trend']}")
            print(f"  Failure : {analysis['dominant_failure']}")
            print(f"  Worst M : {analysis['worst_model']}")
            if analysis["suggested_next"]:
                print("  Suggestions:")
                for model, rationale in analysis["suggested_next"]:
                    print(f"    [{model}] {rationale}")
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Hypothesis Engine — data-driven SciML experiment advisor")
    p.add_argument("--benchmark", default=None, help="Analyze a specific benchmark (default: all)")
    p.add_argument(
        "--intervene",
        metavar="VAL",
        type=float,
        default=None,
        help="With --benchmark: suggest intervention given current val",
    )
    args = p.parse_args()

    engine = HypothesisEngine()

    if args.intervene is not None and args.benchmark:
        result = engine.suggest_intervention(args.benchmark, args.intervene)
        print("Suggested intervention:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        engine.print_report(args.benchmark)
