"""Discovery Engine Tracker — DAG-based experiment lineage for SciML autoresearch.

Manages results.json (the SSoT for lineage) and results.tsv (legacy sync).
Tracks branching via parent_id and structured rationale/conclusions.
"""

import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

# ── Path Constants ────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent
RESULTS_JSON = REPO_ROOT / "results.json"
RESULTS_TSV  = REPO_ROOT / "results.tsv"

class Tracker:
    def __init__(self, json_path: Path = RESULTS_JSON, tsv_path: Path = RESULTS_TSV):
        self.json_path = json_path
        self.tsv_path  = tsv_path
        self.experiments = []
        self._load()

    def _load(self):
        """Load JSON lineage.  If missing, migrate from TSV."""
        if self.json_path.exists():
            with open(self.json_path, 'r') as f:
                self.experiments = json.load(f)
        elif self.tsv_path.exists():
            self._migrate_from_tsv()
        else:
            self.experiments = []

    def _save(self):
        """Save JSON lineage and sync to TSV."""
        with open(self.json_path, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        self._sync_to_tsv()

    def _migrate_from_tsv(self):
        """Import legacy TSV data into JSON DAG."""
        import csv
        exps = []
        if not self.tsv_path.exists():
            return
        
        with open(self.tsv_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            # Infer parentage linearly for migration
            last_keep = {} # benchmark -> id
            for i, row in enumerate(reader):
                bm = row['benchmark']
                exp_id = f"legacy_{i}_{row['commit']}"
                parent = last_keep.get(bm)
                
                def safe_float(v, default=1.0):
                    try:
                        return float(v)
                    except:
                        return default

                exp = {
                    "id": exp_id,
                    "parent_id": parent,
                    "timestamp": int(time.time()) - (1000 * (100 - i)), # fake order
                    "benchmark": bm,
                    "model": row['model'],
                    "val_l2_rel": safe_float(row['val_l2_rel']),
                    "memory_gb": safe_float(row['memory_gb'], 0.0),
                    "status": row['status'],
                    "description": row['description'],
                    "commit": row['commit'],
                    "config": {}, # unknown for legacy
                    "rationale": "Migrated from legacy results.tsv",
                    "conclusion": ""
                }
                exps.append(exp)
                if row['status'] == 'keep':
                    last_keep[bm] = exp_id
        
        self.experiments = exps
        self._save()

    def _sync_to_tsv(self):
        """Keep results.tsv updated for backwards compatibility with legacy tools."""
        header = ["commit", "benchmark", "model", "val_l2_rel", "memory_gb", "status", "description"]
        with open(self.tsv_path, 'w') as f:
            f.write("\t".join(header) + "\n")
            for e in self.experiments:
                row = [
                    str(e.get("commit", "none")),
                    str(e["benchmark"]),
                    str(e["model"]),
                    f"{e['val_l2_rel']:.6f}",
                    f"{e['memory_gb']:.2f}",
                    str(e["status"]),
                    str(e["description"])
                ]
                f.write("\t".join(row) + "\n")

    def log_experiment(self,
                       benchmark: str,
                       model: str,
                       val_l2_rel: float,
                       memory_gb: float,
                       status: str,
                       description: str,
                       commit: str,
                       parent_id: Optional[str] = None,
                       parent_name: Optional[str] = None,
                       config: Optional[Dict] = None,
                       rationale: str = "",
                       conclusion: str = "",
                       diag: Optional[Dict] = None):
        """Log a new experiment node to the discovery tree."""
        exp_id = f"{benchmark}_{model}_{int(time.time())}"

        # Resolve parent_id: explicit > by parent_name > last keep for benchmark
        if not parent_id:
            if parent_name:
                for e in reversed(self.experiments):
                    if e.get("description", "").startswith(parent_name):
                        parent_id = e["id"]
                        break
            if not parent_id:
                for e in reversed(self.experiments):
                    if e['benchmark'] == benchmark and e['status'] == 'keep':
                        parent_id = e['id']
                        break

        node = {
            "id": exp_id,
            "parent_id": parent_id,
            "timestamp": int(time.time()),
            "benchmark": benchmark,
            "model": model,
            "val_l2_rel": val_l2_rel,
            "memory_gb": memory_gb,
            "status": status,
            "description": description,
            "commit": commit,
            "config": config or {},
            "rationale": rationale,
            "conclusion": conclusion,
            "diag": diag or {},
        }
        
        self.experiments.append(node)
        self._save()
        return exp_id

    def get_lineage(self) -> List[Dict]:
        return self.experiments

    def get_experiment(self, exp_id: str) -> Optional[Dict]:
        for e in self.experiments:
            if e['id'] == exp_id:
                return e
        return None

    def analyze_lineage(self, benchmark: str = None) -> Dict:
        """Analyze experiment history: HP importance, model ranking, trends."""
        exps = self.experiments
        if benchmark:
            exps = [e for e in exps if e.get("benchmark") == benchmark]

        by_benchmark: Dict[str, list] = {}
        for e in exps:
            b = e.get("benchmark", "unknown")
            by_benchmark.setdefault(b, []).append(e)

        summaries = {}
        for bm, bm_exps in by_benchmark.items():
            valid = [e for e in bm_exps if e.get("val_l2_rel") and 0 < e["val_l2_rel"] < 10.0]
            if not valid:
                continue

            vals = [e["val_l2_rel"] for e in valid]
            best_exp = min(valid, key=lambda e: e["val_l2_rel"])

            # Hyperparameter → val_l2_rel Pearson correlation
            hp_importance = {}
            for field in ["hidden_dim", "n_layers", "n_modes", "lr", "batch_size"]:
                fv, mv = [], []
                for e in valid:
                    cfg = e.get("config") or {}
                    if field in cfg and cfg[field] is not None:
                        try:
                            fv.append(float(cfg[field]))
                            mv.append(e["val_l2_rel"])
                        except (TypeError, ValueError):
                            pass
                if len(fv) >= 3:
                    fa, ma = np.array(fv), np.array(mv)
                    if fa.std() > 0 and ma.std() > 0:
                        hp_importance[field] = round(float(np.corrcoef(fa, ma)[0, 1]), 3)

            # Average val per model
            model_vals: Dict[str, list] = {}
            for e in valid:
                model_vals.setdefault(e.get("model", "?"), []).append(e["val_l2_rel"])
            model_avg = {m: round(sum(vs) / len(vs), 6) for m, vs in model_vals.items()}

            # Trend: is performance improving over time?
            sorted_t = sorted(valid, key=lambda e: e.get("timestamp", 0))
            early  = [e["val_l2_rel"] for e in sorted_t[:5]]
            recent = [e["val_l2_rel"] for e in sorted_t[-5:]]
            trend = "improving" if (early and recent and min(recent) < min(early)) else "plateaued"

            summaries[bm] = {
                "n_experiments": len(valid),
                "best_val": round(best_exp["val_l2_rel"], 6),
                "best_model": best_exp.get("model"),
                "best_description": best_exp.get("description", ""),
                "mean_val": round(float(np.mean(vals)), 6),
                "std_val": round(float(np.std(vals)), 6),
                "hp_importance": hp_importance,
                "model_avg_val": model_avg,
                "trend": trend,
            }

        # Global cross-benchmark patterns
        patterns = []
        model_wins: Dict[str, int] = {}
        for bm, bm_exps in by_benchmark.items():
            valid_bm = [e for e in bm_exps if e.get("val_l2_rel") and 0 < e["val_l2_rel"] < 10.0]
            if valid_bm:
                winner = min(valid_bm, key=lambda e: e["val_l2_rel"])
                m = winner.get("model", "?")
                model_wins[m] = model_wins.get(m, 0) + 1
        if model_wins:
            top = max(model_wins, key=model_wins.get)
            patterns.append(f"{top} wins on {model_wins[top]}/{len(by_benchmark)} benchmarks")

        return {
            "benchmark_summaries": summaries,
            "global_patterns": patterns,
            "total_experiments": len(self.experiments),
            "benchmarks_covered": list(summaries.keys()),
        }

    def print_analysis(self, benchmark: str = None):
        """Print a human-readable analysis report."""
        a = self.analyze_lineage(benchmark)
        print(f"\n{'='*60}")
        print(f"SciML Lineage Analysis  ({a['total_experiments']} total experiments)")
        print(f"{'='*60}")
        for bm, s in a["benchmark_summaries"].items():
            print(f"\n{bm}:")
            print(f"  Experiments : {s['n_experiments']}")
            print(f"  Best        : {s['best_val']:.6f}  ({s['best_model']})")
            print(f"  Trend       : {s['trend']}")
            if s["hp_importance"]:
                top_hp = sorted(s["hp_importance"].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                print(f"  HP impact   : " + ", ".join(f"{k}={v:+.2f}" for k, v in top_hp))
            if s["model_avg_val"]:
                best_m = min(s["model_avg_val"].items(), key=lambda x: x[1])
                print(f"  Best model  : {best_m[0]} (avg {best_m[1]:.4f})")
        if a["global_patterns"]:
            print(f"\nGlobal patterns:")
            for p in a["global_patterns"]:
                print(f"  • {p}")
        print()

if __name__ == "__main__":
    t = Tracker()
    print(f"Discovery Engine Initialized with {len(t.experiments)} experiments.")
    if t.experiments:
        latest = t.experiments[-1]
        print(f"Latest Result: {latest['benchmark']} | {latest['model']} | val={latest['val_l2_rel']:.4f}")
    t.print_analysis()
