"""Bayesian Hyperparameter Optimization for SciML experiments.

Replaces the static grid in experiments.py with adaptive search using a
Gaussian Process surrogate with Expected Improvement acquisition.

Supports multi-objective optimization via weighted scalarization and
Pareto-front extraction when secondary objectives are present.

Usage (single-objective, original):
    hpo = BayesianHPO(benchmark="burgers_1d")
    hpo.load_history()           # seed from results.json
    config = hpo.ask()           # get next config to try
    hpo.tell(config, val=0.15)   # report result
    hpo.suggest_top(n=5)         # print top-N suggestions

Usage (multi-objective: accuracy + memory):
    hpo = BayesianHPO(
        benchmark="burgers_1d",
        objectives=[
            ("val_l2_rel", 1.0, "minimize"),   # primary — accuracy
            ("memory_gb",  0.3, "minimize"),   # secondary — memory
        ],
    )
    hpo.load_history()                         # loads val_l2_rel + memory_gb
    config = hpo.ask()                         # EI on composite score
    hpo.tell_multi(config, {"val_l2_rel": 0.15, "memory_gb": 0.12})
    front = hpo.pareto_front()                 # Pareto-optimal (acc, mem) configs

    uv run bayesian_hpo.py --benchmark burgers_1d --top 5
    uv run bayesian_hpo.py --benchmark burgers_1d --multi --pareto
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from core.utils import REPO_ROOT
RESULTS_JSON = REPO_ROOT / "results.json"

# Default single-objective spec: (metric_name, weight, direction)
_DEFAULT_OBJECTIVES = [("val_l2_rel", 1.0, "minimize")]

# ── Search space (normalized to [0,1] internally) ─────────────────────────────
SEARCH_SPACE = {
    "hidden_dim": {"type": "int",   "low": 32,   "high": 256,  "log": False},
    "n_layers":   {"type": "int",   "low": 2,    "high": 12,   "log": False},
    "n_modes":    {"type": "int",   "low": 8,    "high": 32,   "log": False},
    "lr":         {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
}

PARAM_KEYS = list(SEARCH_SPACE.keys())
N_DIM      = len(PARAM_KEYS)


def _normalize(config: dict) -> np.ndarray:
    """Config dict → normalized [0,1]^N vector."""
    x = np.zeros(N_DIM)
    for i, key in enumerate(PARAM_KEYS):
        sp  = SEARCH_SPACE[key]
        val = float(config.get(key, (sp["low"] + sp["high"]) / 2))
        if sp["log"]:
            lo, hi = math.log(sp["low"]), math.log(sp["high"])
            x[i] = (math.log(max(val, sp["low"])) - lo) / (hi - lo)
        else:
            x[i] = (val - sp["low"]) / (sp["high"] - sp["low"])
    return np.clip(x, 0.0, 1.0)


def _denormalize(x: np.ndarray) -> dict:
    """Normalized vector → config dict."""
    config = {}
    for i, key in enumerate(PARAM_KEYS):
        sp = SEARCH_SPACE[key]
        if sp["log"]:
            lo, hi = math.log(sp["low"]), math.log(sp["high"])
            val = math.exp(lo + x[i] * (hi - lo))
        else:
            val = sp["low"] + x[i] * (sp["high"] - sp["low"])
        config[key] = int(round(val)) if sp["type"] == "int" else float(val)
    return config


# ── Gaussian Process ──────────────────────────────────────────────────────────

def _rbf_kernel(X1: np.ndarray, X2: np.ndarray,
                lengthscale: float = 0.3, variance: float = 1.0) -> np.ndarray:
    """Squared-exponential (RBF) kernel: k(x,x') = σ² exp(-||x-x'||²/2l²)."""
    diff = X1[:, None, :] - X2[None, :, :]          # (n1, n2, d)
    sq   = np.sum(diff ** 2, axis=-1)                # (n1, n2)
    return variance * np.exp(-sq / (2 * lengthscale ** 2))


def _gp_predict(X_train: np.ndarray, y_train: np.ndarray,
                x_new: np.ndarray, noise: float = 0.01) -> Tuple[float, float]:
    """GP posterior mean and std at x_new (shape (d,))."""
    n = len(X_train)
    if n < 2:
        return float(np.mean(y_train)) if n == 1 else 0.5, 1.0

    K    = _rbf_kernel(X_train, X_train) + noise * np.eye(n)
    k_s  = _rbf_kernel(X_train, x_new[None, :]).flatten()   # (n,)
    k_ss = float(_rbf_kernel(x_new[None, :], x_new[None, :])[0, 0])

    try:
        L    = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        v    = np.linalg.solve(L, k_s)
        mu   = float(k_s @ alpha)
        var  = max(k_ss - float(v @ v), 1e-10)
    except np.linalg.LinAlgError:
        mu, var = float(np.mean(y_train)), 1.0

    return mu, math.sqrt(var)


def _expected_improvement(mu: float, sigma: float,
                           best_y: float, xi: float = 0.01) -> float:
    """Expected Improvement acquisition (minimization)."""
    if sigma <= 1e-10:
        return 0.0
    z = (best_y - mu - xi) / sigma
    return (best_y - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)


# ── Main class ────────────────────────────────────────────────────────────────

class BayesianHPO:
    """Per-benchmark Bayesian HPO study backed by results.json.

    When objectives contains more than one entry, the class operates in
    multi-objective mode:
      - `ask()` optimises a weighted composite score (scalarization)
      - `tell_multi()` records per-metric observations
      - `pareto_front()` returns the Pareto-optimal subset of observations

    Single-objective mode is the default and fully backwards-compatible.
    """

    def __init__(self, benchmark: str, model: str = "FNO", random_seed: int = 42,
                 objectives: Optional[List[Tuple[str, float, str]]] = None):
        """
        Args:
            benchmark:  benchmark name (e.g. "burgers_1d")
            model:      model type (e.g. "FNO")
            random_seed: RNG seed for reproducibility
            objectives: list of (metric_name, weight, direction) tuples.
                        direction must be "minimize" or "maximize".
                        Default: [("val_l2_rel", 1.0, "minimize")]
        """
        self.benchmark   = benchmark
        self.model       = model
        self.rng         = np.random.RandomState(random_seed)
        self.objectives  = objectives or list(_DEFAULT_OBJECTIVES)
        self.X: List[np.ndarray] = []   # normalized configs
        self.y: List[float]      = []   # composite scalarized scores (for GP)
        # Per-objective raw values: {metric_name: [v1, v2, ...]}
        self._raw: Dict[str, List[float]] = {obj[0]: [] for obj in self.objectives}

    # ── Scalarization helpers ─────────────────────────────────────────────────

    def _scalarize(self, metrics: Dict[str, float]) -> Optional[float]:
        """Compute weighted composite score (lower = better).

        Returns None if any required objective metric is missing.
        """
        total_weight = sum(w for _, w, _ in self.objectives)
        score = 0.0
        for name, weight, direction in self.objectives:
            val = metrics.get(name)
            if val is None or not math.isfinite(val):
                return None
            # Normalise: flip "maximize" so that lower composite = better
            signed = -val if direction == "maximize" else val
            score += (weight / total_weight) * signed
        return score

    # ── Observations ─────────────────────────────────────────────────────────

    def tell(self, config: dict, val: float) -> None:
        """Record a single-objective observation (val_l2_rel). Backwards-compatible."""
        self.tell_multi(config, {self.objectives[0][0]: val})

    def tell_multi(self, config: dict, metrics: Dict[str, float]) -> None:
        """Record a multi-metric observation.

        Args:
            config:  hyperparameter dict (hidden_dim, n_layers, n_modes, lr)
            metrics: dict of {metric_name: value} — must include the primary
                     objective; secondary objectives are optional.
        """
        score = self._scalarize(metrics)
        if score is None:
            return  # skip incomplete observations
        self.X.append(_normalize(config))
        self.y.append(score)
        for name, _, _ in self.objectives:
            self._raw[name].append(metrics.get(name, float("nan")))

    def load_history(self, results_path: Path = RESULTS_JSON) -> int:
        """Seed from results.json; return number of observations loaded.

        Loads all objective metrics present in the results file. For legacy
        entries without a `config` block, parses h=/l=/m= from description.
        """
        if not results_path.exists():
            return 0
        with open(results_path) as f:
            data = json.load(f)
        loaded = 0
        obj_names = {obj[0] for obj in self.objectives}
        for e in data:
            if e.get("benchmark") != self.benchmark:
                continue
            if e.get("model") != self.model:
                continue
            primary = self.objectives[0][0]
            primary_val = e.get(primary) or e.get("val_l2_rel")
            if not primary_val or not (0 < primary_val < 5.0):
                continue
            cfg = e.get("config") or {}
            if not cfg:
                import re
                desc = e.get("description", "")
                for key, pat in [("hidden_dim", r"h=(\d+)"),
                                 ("n_layers",   r"l=(\d+)"),
                                 ("n_modes",    r"m=(\d+)")]:
                    m = re.search(pat, desc)
                    if m:
                        cfg[key] = int(m.group(1))
            if not cfg:
                continue
            # Collect all available objective values
            metrics: Dict[str, float] = {}
            for name in obj_names:
                v = e.get(name)
                if v is not None and math.isfinite(float(v)):
                    metrics[name] = float(v)
            # Primary is required
            if primary not in metrics:
                metrics[primary] = primary_val
            self.tell_multi(cfg, metrics)
            loaded += 1
        return loaded

    # ── Suggestion ───────────────────────────────────────────────────────────

    def ask(self, n_candidates: int = 500) -> dict:
        """Return the next config to try (highest Expected Improvement).

        In multi-objective mode the GP is fitted to the composite scalarized
        score, so EI balances all objectives according to their weights.
        """
        candidates = self.rng.rand(n_candidates, N_DIM)

        # Pure exploration for the first few runs
        if len(self.y) < 3:
            return _denormalize(candidates[0])

        X_arr  = np.array(self.X)
        y_arr  = np.array(self.y)
        best_y = float(np.min(y_arr))

        best_ei, best_x = -1.0, candidates[0]
        for x in candidates:
            mu, sigma = _gp_predict(X_arr, y_arr, x)
            ei = _expected_improvement(mu, sigma, best_y)
            if ei > best_ei:
                best_ei, best_x = ei, x

        return _denormalize(best_x)

    def predict(self, config: dict) -> Tuple[float, float]:
        """Return GP posterior (mean, std) for a given config."""
        if len(self.y) < 2:
            return 0.5, 1.0
        return _gp_predict(np.array(self.X), np.array(self.y), _normalize(config))

    def suggest_top(self, n: int = 5) -> List[dict]:
        """Print and return top-N configs by predicted mean (exploitation)."""
        candidates = self.rng.rand(1000, N_DIM)
        scored = []
        X_arr = np.array(self.X) if self.X else None
        y_arr = np.array(self.y) if self.y else None

        for x in candidates:
            config = _denormalize(x)
            if X_arr is not None and len(y_arr) >= 2:
                mu, sigma = _gp_predict(X_arr, y_arr, x)
            else:
                mu, sigma = 0.5, 1.0
            scored.append((mu, sigma, config))

        scored.sort(key=lambda t: t[0])
        top = scored[:n]

        obj_label = (f"composite({','.join(o[0] for o in self.objectives)})"
                     if len(self.objectives) > 1 else self.objectives[0][0])
        print(f"\nTop-{n} configs for {self.benchmark}/{self.model} "
              f"({obj_label}, {len(self.y)} obs):")
        print(f"{'Rank':>4}  {'Pred mean':>10}  {'Pred std':>9}  Config")
        print("─" * 70)
        for rank, (mu, sigma, cfg) in enumerate(top, 1):
            parts = "  ".join(f"{k}={v}" for k, v in cfg.items())
            print(f"{rank:>4}  {mu:>10.6f}  {sigma:>9.6f}  {parts}")
        print()

        return [cfg for _, _, cfg in top]

    # ── Multi-objective: Pareto front ─────────────────────────────────────────

    def pareto_front(self) -> List[Dict]:
        """Return the Pareto-optimal subset of observed configurations.

        A configuration is Pareto-optimal (non-dominated) if no other
        observed configuration is strictly better in all objectives.

        Returns a list of dicts, each containing:
            "config": {hidden_dim, n_layers, n_modes, lr}
            <metric_name>: observed value for each objective
        """
        if not self.X:
            return []

        obj_names  = [obj[0] for obj in self.objectives]
        directions = [obj[2] for obj in self.objectives]
        n_obs      = len(self.X)

        # Build objective matrix (n_obs × n_obj); flip "maximize" → lower = better
        matrix = np.full((n_obs, len(obj_names)), np.nan)
        for j, (name, direction) in enumerate(zip(obj_names, directions)):
            vals = self._raw.get(name, [])
            for i, v in enumerate(vals):
                if i < n_obs and math.isfinite(v):
                    matrix[i, j] = -v if direction == "maximize" else v

        # Rows with any NaN cannot be compared — exclude from front
        valid_mask = ~np.any(np.isnan(matrix), axis=1)
        valid_idx  = np.where(valid_mask)[0]

        if len(valid_idx) == 0:
            return []

        valid_matrix = matrix[valid_idx]
        n_valid = len(valid_idx)

        # O(n²) dominance check (adequate for typical HPO study size ≤ 1000)
        dominated = np.zeros(n_valid, dtype=bool)
        for i in range(n_valid):
            for j in range(n_valid):
                if i == j:
                    continue
                # j dominates i if j ≤ i in all objectives and j < i in at least one
                if (np.all(valid_matrix[j] <= valid_matrix[i])
                        and np.any(valid_matrix[j] < valid_matrix[i])):
                    dominated[i] = True
                    break

        front_idx = valid_idx[~dominated]

        results = []
        for idx in front_idx:
            entry: Dict = {"config": _denormalize(self.X[idx])}
            for name, direction in zip(obj_names, directions):
                raw_vals = self._raw.get(name, [])
                v = raw_vals[idx] if idx < len(raw_vals) else float("nan")
                entry[name] = v
            results.append(entry)

        # Sort by primary objective ascending (lower = better after minimize flip)
        primary = obj_names[0]
        results.sort(key=lambda r: r.get(primary, float("inf")))
        return results

    def print_pareto_front(self) -> None:
        """Print a formatted Pareto-front table."""
        front = self.pareto_front()
        if not front:
            print(f"  No Pareto front available for {self.benchmark}/{self.model} "
                  f"(need ≥1 complete multi-objective observation).")
            return

        obj_names = [obj[0] for obj in self.objectives]
        print(f"\nPareto front for {self.benchmark}/{self.model} "
              f"({len(front)} non-dominated configs):")
        header = "  ".join(f"{n:>12}" for n in obj_names)
        print(f"  {header}  Config")
        print("─" * (14 * len(obj_names) + 50))
        for row in front:
            vals = "  ".join(f"{row.get(n, float('nan')):>12.6f}" for n in obj_names)
            cfg  = row["config"]
            parts = " ".join(f"{k}={v}" for k, v in cfg.items())
            print(f"  {vals}  {parts}")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Bayesian HPO for SciML experiments")
    p.add_argument("--benchmark", default="burgers_1d")
    p.add_argument("--model",     default="FNO")
    p.add_argument("--top",       type=int, default=5)
    p.add_argument("--multi",     action="store_true",
                   help="Enable multi-objective mode (val_l2_rel + memory_gb)")
    p.add_argument("--pareto",    action="store_true",
                   help="Print Pareto front (requires --multi)")
    p.add_argument("--mem-weight", type=float, default=0.3, dest="mem_weight",
                   help="Weight for memory_gb objective (default 0.3)")
    args = p.parse_args()

    if args.multi:
        objectives = [
            ("val_l2_rel", 1.0,            "minimize"),
            ("memory_gb",  args.mem_weight, "minimize"),
        ]
    else:
        objectives = None

    hpo = BayesianHPO(args.benchmark, args.model, objectives=objectives)
    n   = hpo.load_history()
    print(f"Loaded {n} observations for {args.benchmark}/{args.model}")
    if hpo.y:
        primary_name = hpo.objectives[0][0]
        raw_primary  = hpo._raw.get(primary_name, [])
        if raw_primary:
            valid = [v for v in raw_primary if math.isfinite(v)]
            if valid:
                print(f"Best {primary_name}: {min(valid):.6f}")
                print(f"Mean {primary_name}: {sum(valid)/len(valid):.6f}")
        if args.multi and hpo._raw.get("memory_gb"):
            mem_vals = [v for v in hpo._raw["memory_gb"] if math.isfinite(v)]
            if mem_vals:
                print(f"Best memory_gb: {min(mem_vals):.4f} GB")

    hpo.suggest_top(args.top)

    if args.pareto:
        hpo.print_pareto_front()

    print("Next config to try (EI-optimal):")
    next_cfg = hpo.ask()
    for k, v in next_cfg.items():
        print(f"  {k}: {v}")
