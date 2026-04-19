# System Architecture

Deep technical reference for the autonomous research loop components.

---

## Two-Mode Operation

The system supports two operating modes that can be mixed within a session:

**Mode A — Human-Guided**  
A human (or AI agent like Claude Code) reads `RESEARCH_BRAIN.md`, interprets results,
edits `experiments.yaml` directly, and invokes `autorun.py` to execute the queue.
The system handles execution, retry, and logging; the human handles strategy.

**Mode B — Fully Autonomous (`agent_loop.py`)**  
`agent_loop.py` performs one full autonomous cycle:
1. Calls `tracker.analyze_lineage()` to build per-benchmark summaries
2. Calls `HypothesisEngine.analyze_benchmark()` for each priority benchmark
3. Calls `BayesianHPO.ask()` to sample hyperparameters
4. Generates new `ExperimentConfig` entries and appends them to `experiments.yaml`
5. Optionally triggers `autorun.py` as a subprocess

Mode B is activated via `autorun.py --auto`. The loop repeats until
`--max-auto-experiments` or `--max-auto-time` is reached.

---

## HypothesisEngine (`core/hypothesis.py`)

The HypothesisEngine reads `results.json` and translates empirical patterns into
actionable next-experiment suggestions.

### Failure Mode Taxonomy

Every completed experiment is classified into one of six failure modes:

| Mode | Detection Logic |
|---|---|
| `gradient_collapse` | `val_l2_rel ≥ 1.0`, crash status, or log keywords: "early stop", "diverged", "collapsed", "nan" |
| `spectral_bias` | High-frequency diagnostic error > 0.3, OR (`val > 0.3` AND `n_modes ≥ 24`) |
| `wrong_inductive_bias` | AFNO or WNO on periodic benchmarks (burgers, kdv, wave) with `val > 0.4` |
| `step_limited` | `val > 0.5` with VRAM usage > 1.0 GB (training cut short by memory pressure) |
| `capacity_limited` | `hidden_dim < 64` and `val > 0.15` (model too small) |
| `nominal` | None of the above |

### `analyze_benchmark(benchmark)` Output

```python
{
    "n_experiments": int,           # total runs on this benchmark
    "best_val": float,              # lowest val_l2_rel seen
    "best_config": {                # hyperparams of the best run
        "model": str,
        "hidden_dim": int,
        "n_layers": int,
        "n_modes": int,
    },
    "worst_model": str,             # model key with highest average error
    "improvement_trend": str,       # "improving" | "degrading" | "plateaued"
                                    # (compares first ⅓ vs last ⅓ of timeline)
    "dominant_failure": str,        # most common failure mode across runs
    "suggested_next": list[tuple],  # [(model_key, rationale_str), ...]
}
```

### `suggest_intervention(benchmark, current_val, best_val)` Decision Tree

```
current_val > 2 × best_val
    → Return best known config (replicate the champion)

best_val < current_val < 1.1 × best_val
    → Increment depth: n_layers + 1

|current_val - best_val| / best_val < 5%
    → Switch model family: cycle through ["RFNO", "FNO", "FFNO", "UNO"]

else
    → Increment modes: n_modes + 4 (capped at 48)
```

Returns:
```python
{
    "model": str,
    "hidden_dim": int,
    "n_layers": int,
    "n_modes": int,
    "loss_type": str,
    "rationale": str,
    "paper_ref": str,
}
```

---

## Bayesian HPO (`core/hpo.py`)

A Gaussian Process surrogate model with Expected Improvement acquisition for
hyperparameter search. Seeded from historical results in `results.json`.

### RBF Kernel

```
k(x, x') = σ² · exp( -‖x - x'‖² / (2l²) )
```

where `σ² = 1.0` (variance) and `l = 0.3` (lengthscale). Both are fixed — no
kernel hyperparameter optimization.

### GP Posterior

Given training observations `{x_i, y_i}` (hyperparams → val_l2_rel):

```
μ(x*) = k_s ᵀ · K⁻¹ · y
σ²(x*) = k(x*, x*) − k_s ᵀ · K⁻¹ · k_s
```

where:
- `K` = Gram matrix of training points + `noise · I` (noise = 1e-6)
- `k_s` = vector of kernel evaluations between `x*` and all training points
- Solved via Cholesky decomposition; falls back to `(mean=1.0, std=1.0)` on
  `LinAlgError`

### Expected Improvement

For minimization (lower `val_l2_rel` is better):

```
z   = (f_best - μ(x*) - ξ) / σ(x*)
EI  = (f_best - μ(x*) - ξ) · Φ(z) + σ(x*) · φ(z)
```

where `ξ = 0.01` (exploration-exploitation trade-off), `Φ` is the standard normal
CDF, and `φ` is the standard normal PDF.

`ask()` samples 1000 random candidates from the search space, evaluates EI for each,
and returns the candidate with the highest EI.

### Search Space

| Hyperparameter | Range | Scale |
|---|---|---|
| `hidden_dim` | 32 – 256 | linear |
| `n_layers` | 2 – 12 | linear |
| `n_modes` | 8 – 32 | linear |
| `lr` | 1e-4 – 1e-2 | log |

### Multi-Objective Extension

`BayesianHPO` supports multi-objective optimization via:

1. **Pareto dominance**: config A dominates B if `A ≤ B` in all objectives and
   `A < B` in at least one. O(n²) algorithm — adequate for n ≤ 1000.
2. **Scalarization**: for `ask()`, objectives are combined as
   `score = Σ (weight_i / total_weight) · signed_metric_i`
   where "maximize" objectives are negated.

```python
hpo = BayesianHPO(benchmark="darcy_2d")
hpo.load_history()

# Single-objective (default)
config = hpo.ask()

# Multi-objective
config = hpo.ask(
    objectives={"val_l2_rel": "minimize", "params_m": "minimize"},
    weights={"val_l2_rel": 0.8, "params_m": 0.2},
)
pareto = hpo.pareto_front()
```

---

## Retry Escalation (`autorun.py`)

When a training run crashes or produces poor results, `autorun.py` escalates through
four recovery levels before giving up:

### Trigger Conditions

| Level | Trigger |
|---|---|
| **r1** | Non-zero exit code (crash, OOM, NaN) |
| **r2** | r1 run also fails |
| **r3** | r2 run also fails |
| **_adapt** | Run completes but `val_l2_rel > 3.0 × benchmark_baseline` |

### Recovery Actions

**r1 — `smart_fix()`**: Parses the crash log via `core/diagnostics.py`. Detects:
- OOM → halve `hidden_dim`
- NaN loss → reduce `lr` by 10×, switch to `h1` loss
- Shape error → flag for manual review

**r2**: Halve `hidden_dim`, `n_layers`, and `n_modes` from the original config;
multiply `lr` by 0.1.

**r3**: Minimal viable config regardless of the original: `hidden_dim=32`,
`n_layers=2`, `n_modes=min(8, original)`, `lr=1e-4`.

**_adapt**: Calls `HypothesisEngine.suggest_intervention()` with the completed
(but poor) result to generate a follow-up experiment automatically.

---

## Trajectory Log (`logs/trajectories.jsonl`)

An RL-style replay buffer. Every action taken by the autonomous loop and its
observed outcome is appended as one JSON line:

```jsonl
{
  "timestamp": "2026-04-19T02:31:00Z",
  "benchmark": "burgers_1d",
  "state": {
    "best_val": 0.208,
    "n_experiments": 12,
    "dominant_failure": "capacity_limited"
  },
  "hypothesis": "capacity_limited: increase hidden_dim",
  "action": {
    "model": "FNO",
    "hidden_dim": 128,
    "n_layers": 8,
    "n_modes": 24,
    "loss_type": "l2_rel"
  },
  "expected_outcome": 0.15,
  "outcome": 0.1468,
  "diag_snapshot": {
    "low_freq_error": 0.08,
    "high_freq_error": 0.23,
    "vram_peak_gb": 0.4
  }
}
```

The log is the foundation for future meta-learning: training a policy that maps
`(state, hypothesis) → action` to improve the autonomous loop itself.

---

## Data Immutability Contract

`data/prepare.py` is **read-only** by convention. It contains the ground-truth
data generators and the evaluation harness with fixed constants:

```python
GRID_SIZE    = 64
N_TRAIN      = 4096
N_VAL        = 256
VAL_SEED     = 42
TRAIN_SEED   = 7
SOLVER_STEPS = 500
```

These constants must never change. All experiments in `results.json` share the
same validation set (same seed, same solver). Changing any constant would make
historical results incomparable — breaking the lineage DAG.

The evaluation function `BENCHMARK_REGISTRY.evaluate()` always calls
`prepare.py`'s fixed solver to generate validation data; it does not accept
pre-generated data files. This guarantees bit-for-bit reproducibility.
