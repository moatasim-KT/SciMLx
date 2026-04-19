# SKILL.md — SciML AutoResearch Living Skill Reference

> **This document is self-evolving.** Every hurdle encountered, mitigation
> applied, and insight extracted from `logs/trajectories.jsonl` must be
> distilled back into the relevant section below. See
> [Section 10: How to Update This Document](#10-how-to-update-this-document)
> for the protocol. Do not let a hard-won lesson die in a log file.

---

## Table of Contents

1. [Running the Research Loop](#1-running-the-research-loop)
2. [Adding a Research Paper](#2-adding-a-research-paper)
3. [Adding a Benchmark](#3-adding-a-benchmark)
4. [Scaffolding a New Model](#4-scaffolding-a-new-model)
5. [Extending Existing Models](#5-extending-existing-models)
6. [Loss Functions — Full Decision System](#6-loss-functions--full-decision-system)
7. [Parameter Optimization Playbook](#7-parameter-optimization-playbook)
8. [RL Trajectory Protocol — and How It Feeds Back Here](#8-rl-trajectory-protocol--and-how-it-feeds-back-here)
9. [Invariants & Hard Constraints](#9-invariants--hard-constraints)
10. [How to Update This Document](#10-how-to-update-this-document)
11. [Quality Checklist](#11-quality-checklist)
12. [Learned Lessons Log](#12-learned-lessons-log)
13. [Agent-First Reasoning Protocol (v2.0)](#13-agent-first-reasoning-protocol-v20)
14. [Multi-Metric Monitoring Protocol (v2.0)](#14-multi-metric-monitoring-protocol-v20)

---

## 1. Running the Research Loop

### Standard session (interactive)

```bash
# 0. One-time setup per machine
uv sync
uv run data/prefetch_data.py --skip-slow    # ~2 min; caches PDE datasets

# 1. Read current state — always start here, never skip
uv run analyze.py --papers

# 2. Get ranked next actions with Adversarial Review
uv run auto_suggest.py --generate

# 3. Handle [AGENT] requests in auto_suggest output (see §13)

# 4. Edit experiments.yaml (add or reprioritize)

# 5. Run priority-1 queue with Scientific Debugging
uv run autorun.py --priority 1 --commit

# 6. Handle [AGENT] requests in scientific debug logs (see §13)
```

### Overnight / unattended run

```bash
uv run autorun.py --auto --commit \
    --max-auto-experiments 20 \
    --max-auto-time 10800       # 3-hour wall clock cap
```

`--auto` invokes `agent_loop.py --top 5` (Bayesian HPO) whenever the queue
empties. If that produces no new experiments, it falls back to
`BayesianHPO.suggest_top(3)` which injects HPO-generated configs directly —
ensuring the loop never stalls. Recurses until the time or experiment cap is hit.

### Filtering runs

```bash
uv run autorun.py --priority 1 --benchmark burgers_1d   # one benchmark only
uv run autorun.py --priority 1 --model RFNO             # one model only
uv run autorun.py --dry-run                             # preview without running
```

### Single experiment (quick test before queuing)

```bash
uv run train.py \
    --benchmark burgers_1d \
    --model FNO \
    --hidden 128 \
    --layers 8 \
    --modes 24 \
    --loss h1 \
    --budget 300 \
    --lr_schedule cosine \   # warmup_cosine (default), cosine, onecycle, none
    --seed 42                # default 42; set for reproducibility
```

### Pause / resume / kill

```bash
touch .autorun_pause            # pauses after current experiment finishes
rm .autorun_pause               # resumes
curl -X POST http://localhost:8000/api/kill/<name>   # kill specific run
```

### Live dashboard

```bash
python3 dashboard/app.py
# open dashboard/ui/dashboard.html
# shows: live loss curve, VRAM usage, kill button, lineage DAG, log tail
```

### Reading diagnostics after a run

```bash
grep "^diag_" logs/<name>.log          # spectral error breakdown
tail -100 logs/<name>.log              # last 100 log lines
uv run analyze.py --papers             # compare new result to SOTA
```

### Checking the experiment lineage

```bash
# Raw DAG
python3 -c "
import json
with open('results.json') as f: r = json.load(f)
for e in sorted(r, key=lambda x: x.get('val_l2_rel', 1)):
    print(e['val_l2_rel'], e['name'], e.get('parent_name',''))
" | head -20

# Paper registry gap table
uv run -m core.paper_registry --gaps

# Hypothesis engine full report
python3 -m core.hypothesis
```

### Bayesian HPO (after ≥15 runs on a benchmark)

```bash
uv run -m core.hpo --benchmark burgers_1d --top 5
```

The GP surrogate over `(n_modes, hidden_dim, n_layers, lr)` becomes
meaningfully accurate after ~15 experiments. Before that, use the
empirical priors in Section 7.

---

## 2. Adding a Research Paper

### What the registry does

`docs/papers/*.yaml` is the literature database. `auto_suggest.py` reads it
and ranks pending ideas by expected improvement. `analyze.py --papers`
cross-references it with live `results.json` to expose SOTA gaps.
`core/hypothesis.py` uses the `model_class` field to surface paper references
in intervention suggestions.

### Step-by-step

**Step 1 — Create the YAML file**

```bash
touch docs/papers/<first_author_keyword>_<year>.yaml
```

**Step 2 — Fill in the required schema**

```yaml
id: <keyword>-<year>                         # kebab-case, globally unique
title: "Full paper title"
arxiv: "XXXX.XXXXX"                          # arXiv ID only, no URL
venue: "NeurIPS 2024"
authors: ["First Author", "et al."]
year: 2024

key_idea: >
  One paragraph. State the mathematical operation that is new and why it
  should outperform FNO on which class of PDEs. Be specific about the
  inductive bias (spectral, wavelet, attention, SSM, physics-informed).

architecture: ModelClassName                  # exact Python class name
model_class: RegistryKey                      # key used in experiments.yaml
status: pending                               # pending | partial | implemented | failed

benchmarks:
  burgers_1d:
    reported_val_l2_rel: 0.XXXX
    paper_epochs: 500
    notes: "Dataset differences, normalisation quirks, grid size."
  darcy_2d:
    reported_val_l2_rel: 0.XXXX
    notes: ""

key_hyperparams:
  n_modes: 16
  hidden_dim: 64
  n_layers: 4
  lr: 1e-3
  batch_size: 20

implementation_notes: >
  What MLX building blocks are needed. Call out ops that MLX lacks
  natively (flash attention, sparse ops, complex einsum) and
  propose workarounds using allowed primitives (see Section 4).

difficulty: 5     # 1–10: implementation complexity on Apple Silicon

suggested_experiments:
  - name: <model>_<benchmark>_h<H>_l<L>     # globally unique
    benchmark: burgers_1d
    model: ModelClassName
    hidden_dim: 64
    n_layers: 4
    n_modes: 16
    lr: 0.001
    batch_size: 32
    budget_s: 300
    priority: 2
    rationale: "One-line hypothesis."
    expected: "~0.05–0.10 (reason)"

verdict: >
  Is this worth implementing? On which benchmark? What is the
  risk of step-time or OOM failure on Apple Silicon?
```

**Step 3 — Verify the registry picks it up**

```bash
uv run -m core.paper_registry --pending    # new paper should appear
uv run -m core.paper_registry --gaps       # new SOTA targets should appear
```

**Step 4 — Update `status` and `our_best` as results arrive**

```yaml
status: implemented
benchmarks:
  burgers_1d:
    our_best: 0.0842          # fill in after first run
    gap_ratio: 27.2           # our_best / reported_val_l2_rel
```

### Naming conventions

| Field | Convention | Example |
|-------|-----------|---------|
| File | `<keyword>_<year>.yaml` | `mambano_2024.yaml` |
| `id` | kebab-case | `mamba-no-2024` |
| `model_class` / `architecture` | PascalCase Python class | `MambaNO` |

---

## 3. Adding a Benchmark

### When to add

- A registered paper reports on a PDE not in `docs/SOTA.md`
- The new PDE tests a distinct physical property (non-periodic BCs,
  stochastic forcing, hyperbolic vs parabolic regime, multi-physics)
- You need a resolution-invariance stress test at a different grid size

### File touch-points

| File | What to change |
|------|---------------|
| `data/prepare.py` | **READ-ONLY.** Coordinate with maintainer for new solvers. |
| `data/simulations/` | New solver module; must not alter `prepare.py`'s public API |
| `core/research_plugins.py` | Register the benchmark loader |
| `docs/SOTA.md` | Add a new benchmark block |
| `docs/papers/*.yaml` | Add the new key to existing `benchmarks:` blocks where relevant |
| `WIKI.md` | Add row to the Benchmark Catalog table |

### Registering a loader

```python
# core/research_plugins.py — add inside _register_all_benchmarks()

@BENCHMARK_REGISTRY.register("heat_1d")
def _load_heat_1d(n_train=1000, n_test=200, seed=42, **kwargs):
    from data.prepare import generate_heat_1d
    return generate_heat_1d(n_train=n_train, n_test=n_test, seed=seed)
```

### SOTA entry format (`docs/SOTA.md`)

```markdown
## <PDE Name> (<description, grid, params>)

| Model          | Relative L2 | Notes                    |
|----------------|-------------|--------------------------|
| FNO (paper)    | 0.XXXX      | Li et al. 2020           |
| **This repo**  | **null**    | Not yet run              |

**Gap to SOTA:** Unknown — baseline needed.
**Constraints:** List OOM risk, solver instability, non-periodic BCs.
**2D Constraint (if applicable):** `hidden_dim ≤ 32`, `n_layers ≤ 4`,
`n_modes ≤ 12`, `budget_s ≥ 480`.
```

### Validation requirement

A new benchmark is not usable until it produces a fixed validation set
with `seed=42` and a callable `evaluate_l2_rel(pred, gt)`. These must
live in `data/prepare.py` via maintainer merge. Do not write a parallel
evaluator.

---

## 4. Scaffolding a New Model

### Guiding principle: read trajectories before designing

Before scaffolding, read `logs/trajectories.jsonl` and filter by
benchmark. Look for:

- Which failure modes have been dominant (spectral bias, gradient
  collapse, capacity-limited, step-limited)?
- Which architectural interventions produced the best δ improvements?
- Are there repeated `"hypothesis"` patterns that suggest a direction?

```bash
python3 -c "
import json
with open('logs/trajectories.jsonl') as f:
    entries = [json.loads(l) for l in f if l.strip()]
# Filter to benchmark of interest
for e in entries:
    if 'burgers' in e.get('state',''):
        print(e.get('hypothesis',''), '|', e.get('outcome',''))
"
```

Only scaffold a new architecture if the trajectory log shows that:
1. Existing models have plateaued (≥3 experiments within 5% of each other)
2. The hypothesis engine confirms a structural mismatch (not just HPO gap)
3. The paper YAML for the target architecture has `difficulty ≤ 7`

### Gated workflow — always use the scaffold gate

Never add a model file manually. The gate runs three checks before the
model enters the registry:

```
Syntax check → Import check → Shape smoke test
```

Bypassing the gate causes silent failures in `autorun.py`.

```bash
# 1. Generate stub
uv run -m core.scaffold --stub MyModel --base FNO \
    --notes "Add windowed attention after each spectral block"
# → writes models/mymodel.py

# 2. Implement in models/mymodel.py (see Allowed Ops below)

# 3. Validate (syntax + import + smoke test)
uv run -m core.scaffold --validate MyModel models/mymodel.py

# 4. Register (wires __init__.py + research_plugins.py + experiments.yaml)
uv run -m core.scaffold --register MyModel models/mymodel.py \
    --benchmarks burgers_1d kdv_1d
```

Registration automatically:
- Adds to `models/__init__.py` exports
- Adds to `core/research_plugins.py` `MODEL_REGISTRY`
- Appends priority-3 starter entries to `experiments.yaml`

### Allowed operations (Apple Silicon MLX)

| Allowed | Forbidden |
|---------|-----------|
| `nn.Linear` projections | 3D convolutions |
| Spectral gating: `mx.fft.rfft` → `W·x` → `mx.fft.irfft` | Full 2D self-attention on grids > 16×16 |
| Windowed 1D attention (window ≤ 32 tokens) | Any op requiring > 4 GB unified memory |
| 1D and 2D convolutions (`nn.Conv1d`, `nn.Conv2d`) | External CUDA kernels |
| Haar wavelet transforms (copy from `models/wno.py`) | New pip packages |
| SSM / S4D recurrence (copy from `models/s4d.py`) | Modifying `data/prepare.py` |
| `SpectralConv1d` / `SpectralConv2d` from `models/fno.py` | |
| `nn.LayerNorm`, `nn.GroupNorm` | |
| Gated linear units (`nn.GELU`, manual sigmoid gate) | |

### Attention on Apple Silicon

Full 2D self-attention on N×N grids is OOM above N=16. Use one of:

- **Windowed attention**: chunk sequence into windows of ≤ 32 tokens
- **Physics slices** (Transolver): soft-assign N grid points to S << N
  "slice tokens", attend in S×S space, broadcast back to grid
- **Random Fourier feature attention**: O(N) approximate kernel

### Model file template (1D)

```python
"""
<ModelName> — SciML Neural Operator
Implements: <arXiv ID>
Key innovation: <one sentence>
"""
import mlx.core as mx
import mlx.nn as nn
from .fno import SpectralConv1d


class <ModelName>(nn.Module):
    """
    Args:
        n_modes   : Fourier modes retained
        hidden_dim: channel width
        n_layers  : operator block depth
    """
    def __init__(self, n_modes: int = 16, hidden_dim: int = 64,
                 n_layers: int = 4, **kwargs):
        super().__init__()
        self.fc0    = nn.Linear(2, hidden_dim)
        self.blocks = [_Block(n_modes, hidden_dim) for _ in range(n_layers)]
        self.fc1    = nn.Linear(hidden_dim, 128)
        self.fc2    = nn.Linear(128, 1)

    def __call__(self, x):
        # x: [B, N, 1]
        grid = mx.linspace(0, 1, x.shape[1])[None, :, None].broadcast_to(
            (x.shape[0], x.shape[1], 1))
        x = mx.concatenate([x, grid], axis=-1)   # [B, N, 2]
        x = self.fc0(x)
        for block in self.blocks:
            x = block(x)
        x = nn.gelu(self.fc1(x))
        return self.fc2(x)                        # [B, N, 1]
```

### 2D model variant — memory ceiling must be enforced at `__init__`

```python
def __init__(self, n_modes: int = 8, hidden_dim: int = 16,
             n_layers: int = 4, **kwargs):
    assert hidden_dim <= 32, f"2D OOM: hidden_dim={hidden_dim} must be ≤ 32"
    assert n_layers  <= 4,  f"2D OOM: n_layers={n_layers} must be ≤ 4"
    super().__init__()
    ...
```

### Smoke test contract

The gate calls:
- 1D: `model(mx.random.normal([2, 64, 1]))` → must return `[2, 64, 1]`
- 2D: `model(mx.random.normal([2, 64, 64, 1]))` → must return `[2, 64, 64, 1]`

If your model takes extra constructor args with no default, it will fail the
gate. Provide safe defaults in `__init__`.

### Naming conventions

| Asset | Convention | Example |
|-------|-----------|---------|
| Python class | PascalCase | `MambaNO` |
| File | `models/<lowercase>.py` | `models/mambano.py` |
| Registry key | Same as class | `MambaNO` |
| Experiment name prefix | `<modelkey_lower>_` | `mambano_burgers_h64` |

### Reusable building blocks

```python
from models.fno  import SpectralConv1d, SpectralConv2d   # spectral conv
from models.wno  import HaarWavelet1d                     # Haar decomposition
from models.s4d  import S4DLayer                          # state-space recurrence
from models.fno  import FNO                               # full FNO as sub-module
```

---

## 5. Extending Existing Models

### Model family map

| Family | File | Exportable classes |
|--------|------|-------------------|
| FNO, RFNO, UNO, FNO2d | `models/fno.py` | `FNO`, `RFNO`, `UNO1d`, `FNO2d` |
| TFNO, CPFNO | `models/tfno.py` | `TFNO`, `CPFNO`, `TFNO2d` |
| DeepONet | `models/deeponet.py` | `DeepONet`, `PODDeepONet` |
| Time-marching DeepONet | `models/time_deeponet.py` | `TimeDeepONet`, `DualDeepONet` |
| WNO | `models/wno.py` | `WNO` |
| State-space | `models/s4d.py` | `S4DLayer` (building block) |
| SSNO | `models/ssno.py` | `SSNO` |
| Transolver | `models/transolver.py` | `Transolver`, `Transolver2D` |
| GNOT | `models/gnot.py` | `GNOT`, `GNOT2d` |
| HNN / EnergyFNO | `models/hnn.py` | `HNN`, `EnergyFNO` |
| Neural ODE / UDE | `models/neural_ode.py` | `NeuralODE`, `UDE`, `LatentODE` |
| PINN | `models/pinn.py` | `PINN` (PINO broken) |

### Adding a variant to an existing family

1. Add the new class inside the existing `models/<family>.py`
2. Export from `models/__init__.py`
3. Register in `core/research_plugins.py`:
   ```python
   MODEL_REGISTRY.register_class("MyVariant", MyVariant)
   ```
4. Validate with the scaffold gate:
   ```bash
   uv run -m core.scaffold --validate MyVariant models/<family>.py
   ```
5. Write a trajectory log entry (Section 8)

### When trajectories suggest an architectural tweak

Read the last 10 trajectory entries for the benchmark. If `"hypothesis"`
entries cluster around the same structural idea (e.g. "need multi-scale
encoding") but the `"outcome"` values are still > 2× SOTA, escalate from
HPO to a genuine architectural change: add the feature described in the
hypothesis to the best-performing model variant and re-register.

---

## 6. Loss Functions — Full Decision System

### Why `val_l2_rel` alone is insufficient

`val_l2_rel` measures mean relative energy error across the field. It does
not distinguish:

- **Spectral failures**: error concentrated in high-k modes (shock fronts,
  soliton tails) but invisible in L2 because high-k energy is small
- **Gradient failures**: pointwise values correct but derivatives wrong
  (bad for downstream solvers, conservation law checks)
- **Outlier failures**: a few catastrophic predictions averaged away by
  mean reduction
- **Conservation failures**: energy/mass not conserved even if L2 is low

Always read `diag_*` lines from the log alongside `val_l2_rel` before
concluding a run is "good".

### Full loss inventory

All losses live in `core/losses.py` and are callable via `--loss <name>`.

| Name | Flag | Formula | When to use |
|------|------|---------|-------------|
| Relative L2 | `l2_rel` | `‖pred−y‖₂ / ‖y‖₂` | Default. Start every new model here. |
| Sobolev H1 | `h1` | L2 + 0.1·`‖∂(pred−y)/∂x‖₂/‖∂y/∂x‖₂` | `diag_high_freq_error > 0.1`; shock fronts; soliton tails |
| H1 strong | `h1_strong` | H1 with α=1.0 | When gradient error dominates, H1 alone is insufficient |
| Sobolev H2 | `h2` | H1 + β·second derivative | Smoothness-critical fields (Darcy pressure); β=0.01 default |
| Spectral | `spectral` | Frequency-weighted L2 in Fourier space | `diag_high_freq_error > 0.3`; KdV soliton tails |
| Relative L1 | `l1_rel` | `‖pred−y‖₁ / ‖y‖₁` | Outlier-dominated failures; val>>train with L2 |
| MSE | `mse` | `mean((pred−y)²)` | **Debug only.** Not normalised. Never use for benchmarking. |

### Loss selection decision tree

```
Run completed with l2_rel
│
├── diag_high_freq_error > 0.3 ──► try spectral loss
│     └── still high? ──► try h1_strong
│
├── diag_high_freq_error 0.1–0.3 ──► try h1
│     └── val plateaued? ──► try h2
│
├── val >> train (≥ 2×) ──► try l1_rel (outlier robustness)
│     └── also add augmentation (noise or time_reversal)
│
├── loss NaN before step 50 ──► NOT a loss function problem
│     └── halve lr; add grad_clip: 1.0
│
└── val plateaued, all diagnostics nominal ──► try combined loss below
```

### Combined / composite loss

Compose losses by weighting. In `experiments.yaml`:

```yaml
loss: h1
loss_kwargs:
  alpha: 0.2    # weight on gradient term (default 0.1)
```

For physics-informed composite (manual `core/losses.py` extension):

```python
def pde_augmented_loss(pred, target, pde_residual_fn,
                       lambda_pde: float = 0.1, **kwargs):
    """
    L = L_data(pred, target) + λ · L_physics(pred)

    L_data should be h1 or spectral (not raw l2_rel) for SciML experiments.
    L_physics = mean(pde_residual(pred)²) evaluated at predicted solution.
    """
    l_data    = h1_loss(pred, target, alpha=0.1)
    residual  = pde_residual_fn(pred)
    l_physics = mx.mean(residual ** 2)
    return l_data + lambda_pde * l_physics

_LOSS_REGISTRY["pde_augmented"] = pde_augmented_loss
```

### Multi-metric monitoring protocol

After every run, record **all** of these — not just `val_l2_rel`:

```bash
grep "^diag_" logs/<name>.log
# diag_high_freq_error  — top-1/3 Fourier modes error (target: < 0.05)
# diag_low_freq_error   — bottom-1/3 modes error (if high: data normalisation issue)
# diag_mid_freq_error   — middle band (if high: spectral mode count too low)
# diag_max_step_time    — max per-step wall time (if > 500ms 1D: model too large)
# diag_grad_norm_max    — max gradient norm seen (if > 10: instability risk)
```

If `diag_high_freq_error` is > 3× the val_l2_rel, the L2 metric is
**misleading**: the model is fitting low-frequency structure well but
failing entirely on high-frequency features. Switch to `spectral` or `h1`
loss and report both metrics going forward.

### Extending the loss registry

```python
# core/losses.py

def my_new_loss(pred: mx.array, y: mx.array, **kwargs) -> mx.array:
    """
    Describe what physical quantity this penalises.
    pred, y: [B, N] (1D) or [B, N, N] (2D)
    Returns: scalar
    """
    ...
    return scalar_value

_LOSS_REGISTRY["my_new_loss"] = my_new_loss
```

No other file needs to change. `train.py` reads the registry via
`get_loss_fn(name)`.

---

## 7. Parameter Optimization Playbook

### Guiding principle: read before sweeping

Before touching any hyperparameter, run:

```bash
uv run analyze.py --papers                   # see gap size
uv run -m core.hypothesis --benchmark <bm>  # dominant failure mode
grep "^diag_" logs/<best_run>.log            # spectral breakdown
uv run -m core.hpo --benchmark <bm> --top 5 # Bayesian suggestions
```

Do not sweep what the hypothesis engine already answers.

### Standard sweep order

Sweep one axis at a time. Later axes compound earlier wins.

```
n_modes  →  hidden_dim  →  n_layers  →  lr  →  batch_size
  →  loss_fn  →  lr_schedule  →  augmentation
```

Never run multiple axes simultaneously in the same experiment.

### Empirically validated priors (this hardware, as of sessions 1–8)

| Param | 1D optimum | 2D hard ceiling | Source |
|-------|-----------|----------------|--------|
| `n_modes` | 24 | ≤ 12 | m=24 beats m=16 by ~11% on Burgers |
| `hidden_dim` | 128 | ≤ 32 | h=128 beats h=64 in 5-min budget |
| `n_layers` | 8 | ≤ 4 | l=8 wins; l=10+ step-time-limited |
| `lr` | 1e-3 | 1e-3 | 3e-4 too slow; 3e-3 diverges |
| `batch_size` | 32 | 16 | batch=32 beats batch=16 |
| `budget_s` | 300 min | 480 min (ns_2d: 600) | Below these floors = insufficient epochs |
| `grad_clip` | 1.0 | 1.0 | Disable only if training is provably stable |

### Diagnostic → intervention table

| Observation | Root cause | Lever |
|-------------|-----------|-------|
| `diag_high_freq_error` > 0.1 | Too few modes | Increase `n_modes`; switch to `spectral` or `h1` loss |
| `diag_low_freq_error` > 0.1 | Data normalisation | Inspect input standardisation in loader |
| `diag_mid_freq_error` > 0.1 | Spectral resolution | Increase `n_modes` and `hidden_dim` together |
| Loss NaN before step 50 | LR too high | Halve `lr`; add `grad_clip: 1.0` |
| Loss plateau after step 200 | LR schedule stale | Use `cosine` warmdown; try 1-cycle |
| val >> train loss (≥2×) | Overfitting | Reduce `hidden_dim`; add augmentation |
| val plateaued, train still falling | Capacity hit | Increase `hidden_dim` or `n_layers` |
| Step time > 500 ms (1D) | Model too large | Reduce `hidden_dim` or `n_layers` |
| Step time > 1000 ms (2D) | OOM risk | Reduce `hidden_dim` ≤ 32 immediately |
| `diag_grad_norm_max` > 10 | Instability risk | Add `grad_clip: 0.5` |

### Learning rate schedule

```yaml
# experiments.yaml fields
lr_schedule: cosine        # flat | cosine | warmup_cosine | one_cycle
warmup_steps: 100          # linear ramp from lr/10 → lr (warmup_cosine only)
final_lr_frac: 0.01        # cosine decays to lr × this value
```

Prefer `warmup_cosine` for new architectures (unstable early training);
use `cosine` for established architectures doing HPO.

### Augmentation

```yaml
augmentation: time_reversal   # Burgers: u(x,t) → u(x,T-t) symmetry
augmentation: spatial_flip    # symmetric boundary conditions
augmentation: noise           # ε~N(0,0.01) added to inputs — regularisation
```

Augmentation helps most when val >> train (overfitting signal).

### When HPO stalls: escalation path

```
1. Run core.hpo → if plateau after 15+ experiments
2. Switch loss function (see Section 6 decision tree)
3. Add augmentation
4. Try a different model family (see hypothesis engine suggestions)
5. Scaffold a new architecture (see Section 4)
6. Add to trajectories.jsonl with reasoning at each step
```

---

## 8. RL Trajectory Protocol — and How It Feeds Back Here

### Purpose

`logs/trajectories.jsonl` is the replay buffer for future agents and for
evolving this document. Patterns extracted from it directly drive Section 7
and Section 12 (Learned Lessons Log). An insight that lives only in the log
is half-buried. An insight distilled into this document is permanent.

### Mandatory write events

| Event | Log entry required |
|-------|-------------------|
| Append experiment(s) to `experiments.yaml` | Yes |
| Scaffold a new model | Yes |
| Change experiment priority | Yes |
| Kill or delete a queued experiment | Yes — include reason |
| Switch loss function after diagnosis | Yes |
| Update `our_best` in a paper YAML | No |

### Required JSON schema

```json
{
  "timestamp": "2026-04-15T12:00:00Z",
  "session": 9,
  "benchmark": "burgers_1d",
  "state": "burgers_1d | best val_l2_rel=0.1553 | gap=10.4×",
  "hypothesis": "Spectral conv sees diag_high_freq=0.18 — H1 loss should reduce gradient error",
  "action": "queued fno_burgers_h1_loss_h128_l8 with loss=h1 alpha=0.2",
  "expected_outcome": "0.10–0.13",
  "outcome": null,
  "diag_snapshot": {
    "diag_high_freq_error": 0.18,
    "diag_low_freq_error": 0.02,
    "diag_grad_norm_max": 4.1
  }
}
```

Update `"outcome"` once the run completes — either in-place or as a
follow-up entry with the same `"action"` key.

### How trajectories feed back into this SKILL

After every 5 completed runs on a benchmark, perform this distillation:

```bash
# 1. Extract outcomes from trajectory log
python3 -c "
import json
with open('logs/trajectories.jsonl') as f:
    entries = [json.loads(l) for l in f if l.strip()]
solved = [e for e in entries if e.get('outcome') is not None]
for e in sorted(solved, key=lambda x: x.get('outcome',1)):
    print(e['outcome'], '|', e['hypothesis'][:80])
" | head -20
```

**If a pattern appears ≥ 2 times across benchmarks** (e.g. "H1 loss reduces
val by ~15% when diag_high_freq > 0.15"):

1. Add a row to the diagnostic → intervention table in Section 7
2. Add a `<!-- LEARNED: <date> -->` comment inline
3. Add a summary entry to Section 12 (Learned Lessons Log)

**If a new failure mode appears that is not in Section 7's table:**

1. Add it to the table with a `TBD` lever if the fix is not yet known
2. Log it in Section 12 as an open question
3. Use the hypothesis engine to design a targeted experiment

### Trajectory-informed scaffold decisions

Before scaffolding any new architecture, extract the trajectory
`"hypothesis"` strings for the target benchmark and look for convergent
reasoning:

```bash
python3 -c "
import json
with open('logs/trajectories.jsonl') as f:
    for line in f:
        e = json.loads(line)
        if 'burgers' in e.get('benchmark',''):
            print(e.get('hypothesis',''))
"
```

If ≥ 3 entries mention the same unresolved structural gap (e.g. "model
cannot capture shock location"), that is the signal to scaffold a new
architecture rather than continue HPO.

---

## 9. Invariants & Hard Constraints

### Data integrity

| Rule | Reason | Consequence if broken |
|------|--------|----------------------|
| Never modify `data/prepare.py` | Ground-truth evaluator — every metric depends on it | All results become incomparable across sessions |
| Never hand-edit `results.json` | Lineage DAG and dedup logic live here | Silent experiment skips; corrupted parent_id chain |
| `name` in `experiments.yaml` must be globally unique | `autorun.py` dedup key | New run silently skipped |
| Validation set: `seed=42`, fixed 256 samples | Reproducible metric | Different seed = incomparable val_l2_rel |

### Model safety

| Rule | Reason |
|------|--------|
| Never queue PINO | Endpoint-only formulation always diverges |
| RFNO is 1D-only | Shape mismatch crash on all 2D benchmarks |
| SSNO: `hidden_dim ≤ 64`, `n_layers ≤ 4` | val explodes to 80+ at h≥128 |
| AFNO: do not queue | Wrong spectral bias — consistently 0.50–0.72 on Burgers |
| 2D: `hidden_dim ≤ 32`, `n_layers ≤ 4`, `n_modes ≤ 12` | OOM crash before training starts |
| `budget_s ≥ 480` for 2D, `≥ 300` for 1D | Minimum epochs to evaluate convergence |
| Never run `mse` loss for benchmarking | Unnormalised — produces incomparable numbers |

### Repository hygiene

| Rule | Reason |
|------|--------|
| No new packages beyond `pyproject.toml` | Reproducibility; no internet access in training subprocess |
| Never `git add -A` | Commits data blobs, `.venv`, secrets |
| Stage specific files only | Same as above |
| Log to `logs/trajectories.jsonl` on every queue change | RL replay buffer integrity |
| Stage `SKILL.md` after every update | This document is part of the research record |

---

## 10. How to Update This Document

### This document is self-evolving. It must be updated when:

| Trigger | Section to update |
|---------|------------------|
| A new diagnostic signal → intervention is discovered | §7 diagnostic table |
| A loss function produces a reliable δ improvement | §6 decision tree |
| A new failure mode appears that is not in §7 | §7 table + §12 |
| An architecture constraint is tightened or relaxed | §4 allowed ops, §9 |
| A new model family is scaffolded and produces results | §5 family map |
| A trajectory pattern repeats ≥ 2 times | §12 Learned Lessons |
| A hard constraint is violated and causes a bug | §9 with the exact consequence |

### Update protocol

1. **Identify the lesson**: read the trajectory entry or log that prompted it
2. **Find the section**: use the table above to locate where it belongs
3. **Write concisely**: one row in a table or one bullet is enough — the
   trajectory log has the full context
4. **Mark with a date comment** inline: `<!-- LEARNED: 2026-04-15 -->`
5. **Add to §12** with a one-line summary referencing the session number
6. **Commit `SKILL.md`** alongside the experiment commit:
   ```bash
   git add SKILL.md
   git commit -m "skill: distill lesson from session <N> — <one-line summary>"
   ```

### What NOT to put here

- Raw experiment data (goes in `results.json`)
- Full log output (goes in `logs/`)
- Pending hypothesis that has not yet been tested (goes in `experiments.yaml`
  rationale and trajectories)
- Architecture code (goes in `models/`)

---

## 11. Quality Checklist

### Before any session

- [ ] `uv run analyze.py --papers` reviewed
- [ ] `uv run auto_suggest.py` reviewed
- [ ] `logs/trajectories.jsonl` scanned for recent patterns
- [ ] `SKILL.md §12` checked for open questions relevant to today's benchmarks

### Adding a paper

- [ ] YAML in `docs/papers/` with all required fields
- [ ] `status: pending` set correctly
- [ ] `suggested_experiments` names are globally unique (not in `results.json`)
- [ ] `uv run -m core.paper_registry --pending` lists it
- [ ] `uv run -m core.paper_registry --gaps` shows new SOTA targets

### Adding a benchmark

- [ ] Loader registered in `core/research_plugins.py`
- [ ] Block added to `docs/SOTA.md` with 2D constraint note if applicable
- [ ] `data/prepare.py` untouched
- [ ] At least one baseline experiment queued with `priority: 1`

### Adding a model

- [ ] Trajectory log scanned — structural gap justifies new architecture
- [ ] Scaffold gate passes: syntax → import → shape smoke test
- [ ] Class exported from `models/__init__.py`
- [ ] Registered in `core/research_plugins.py`
- [ ] 2D models assert `hidden_dim ≤ 32` and `n_layers ≤ 4` in `__init__`
- [ ] Paper YAML `status` updated to `implemented`
- [ ] Baseline experiment queued with multi-loss run plan (l2_rel first, then h1)
- [ ] Trajectory entry written
- [ ] `SKILL.md §5` family map updated

### Queuing experiments

- [ ] No name collision: `grep "<name>" results.json experiments.yaml`
- [ ] Rationale written referencing the diagnostic or hypothesis that prompted it
- [ ] Loss function chosen from §6 decision tree (not always `l2_rel`)
- [ ] Trajectory entry written before running

### After results land

- [ ] `diag_*` lines read alongside `val_l2_rel`
- [ ] Paper YAML `our_best` and `gap_ratio` updated
- [ ] `docs/SOTA.md` "This repo" row updated
- [ ] Trajectory `outcome` and `diag_snapshot` filled in
- [ ] If val_l2_rel is new best: update memory file
- [ ] If a new pattern emerged: update `SKILL.md §7` and `§12`
- [ ] `SKILL.md` committed with run results

---

## 12. Learned Lessons Log

> Append a new row every time a repeating pattern is distilled from
> `logs/trajectories.jsonl`. Date format: YYYY-MM-DD. Session number
> from the trajectory entries.

| Date | Session | Benchmark | Lesson | Lever applied | δ improvement |
|------|---------|-----------|--------|---------------|---------------|
| 2026-04-15 | 1–8 | burgers_1d | `n_modes=24` consistently beats `n_modes=16` by ~11% | Raised default `n_modes` prior to 24 in §7 | ~11% |
| 2026-04-15 | 1–8 | burgers_1d | `hidden_dim=128` beats `hidden_dim=64` within 5-min budget | Raised default `hidden_dim` prior to 128 in §7 | ~15% |
| 2026-04-15 | 1–8 | burgers_1d | `n_layers=8` is step-time-limited at 10+ on M-series | Capped `n_layers` recommendation at 8 in §7 | — |
| 2026-04-15 | 1–8 | all 2D | `hidden_dim ≥ 64` causes OOM crash before first epoch | Enforced `hidden_dim ≤ 32` assertion in model `__init__` | — |
| 2026-04-15 | 1–8 | kdv_1d | RFNO with Pre-LN residuals unlocks stable depth ≥ 10 | RFNO now preferred for KdV; achieved 5× below SOTA | SOTA beat |
| 2026-04-15 | 1–8 | wave_1d | RFNO generalises well to wave physics; 5× below SOTA | RFNO added to wave_1d suggested experiments | SOTA beat |
| 2026-04-15 | 1–8 | darcy_2d | Darcy solver uses averaged permeability; FNO baseline fails | Focus Darcy work on heterogeneous solver fix before HPO | — |
| 2026-04-15 | 1–8 | ns_2d | NS solver diverges at step 67 for standard ICs | ns_2d deprioritised until solver is stabilised | — |
| 2026-04-15 | 1–8 | burgers_1d | `val_l2_rel` misleads when `diag_high_freq_error` >> L2 | Multi-metric monitoring protocol added to §6 | — |
| 2026-04-16 | 9 (code) | all | **RFNO2d defined twice in `models/fno.py`**: second (inferior, no Pre-LN) definition silently overrode first. All RFNO2d experiments trained without LayerNorm for multiple sessions. | Deleted duplicate definition; Pre-LN RFNO2d now active | TBD |
| 2026-04-16 | 9 (code) | all | **GNOT_FFNO gate was non-trainable** (`mx.zeros` bare array not tracked by `nn.Module`): gate frozen at 0.5 blend for all training. Spectral-attention weighting never adapted. | Replaced with `nn.Linear(dims, 1)` trainable gate | TBD |
| 2026-04-16 | 9 (code) | all 2D | **H1/spectral losses silently fell back to L2 on 2D inputs**: any run on 2D benchmarks with `loss=h1` or `loss=spectral` was actually training with plain L2. Warning fired once and was ignored. | Implemented proper 2D paths via `rfft2`; no fallback | Restored |
| 2026-04-16 | 9 (code) | all | `diag_grad_norm_max` was never populated from logs: trainer logged `gnorm:` per step but `parse_log_file()` did not extract it. Instability signals in §7 were never triggering. | Added regex extractor in `core/diagnostics.py` | Monitoring enabled |
| 2026-04-16 | 9 (code) | all | HPO auto-trigger loop stalled when `agent_loop.py` returned no new experiments: `--auto` flag would stop rather than self-generate new configs. | Added `BayesianHPO.suggest_top(3)` fallback in `autorun.py` | Continuous loop |
| 2026-04-16 | 9 (code) | all | LR schedule had no selection — `warmup_cosine` was hardcoded. `cosine` and `onecycle` showed ~10-15% faster convergence in early experiments. | Added `--lr_schedule` flag with 4 choices; `ExperimentConfig.lr_schedule` field | ~10-15% on select runs |

### Open questions (no confirmed mitigation yet)

- **Darcy 2D gap (25×)**: Best is 0.1041 vs GNOT paper 0.0041. FEDONet2D is our
  best architecture but 2D memory constraints limit capacity.
- **NS 2D near-SOTA (1.12×)**: 0.01428 vs FNO 0.0128 — can AFNO-style architecture
  close this? (AFNO itself is not queueable due to spectral bias issue.)
- **Burgers gap (47× vs GNOT)**: GNOT architecture not yet producing strong results
  on 1D benchmarks. Is this a hyperparameter issue or architecture adaptation needed?
- **SSNO divergence at h=128**: Why does val explode to 80+? Gradient norm?
  LayerNorm position? No confirmed fix yet.
- **MHD/NS-HRE all-fail**: All models return val=1.0. Physics too complex for
  current capacity constraints or data distribution mismatch?

---

## 13. Agent-First Reasoning Protocol (v2.0)

The system is refactored to prioritize *agent intelligence* over external API calls. Fulfill requests when you see the `[AGENT_REQUEST]` pattern.

### Fulfilling Scientific Diagnoses (/debug)

When a model fails with NaNs, the `ScientificDebugger` provides a probe log summary.
1. **Analyze Gradient flow**: Look for the layer where grad_norm exceeds 1e5.
2. **Identify Root Cause**: Usually astronomical weight initialization, missing LayerNorm, or an unstable spectral convolution.
3. **Propose a Fix**: 
   - *Example*: "Weights in `bomb.weight` initialized to 1e20. Fix: Use `nn.init.xavier_uniform`."
   - *Example*: "Gradient exploded in Layer 3. Fix: Add `nn.LayerNorm` before spectral blocks."

### Fulfilling Adversarial Reviews (/reason)

When generating new configs or registering models:
1. **Skepticism First**: Assume the new architecture will fail due to spectral bias or OOM.
2. **Check High-Freq Error**: If past runs on this benchmark have `diag_high_freq > 0.3`, ensure the new config uses `loss: h1` or `loss: spectral`.
3. **Check Memory**: If 2D, strictly enforce `h <= 32`, `l <= 4`.
4. **Verdict**: Provide a concise verdict: **ACCEPT**, **REJECT**, or **REFINE** (with specific param changes).

---

## 14. Multi-Metric Monitoring Protocol (v2.0)

| Metric | Target | Action if High |
|--------|--------|----------------|
| `val_l2_rel` | < 0.05 | Success — push to SOTA |
| `diag_high_freq_error` | < 0.10 | Switch to `h1` or `spectral` loss |
| `diag_grad_norm_max` | < 5.0 | Reduce `lr` or add `grad_clip: 0.5` |
| `diag_early_stopped` | True | Model converged; try different architecture |
| `scientific_diagnosis` | - | Apply fix branch immediately |
