# AutoResearch-MLX

**Autonomous neural operator research loop for PDE solving on Apple Silicon.**  
Queue an experiment, go to sleep — the system trains, evaluates, diagnoses failures,
proposes follow-ups, and updates itself overnight.

[![MLX](https://img.shields.io/badge/Platform-MLX%20%28Apple%20Silicon%29-blue.svg)](https://github.com/ml-explore/mlx)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Active%20Research-green.svg)](#project-status)

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Architecture](#architecture)
5. [Model Zoo](#model-zoo)
6. [PDE Benchmarks](#pde-benchmarks)
7. [Configuration](#configuration)
8. [Training a Single Model](#training-a-single-model)
9. [Running the Autonomous Loop](#running-the-autonomous-loop)
10. [Loss Functions](#loss-functions)
11. [Bayesian HPO](#bayesian-hpo)
12. [Adding a New Model](#adding-a-new-model)
13. [Dashboard](#dashboard)
14. [Results & Tracking](#results--tracking)
15. [Deployment & Production](#deployment--production)
16. [Troubleshooting](#troubleshooting)
17. [Project Status](#project-status)
18. [Additional Resources](#additional-resources)

---

## Overview

AutoResearch-MLX is a self-driving experiment harness for **neural operator** research —
the class of deep learning models that learn mappings between function spaces (e.g.,
PDE initial condition → solution). It targets Apple Silicon (M1/M2/M3/M4) via the
[MLX](https://github.com/ml-explore/mlx) framework.

**What it does:**

- Provides 28+ neural operator architectures (FNO, MambaNO, Transolver, DeepONet,
  HANO, WNO, KAN, PINN, …) all under one unified training harness
- Supports 15+ PDE benchmarks (Burgers, Darcy, Navier-Stokes, KdV, Shallow Water, …)
- Orchestrates overnight autonomous experiment campaigns: train → evaluate → diagnose →
  propose next experiment → repeat, without human intervention
- Uses Bayesian HPO (Gaussian Process + Expected Improvement) and a data-driven
  HypothesisEngine to propose follow-up experiments backed by empirical results and
  literature
- Tracks full experiment lineage in `results.json`, champions in `model_registry.json`,
  and metrics in MLflow

**Who this is for:** ML researchers and engineers working on scientific ML / neural PDEs
who want to explore many architectures and hyperparameters systematically on a Mac.

---

## Quick Start

### Prerequisites

- Apple Silicon Mac (M1 / M2 / M3 / M4)
- Python 3.10–3.13
- [`uv`](https://github.com/astral-sh/uv) package manager
- ~5 GB disk space (PyTorch is a hard dependency for data utilities; MLX handles all training)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url> autoresearch-mlx
cd autoresearch-mlx
uv sync
```

### Your First Training Run (~2 minutes)

```bash
uv run train.py \
  --benchmark burgers_1d \
  --model_type FNO \
  --hidden_dim 64 \
  --n_layers 4 \
  --n_modes 16 \
  --lr 1e-3 \
  --budget 120
```

**Expected output:**

```
[train] burgers_1d | FNO | budget=120s
epoch 10 | train_loss=0.4231 | val_l2_rel=0.3105 | lr=9.9e-4
epoch 20 | train_loss=0.2814 | val_l2_rel=0.2208 | lr=9.7e-4
...
[eval] val_l2_rel = 0.2087  SOTA_target = 0.0149  gap = 13.97x
[done] checkpoint saved → checkpoints/burgers_1d_FNO_....npz
```

### Overnight Autonomous Campaign

```bash
# Queue experiments in experiments.yaml, then let the system self-drive
uv run autorun.py --auto --commit --max-auto-experiments 50 --max-auto-time 86400
```

---

## Core Concepts

### Neural Operator

A neural network trained to approximate a mapping between infinite-dimensional function
spaces — e.g., mapping a PDE's initial condition (a function) to its solution at time T
(another function). Unlike standard networks, neural operators are discretization-invariant:
they generalize to different grid resolutions at test time.

### `experiments.yaml` — The Queue

A declarative YAML list of experiments. Each entry becomes one training run.
The system deduplicates against `results.json`, so re-queueing a completed experiment
is harmless.

```yaml
- name: fno_burgers_baseline
  benchmark: burgers_1d
  model: FNO
  hidden_dim: 128
  n_layers: 4
  n_modes: 24
  lr: 1e-3
  budget_s: 1800
  rationale: "FNO baseline per Li et al. 2021"
```

### `results.json` — Single Source of Truth

A lineage-aware DAG of all completed experiments. Every entry records:
`{name, benchmark, model, val_l2_rel, hyperparams, parent, status, diagnostics, timestamp}`.
`analyze.py` reads this to produce SOTA gap reports and improvement trajectories.

### `model_registry.json` — Champion Registry

Tracks the best checkpoint path and hyperparameters per benchmark. A new champion
is registered automatically when a run beats the previous best.

### SOTA Targets

Hard-coded reference values from published papers (see `core/utils.py`). Used to
compute "SOTA gap" — how far the current best is from the published state of the art.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     experiments.yaml                        │
│            (declarative experiment queue)                   │
└────────────────────────┬────────────────────────────────────┘
                         │ reads / appends
          ┌──────────────▼──────────────┐
          │        autorun.py           │◄──── agent_loop.py
          │  (subprocess orchestrator)  │      (Mode B AI loop)
          └──────────────┬──────────────┘
                         │ subprocess
          ┌──────────────▼──────────────┐
          │          train.py           │
          │   (single training run)     │
          │                             │
          │  ModelRegistry.build()      │
          │  BenchmarkRegistry.eval()   │
          │  Trainer.train()            │
          └──┬───────────┬──────────────┘
             │           │
    ┌────────▼──┐   ┌────▼──────────────┐
    │results.   │   │checkpoints/       │
    │json (DAG) │   │<name>.npz         │
    └────────┬──┘   └───────────────────┘
             │
    ┌────────▼──────────────┐
    │   auto_suggest.py     │  ← Bayesian HPO + HypothesisEngine
    │   (next experiment)   │    (reads results + papers/*.yaml)
    └───────────────────────┘
```

### Key Modules

| Module | Responsibility |
|---|---|
| `train.py` | Single training run: build model, train, evaluate, log |
| `autorun.py` | Orchestrates queue, retries, adaptive fallback, git commits |
| `agent_loop.py` | Fully autonomous Mode B: generates new experiments programmatically |
| `auto_suggest.py` | Ranks and prints next-experiment suggestions |
| `analyze.py` | Summarizes `results.json` into SOTA gap tables |
| `core/research_plugins.py` | Central `ModelRegistry` + `BenchmarkRegistry` |
| `core/trainer.py` | MLX training loop, EMA, snapshot ensemble, early stopping |
| `core/loader.py` | `ExperimentConfig` dataclass + YAML parser |
| `core/losses.py` | L2-rel, H1, spectral, and adaptive loss functions |
| `core/hpo.py` | Gaussian Process HPO with EI acquisition |
| `core/hypothesis.py` | Data-driven intervention engine |
| `core/scaffold.py` | Gated model scaffolding (stub → validate → register) |
| `core/tracker.py` | Lineage-aware `results.json` writer |
| `core/diagnostics.py` | Log parsing, spectral bias detection, early-stop analysis |
| `data/prepare.py` | Ground-truth data generator and evaluation harness |
| `dashboard/app.py` | FastAPI Command Center API |

---

## Model Zoo

28+ neural operator architectures, all implemented in MLX:

### Fourier-Based

| Registry Key | Architecture | Key Reference |
|---|---|---|
| `FNO` | 1D Fourier Neural Operator | Li et al. 2021 |
| `FNO2D` | 2D FNO (auto-selected for 2D benchmarks) | Li et al. 2021 |
| `RFNO` / `RFNO2D` | Residual FNO with Pre-LayerNorm | — |
| `AFNO` | Adaptive FNO (softshrink sparsity) | Guibas et al. 2022 |
| `FFNO` | Factorized FNO | — |
| `TFNO` / `RTFNO` | Tucker-factorized FNO (4–10x fewer params) | — |
| `UNO` / `UNO2d` | U-shaped Neural Operator | — |
| `SNO2D` | Spectral Neural Operator 2D | — |
| `AttentionEnhancedFNO2D` | FNO blocks + cross-attention | — |

### State-Space / SSM-Based

| Registry Key | Architecture | Key Reference |
|---|---|---|
| `MambaNO` / `MambaNO1d` | Mamba SSM-based operator | Gu & Dao 2023 |
| `SSNO` | Dual-branch S4D + Spectral FNO with gating | — |
| `S4NO` | S4D diagonal state-space operator | — |
| `MemNO` | Memory-augmented Neural Operator | 2025 |

### Attention-Based

| Registry Key | Architecture | Key Reference |
|---|---|---|
| `GNOT` | Graph Neural Operator Transformer | — |
| `GNOT_Axial2d` | GNOT with axial attention | — |
| `Transolver` / `Transolver2D` | Physics-slice attention | Wu et al. 2024 |
| `HANO2D` | Hierarchical Attention Neural Operator (2D only) | — |

### DeepONet Family

| Registry Key | Architecture | Key Reference |
|---|---|---|
| `DeepONet` | Branch-Trunk DeepONet | Lu et al. 2021 |
| `PODDeepONet` | SVD basis DeepONet | — |
| `TimeDeepONet` | Time-conditioned DeepONet | — |
| `HybridDecoderDeepONet2D` | Hybrid DeepONet with 2D decoder | — |
| `FEDONet2D` | Frequency-Enhanced DeepONet 2D | — |

### Physics-Informed / Other

| Registry Key | Architecture |
|---|---|
| `WNO` | Wavelet Neural Operator (Haar multi-resolution) |
| `PINN` | Physics-Informed Neural Network (coordinate MLP) |
| `KAN` | Kolmogorov-Arnold Network |
| `cPIKAN` | Chebyshev polynomial KAN for physics |
| `PACMANN` | Scaffold-generated stub (placeholder architecture; replace operator blocks before use) |
| `VSMNO2D` | Variational Spectral Mixture NO — dual `SpectralConv2d` branches with learned mixture weights |

---

## PDE Benchmarks

### 1D

| Key | PDE | Grid | Notes |
|---|---|---|---|
| `burgers_1d` | Viscous Burgers: u_t + u·u_x = ν·u_xx, ν=0.01/π | 64 | Default benchmark |
| `burgers_nu_001` | Burgers with ν=0.001 | 64 | Strong shocks |
| `kdv_1d` | Korteweg-de Vries | 64 | Soliton dynamics |
| `wave_1d` | Wave: u_tt = c²·u_xx | 64 | Propagation |
| `euler_1d` | 1D Euler (multi-channel) | 64 | Subsonic smooth flow |

### 2D

| Key | PDE | Grid | Notes |
|---|---|---|---|
| `darcy_2d` | Darcy: -div(a·∇u) = f | 64×64 | Variable-coefficient elliptic |
| `ns_2d` | Navier-Stokes 2D (vorticity) | 64×64 | Semi-implicit spectral |
| `ns_hre_2d` | Navier-Stokes Re=1000 | 64×64 | High-Reynolds |
| `swe_2d` | Shallow Water Equations | 64×64 | Linearized gravity waves |
| `allen_cahn_2d` | Allen-Cahn phase field | 64×64 | Phase-field coarsening |
| `elasticity_2d` | Linear elasticity | 64×64 | Multi-physics |
| `wavebench_2d` | WaveBench 2D | 64×64 | Wave propagation |
| `pdebench_2d` | PDEBench suite | 64×64 | Comprehensive |
| `multiphysics_2d` | Coupled multi-physics | 64×64 | Multi-field |

**Fixed data constants** (from `data/prepare.py` — do not modify):

| Constant | Value |
|---|---|
| `GRID_SIZE` | 64 |
| `N_TRAIN` | 4096 |
| `N_VAL` | 256 |
| `VAL_SEED` | 42 |
| `TRAIN_SEED` | 7 |
| `SOLVER_STEPS` | 500 |

---

## Configuration

### `ExperimentConfig` Fields

The full set of fields supported in `experiments.yaml`:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | str | required | Unique experiment name |
| `benchmark` | str | `burgers_1d` | Benchmark key |
| `model` | str | `FNO` | Model registry key |
| `hidden_dim` | int | 64 | Channel width |
| `n_layers` | int | 4 | Number of operator blocks |
| `n_modes` | int | 16 | Fourier modes (1D) or max modes per dim (2D) |
| `n_levels` | int | 3 | Hierarchy levels (HANO, UNO) |
| `n_head` | int | 4 | Attention heads (GNOT, Transolver) |
| `slice_num` | int | 32 | Physics slices (Transolver) |
| `lr` | float | 1e-3 | Initial learning rate |
| `batch_size` | int | 64 | Training batch size |
| `grad_clip` | float | 1.0 | Gradient clipping norm |
| `loss_type` | str | `l2_rel` | Loss function key |
| `h1_alpha` | float | 0.1 | H1 gradient penalty weight |
| `augment` | bool | false | Random flip/roll data augmentation |
| `curriculum` | bool | false | Curriculum: start with easier samples |
| `curriculum_epochs` | int | 10 | Epochs before full difficulty |
| `budget_s` | int | 1200 | Training time budget in seconds (raised to budget floor by autorun.py) |
| `lr_schedule` | str | `warmup_cosine` | `warmup_cosine`, `cosine`, `onecycle`, `none` |
| `ema_decay` | float | 0.0 | EMA decay (0 = disabled; set to 0.999 to enable) |
| `patience` | int | 5 | Early-stop patience (in eval intervals) |
| `snapshot_ensemble` | int | 0 | Number of snapshots to average (0 = disabled) |
| `resume` | bool | false | Resume from champion checkpoint |
| `resume_from` | str | null | Checkpoint path or `champion:<benchmark>` |
| `save_ckpt` | bool | true | Save checkpoint after training |
| `seed` | int | 42 | Global random seed |
| `pino_lambda` | float | 0.0 | PINO physics-loss weight |
| `refine_grid` | bool | false | Adaptive grid refinement |
| `cheb_degree` | int | 5 | Chebyshev degree for cPIKAN |
| `curriculum_epochs` | int | 0 | Curriculum ramp epochs (requires `curriculum: true`) |
| `parent_name` | str | — | Name of parent experiment (lineage tracking) |
| `expected` | str | — | Expected val_l2_rel range (informational) |
| `rationale` | str | — | Human-readable justification |
| `paper_ref` | str | — | Paper ID from `docs/papers/*.yaml` |
| `priority` | int | 5 | Queue priority (1=highest) |

### 2D Memory Safety Limits

Enforced at `ModelRegistry.build()` for all 2D benchmarks (raises `ValueError`):

- **Hard limit**: `hidden_dim < 64` and `n_layers < 8`
- **Recommended practice**: `hidden_dim = 32`, `n_layers ≤ 4` to avoid OOM on 16 GB devices

### Budget Floors (enforced by `autorun.py`)

- 1D benchmarks: ≥ 1800 seconds
- 2D benchmarks: ≥ 3600 seconds

> **Note:** These are hard-coded constants — not configurable via CLI. To change them,
> edit `BUDGET_FLOOR_1D` / `BUDGET_FLOOR_2D` in `autorun.py` lines 82–83.

### SOTA Reference Targets

Targets used by `analyze.py` and `auto_suggest.py` for gap calculations. Source: `core/utils.py`.

```python
SOTA = {
    # 1D
    "burgers_1d":      0.0031,  # GNOT (Hao et al. 2023)
    "kdv_1d":          0.010,   # estimated
    "wave_1d":         0.005,   # estimated
    "euler_1d":        0.003,   # estimated
    "burgers_nu_001":  0.080,   # estimated
    # 2D
    "darcy_2d":        0.0041,  # GNOT (Hao et al. 2023)
    "ns_2d":           0.0128,  # Li et al. 2021
    "ns_hre_2d":       0.050,   # estimated
    "swe_2d":          0.015,   # estimated
    "allen_cahn_2d":   0.080,   # estimated
    "elasticity_2d":   0.010,   # estimated
    "wavebench_2d":    0.015,   # estimated
    "pdebench_2d":     0.005,   # estimated
    "mhd_2d":          0.050,   # estimated
    "multiphysics_2d": 0.200,   # estimated
}
```

---

## Training a Single Model

```bash
# Minimal run (120-second budget)
uv run train.py --benchmark burgers_1d --model_type FNO --budget 120

# Full options example
uv run train.py \
  --benchmark darcy_2d \
  --model_type Transolver2D \
  --hidden_dim 32 \
  --n_layers 4 \
  --n_head 4 \
  --slice_num 32 \
  --lr 2e-3 \
  --lr_schedule warmup_cosine \
  --loss_type h1 \
  --h1_alpha 0.1 \
  --budget 3600 \
  --ema_decay 0.999 \
  --save_ckpt \
  --seed 42

# Resume from champion checkpoint and fine-tune
uv run train.py \
  --benchmark burgers_1d \
  --model_type FNO \
  --resume_from champion:burgers_1d \
  --lr 1e-4 \
  --budget 1800
```

### CLI Arguments Reference

```
--benchmark         PDE benchmark key (default: burgers_1d)
--model_type        Model registry key (default: FNO)
--hidden_dim        Channel width (default: 64)
--n_layers          Operator block depth (default: 4)
--n_modes           Fourier modes (default: 16)
--n_levels          Hierarchy levels for UNO/HANO (default: 4)
--n_head            Attention heads (default: 4)
--slice_num         Physics slices for Transolver (default: 32)
--lr                Initial learning rate (default: 1e-3)
--lr_schedule       LR schedule type (default: warmup_cosine)
--batch_size        Batch size (default: 32)
--loss_type         Loss function key (default: l2_rel)
--h1_alpha          H1 gradient weight (default: 0.1)
--grad_clip         Gradient clip norm (default: 1.0)
--augment           Enable data augmentation (flag)
--curriculum        Enable curriculum training (flag)
--budget            Time budget in seconds (default: 300)
--ema_decay         EMA decay (default: 0.999)
--patience          Early-stop patience (default: 5)
--snapshot_ensemble Enable snapshot ensemble (flag)
--save_ckpt         Save checkpoint (flag)
--resume            Resume from last checkpoint (flag)
--resume_from       Checkpoint path or "champion:<benchmark>"
--seed              Random seed (default: 42)
```

---

## Running the Autonomous Loop

### Mode A: Human-Guided (Queue + Autorun)

1. Append entries to `experiments.yaml`
2. Run `autorun.py` — it processes the queue sequentially, skipping already-done runs

```bash
# Process queue once
uv run autorun.py

# With auto-commit to git after each successful run
uv run autorun.py --commit
```

### Mode B: Fully Autonomous

The system reads results, generates hypotheses, appends new experiments, and runs them
without any human input.

```bash
# Autonomous overnight run: up to 50 experiments over 24 hours
uv run autorun.py --auto --commit --max-auto-experiments 50 --max-auto-time 86400

# Dry run: see what would be queued without executing
uv run auto_suggest.py --benchmark burgers_1d --top 5

# Run one AI agent cycle (inspect/modify experiments.yaml only)
uv run agent_loop.py
```

### Retry System

When a run crashes or produces poor results, `autorun.py` escalates automatically:

| Level | Trigger | Action |
|---|---|---|
| **r1** | Crash or OOM | `smart_fix()`: context-aware fix from log analysis |
| **r2** | r1 fails | Halve `hidden_dim`, `n_layers`, `n_modes`; `lr × 0.1` |
| **r3** | r2 fails | Minimal viable config: `h=32, l=2, m≤8, lr=1e-4` |
| **_adapt** | Run completes but `val > 3× baseline` | `HypothesisEngine.suggest_intervention()` |

### Pause / Resume

```bash
# Pause between experiments (checked at start of each run)
touch .autorun_pause

# Resume
rm .autorun_pause
```

### Analyze Results

```bash
# Print best-per-benchmark table + SOTA gaps
uv run analyze.py

# Get ranked suggestions for next experiment on a specific benchmark
uv run auto_suggest.py --benchmark darcy_2d
```

---

## Loss Functions

| Key | Formula | Best For |
|---|---|---|
| `l2_rel` | ‖pred−y‖₂ / ‖y‖₂ | Default; most benchmarks |
| `h1` | L2 + α·‖∇pred − ∇y‖₂ | Shock fronts, Burgers, Darcy |
| `h1_adaptive` | H1 with α auto-scaled so gradient term ≈ 30% | Unknown PDEs |
| `h1_strong` | H1 with α = 1.0 | Strong gradient penalty |
| `spectral` | Frequency-weighted L2 (emphasizes high-k) | High-frequency error |
| `l1_rel` | ‖pred−y‖₁ / ‖y‖₁ | Outlier-robust training |
| `mse` | Mean squared error | Debug only |

---

## Bayesian HPO

`core/hpo.py` implements a Gaussian Process surrogate with Expected Improvement (EI)
acquisition for automatic hyperparameter search.

```python
from core.hpo import BayesianHPO

hpo = BayesianHPO(benchmark="burgers_1d")
hpo.load_history()          # seeds GP from results.json history

suggestion = hpo.ask()
# → {"hidden_dim": 192, "n_layers": 6, "n_modes": 20, "lr": 5e-4}

hpo.tell(suggestion, val_l2_rel=0.145)   # update surrogate
```

**Search space:**

| Hyperparameter | Range |
|---|---|
| `hidden_dim` | 32 – 256 |
| `n_layers` | 2 – 12 |
| `n_modes` | 8 – 32 |
| `lr` | 1e-4 – 1e-2 |

Multi-objective extension supports Pareto-front extraction via weighted scalarization.

---

## Adding a New Model

The scaffold system enforces a three-gate pipeline to prevent broken models from
entering the queue:

```bash
# Gate 1: Generate starter file from template
uv run -m core.scaffold --stub MyOperator --base FNO

# Gate 2: Validate (import check + output shape smoke test)
uv run -m core.scaffold --validate MyOperator models/my_operator.py

# Gate 3: Register in MODEL_REGISTRY and experiments.yaml
uv run -m core.scaffold --register MyOperator
```

**Model interface contract** (what `train.py` expects):

```python
class MyOperator(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int, n_modes: int, **kwargs):
        super().__init__()
        ...

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, grid, channels)     [1D benchmarks]
        # x: (batch, h, w, channels)     [2D benchmarks]
        # returns: same spatial shape as x
        ...
```

---

## Dashboard

A FastAPI Command Center for monitoring and injecting experiments.

```bash
# Start dashboard
uv run uvicorn dashboard.app:app --host 0.0.0.0 --port 8000

open http://localhost:8000
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web dashboard UI (serves `dashboard/ui/dashboard.html`) |
| `GET` | `/api/experiments` | Full experiment lineage list |
| `GET` | `/api/experiment/{exp_id}` | Single experiment detail + inspect URL if PNG exists |
| `GET` | `/api/lineage` | DAG as `{"nodes": [...], "links": [...]}` |
| `GET` | `/api/status` | Live VRAM, progress, loss history, active runs, pause state |
| `GET` | `/api/sota` | Per-benchmark SOTA gap ratios `{"sota", "our_best", "ratio", "beats_sota"}` |
| `GET` | `/api/logs` | List of available log file stems |
| `GET` | `/api/logs/{exp_name}?tail=300` | Log tail (`total_lines`, `lines`, `path`) |
| `GET` | `/api/diagnostics/{exp_name}` | Spectral/gradient diagnostics for an experiment |
| `GET` | `/api/queue` | Pending experiments with priority and metadata |
| `GET` | `/api/active` | Names of currently-running experiments |
| `GET` | `/api/arch-map` | Map of model key → architecture diagram file path |
| `GET` | `/api/model-registry` | Full champion registry (all benchmarks) |
| `GET` | `/api/model-registry/{benchmark}` | Champion registry filtered by benchmark |
| `GET` | `/api/mlflow/runs?benchmark=&limit=200` | MLflow run list |
| `POST` | `/api/inject` | Inject a new experiment without restarting autorun |
| `POST` | `/api/priority` | Override priority for a queued experiment |
| `POST` | `/api/kill/{name}` | Request kill of a running experiment |
| `POST` | `/api/pause` | Pause autorun between experiments |
| `POST` | `/api/resume` | Resume autorun |
| `GET` | `/api/pause` | Check pause state `{"paused": bool}` |
| `GET` | `/figs/*` | Diagnostic plot images (static) |
| `GET` | `/logs/*` | Raw log files (static) |

---

## Results & Tracking

### Results Store — SQLite + JSON (race-safe)

All experiment results are written through `core/results_store.py`, which provides
two complementary protection layers against concurrent write corruption:

| Layer | Mechanism | Guarantees |
|---|---|---|
| **SQLite WAL** | `journal_mode=WAL`, `INSERT OR IGNORE` by `id` | Concurrent readers never blocked; duplicate-safe |
| **FileLock + atomic rename** | `filelock` on JSON export, `os.replace()` | JSON never half-written; crash-safe |

**Never write `results.json` directly.** Always use:

```python
from core.results_store import store
store.append(row)          # insert one result (idempotent)
store.load("burgers_1d")   # read, optionally filtered
store.best_per_benchmark() # aggregation via SQL
store.query("SELECT ...")  # arbitrary SQL on the results table
```

`results.json` is kept as a human-readable export and stays in sync automatically.
`results.db` (SQLite) is the primary store and is gitignored.

### MLflow

All runs are also logged to `mlruns/` automatically:

```bash
uv run mlflow ui --port 5000
open http://localhost:5000
```

### Champion Registry

`model_registry.json` stores the best checkpoint per benchmark:

```json
{
  "burgers_1d": {
    "path": "checkpoints/burgers_1d_FNO_h128_l8_m24.npz",
    "val_l2_rel": 0.1468,
    "model": "FNO"
  }
}
```

### Trajectory Log

`logs/trajectories.jsonl` is an RL-style replay buffer logged after every run:

```jsonl
{"timestamp": "...", "benchmark": "burgers_1d", "action": {...}, "outcome": 0.1468, "critique": "..."}
```

### Diagnostics

`core/diagnostics.py` parses logs to detect:

- **Spectral bias**: unusually high error in high-frequency Fourier modes
- **Early stopping**: whether patience threshold was triggered
- **Crash patterns**: OOM, NaN loss, shape mismatches
- **Plateau detection**: loss stagnated for N epochs

---

## Deployment & Production

### Hardware Requirements

This project runs exclusively on **Apple Silicon** (MLX is the training backend):

| Resource | Minimum | Recommended |
|---|---|---|
| Chip | M1 | M2 Pro / M3 |
| Unified Memory | 16 GB | 32 GB (for 2D) |
| Storage | 20 GB | 50 GB |

### DVC Data Versioning

```bash
dvc repro   # Reproduce data pipeline
dvc push    # Push data to remote
dvc pull    # Pull data from remote
```

### Disk Layout

```
checkpoints/      Model weights (.npz), 10–200 MB each
logs/             Per-experiment logs (.log), trajectories.jsonl
mlruns/           MLflow tracking database
figs/             Diagnostic plots (.png)
results.json      Experiment DAG (grows with each run)
model_registry.json  Champion checkpoints per benchmark
```

---

## Troubleshooting

**OOM / out-of-memory on 2D benchmarks**  
Reduce `hidden_dim` to 32 and `n_layers` to ≤ 4. The system auto-retries with
smaller configs (r2/r3 fallback) when run via `autorun.py`.

**`val_l2_rel` stuck at 1.0 or diverging**  
Try `loss_type: h1`, reduce `lr` by 10×, or use `lr_schedule: warmup_cosine`.
Check `logs/<name>.log` for NaN loss entries.

**`RFNO` vs `RFNO2D` — which to use?**  
Use `RFNO` for 1D benchmarks and `RFNO2D` for 2D benchmarks. Both are fully
supported. `RFNO2D` takes a single `n_modes` flag and auto-maps it to
`(n_modes1, n_modes2)` — no manual tuning required.

**Import error after adding a new model**  
Re-run `--validate` before `--register`. If validation passes, check
`models/__init__.py` to ensure the class is exported.

**`autorun.py` skipping new queue entries**  
The `name` field must be unique. Entries whose name matches a completed run in
`results.json` are skipped automatically.

**MLflow UI shows no runs**  
Run `uv run mlflow ui` from the project root, not a subdirectory.

---

## Project Status

**Active research project.** Stable for automated overnight runs. Not a versioned,
published library — `experiments.yaml` schema and `results.json` format may change.

### Current Best Results

Live values from `results.db` (SQLite SSoT). Updated after every committed run.
184 experiments completed across 14 benchmarks. **9 of 14 SOTA targets beaten.**

| Benchmark | Best Model | val_l2_rel | SOTA Target | Status |
|---|---|---|---|---|
| `wave_1d` | FNO | **0.000992** | 0.005 | ✅ 5× below SOTA |
| `kdv_1d` | RFNO | **0.002023** | 0.010 | ✅ 5× below SOTA |
| `euler_1d` | FNO | **0.002413** | 0.003 | ✅ beat SOTA |
| `pdebench_2d` | FEDONet2D | **0.002602** | 0.005 | ✅ beat SOTA |
| `elasticity_2d` | FEDONet2D | **0.007734** | 0.010 | ✅ beat SOTA |
| `wavebench_2d` | SNO2D | **0.009907** | 0.015 | ✅ beat SOTA |
| `swe_2d` | FNO | **0.010729** | 0.015 | ✅ beat SOTA |
| `ns_2d` | FNO | 0.014284 | 0.0128 | 1.12× — near SOTA |
| `allen_cahn_2d` | FNO | **0.062801** | 0.080 | ✅ beat SOTA |
| `burgers_nu_001` | RFNO | **0.077943** | 0.080 | ✅ beat SOTA (shock) |
| `darcy_2d` | FNO | 0.059719 | 0.0041 | 14.6× gap |
| `burgers_1d` | MambaNO | 0.181264 | 0.0031 | 58× gap |
| `multiphysics_2d` | HybridDecoderDeepONet2D | 0.692273 | 0.200 | gap |
| `radiative_2d` | — | 1.000 | — | unsolved |

> SOTA targets: GNOT (Hao et al. 2023) for Burgers/Darcy; FNO (Li et al. 2021) for NS;
> estimated baselines for others. Full references: [`docs/SOTA.md`](./docs/SOTA.md)

### Infrastructure Improvements (this session)

- **Race-safe results store** (`core/results_store.py`): SQLite WAL + FileLock + atomic
  rename. Stress-tested: 8 concurrent processes × 10 writes = 80/80 rows, 0 lost.
- **einops** in `Transolver1d` and `GNOT`: replaces error-prone `reshape+transpose` chains
- **OptunaHPO** (`core/hpo.py`): TPE sampler + MedianPruner as primary HPO backend
- **Novelty scoring** (`agent_loop.py`): cosine similarity filter rejects near-duplicate HP proposals
- **`train.py` 1D detection fix**: `burgers_nu_001`/`burgers_nu_01` now correctly routed to 1D model variants

### Known Limitations

- Apple Silicon only — not portable to Linux/CUDA without replacing MLX
- 2D benchmarks: `hidden_dim ≥ 64` causes OOM on 16 GB devices (enforced at build time)
- `Transolver2D` and `GNOT` are significantly slower per epoch than FNO-family
- `PACMANN` is a scaffold stub — replace operator blocks before using in research

### Planned Directions

- Push `burgers_1d` from 58× gap toward SOTA via PINO + physics-informed loss
- Close `darcy_2d` (14.6×) via GNOT attention or longer budgets
- Solve `ns_2d` last 1.12× gap with fine-grained HPO
- Multi-PDE foundation model pretraining
- Automated paper drafting from `trajectories.jsonl`

---

## Additional Resources

- [`RESEARCH_BRAIN.md`](./RESEARCH_BRAIN.md) — Living research driver: current strategy,
  empirical findings, SOTA gaps, session history (primary reference)
- [`WIKI.md`](./WIKI.md) — High-level system overview and architecture diagrams
- [MLX Documentation](https://ml-explore.github.io/mlx/) — Apple Silicon ML framework

---

*For authoritative research context, see [`RESEARCH_BRAIN.md`](./RESEARCH_BRAIN.md).*


<!-- STRUCTURE_START -->
```text
autoresearch-mlx/
├── agents/
│   └── skills/
│       └── SciMLx/
│           └── SKILL.md
├── core/
│   ├── __init__.py
│   ├── brain_distiller.py
│   ├── diagnostics.py
│   ├── hpo.py
│   ├── hypothesis.py
│   ├── loader.py
│   ├── losses.py
│   ├── mlflow_integration.py
│   ├── model_versioning.py
│   ├── paper_registry.py
│   ├── readme_hook.py
│   ├── research_plugins.py
│   ├── results_store.py
│   ├── scaffold.py
│   ├── tracker.py
│   ├── trainer.py
│   ├── utils.py
│   └── viz.py
├── dashboard/
│   ├── ui/
│   │   └── dashboard.html
│   ├── app.py
│   └── monitor.py
├── data/
│   ├── simulations/
│   │   ├── AGENTS.md
│   │   ├── __init__.py
│   │   ├── allen_cahn.py
│   │   ├── elasticity.py
│   │   ├── euler1d.py
│   │   ├── multiphysics.py
│   │   ├── ns_etdrk4.py
│   │   ├── pdebench.py
│   │   ├── radiative.py
│   │   ├── shallow_water.py
│   │   └── wavebench.py
│   ├── benchmarks_ext.py
│   ├── prefetch_data.py
│   └── prepare.py
├── docs/
│   ├── papers/
│   │   ├── afno_2022.yaml
│   │   ├── augmentation_2023.yaml
│   │   ├── curriculum_2009.yaml
│   │   ├── deeponet_2021.yaml
│   │   ├── ensemble_uq_2023.yaml
│   │   ├── ffno_2023.yaml
│   │   ├── fno_2020.yaml
│   │   ├── gnot_2023.yaml
│   │   ├── h1_loss.yaml
│   │   ├── hnn_2019.yaml
│   │   ├── inverse_pinn_2023.yaml
│   │   ├── mambano_2024.yaml
│   │   ├── memno_2025.yaml
│   │   ├── mppde_2022.yaml
│   │   ├── neural_ode_ude_2020.yaml
│   │   ├── physicsnemo_2024.yaml
│   │   ├── pikan_2025.yaml
│   │   ├── pino_2021.yaml
│   │   ├── rfno_2024.yaml
│   │   ├── ssm_s4_2022.yaml
│   │   ├── tfno_2022.yaml
│   │   ├── time_marching_deeponet_2025.yaml
│   │   ├── transolver_2024.yaml
│   │   ├── uno_2022.yaml
│   │   └── wno_2022.yaml
│   ├── ARCHITECTURE.md
│   ├── BENCHMARKS.md
│   ├── CONTRIBUTING.md
│   ├── LICENSE
│   ├── LITERATURE.md
│   ├── SOTA.md
│   └── TERMINOLOGY.md
├── mlruns/
│   ├── 0/
│   │   └── meta.yaml
│   ├── 1/
│   │   ├── 0258b914e3b4487abaec62470a8d1953/
│   │   │   └── artifacts/
│   │   │       ├── fno_h128_m22_l6.log
│   │   │       └── fno_h128_m22_l6_best.npz
│   │   ├── 04d31d5026ab4c8f99604425c0e8eab5/
│   │   │   └── artifacts/
│   │   │       ├── wno_h64_lvl3_l4_adapt.log
│   │   │       └── wno_h64_lvl3_l4_adapt_best.npz
│   │   ├── 084f1fb5b959456b8baf30f508c2c76f/
│   │   │   └── artifacts/
│   │   │       └── validation_ssno_burgers_best.npz
│   │   ├── 0a3b80a52e564c02891e941a7a808b25/
│   │   │   └── artifacts/
│   │   │       ├── fno_h128_m28_l6.log
│   │   │       └── fno_h128_m28_l6_best.npz
│   │   ├── 0d5c85aff8194650b4c9eb995c0d3c1d/
│   │   │   └── artifacts/
│   │   │       ├── gnot_burgers_h128_l8_adapt.log
│   │   │       └── gnot_burgers_h128_l8_adapt_best.npz
│   │   ├── 0d9843e5ed4f4c6696b4e751083c2cdd/
│   │   │   └── artifacts/
│   │   │       ├── mambano_burgers_curriculum_adapt.log
│   │   │       └── mambano_burgers_curriculum_adapt_best.npz
│   │   ├── 0da8cee76c0940e7a11f61401ccc022f/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_baseline_h128_l8_m24.log
│   │   │       └── fno_burgers_baseline_h128_l8_m24_best.npz
│   │   ├── 1595ab363eea4787a35699408245e097/
│   │   │   └── artifacts/
│   │   │       ├── gnot_burgers_h128_l8.log
│   │   │       └── gnot_burgers_h128_l8_best.npz
│   │   ├── 17e7b93e132a475fb39531e06aaee960/
│   │   │   └── artifacts/
│   │   │       └── afno_fix_burgers_v2_best.npz
│   │   ├── 1b67e87475cb48168969c8703ce1a06e/
│   │   │   └── artifacts/
│   │   │       ├── s4no_burgers_h128_l8_adapt.log
│   │   │       └── s4no_burgers_h128_l8_adapt_best.npz
│   │   ├── 1bdf65634b1c4aed8d443b2033769163/
│   │   │   └── artifacts/
│   │   │       ├── wno_burgers_h128_l8.log
│   │   │       └── wno_burgers_h128_l8_best.npz
│   │   ├── 221601c6125443bd8afbb342e1c65296/
│   │   │   └── artifacts/
│   │   │       ├── mambano_burgers_curriculum_adapt.log
│   │   │       └── mambano_burgers_curriculum_adapt_best.npz
│   │   ├── 24393f5f3f1f42f0af4fb21ba7248c96/
│   │   │   └── artifacts/
│   │   │       ├── fno_h128_m16_l8.log
│   │   │       └── fno_h128_m16_l8_best.npz
│   │   ├── 258ff04b7c214361802b830be31a1ff1/
│   │   │   └── artifacts/
│   │   │       ├── latent_ode_burgers_h64_l4.log
│   │   │       └── latent_ode_burgers_h64_l4_best.npz
│   │   ├── 28a02f8a1dd146c6ae80f5ebf0f65ab4/
│   │   │   └── artifacts/
│   │   │       └── afno_fix_v3_best.npz
│   │   ├── 2a133368cbbb4025802d02d1f79c1e61/
│   │   │   └── artifacts/
│   │   │       └── validation_curriculum_smooth_best.npz
│   │   ├── 39abf812035746f28f5bed6347d02e16/
│   │   │   └── artifacts/
│   │   │       ├── wno_burgers_h128_l8.log
│   │   │       └── wno_burgers_h128_l8_best.npz
│   │   ├── 3bc9ecb103cf4d7d894af3a9dd09b478/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_onecycle_aug_h128_l8_m24.log
│   │   │       └── fno_burgers_onecycle_aug_h128_l8_m24_best.npz
│   │   ├── 3ff8598ac74844fb8ef5c10b0d0517e8/
│   │   │   └── artifacts/
│   │   │       └── validation_pino_fix_best.npz
│   │   ├── 42daa0645e8a4ed1b857578a11a74c92/
│   │   │   └── artifacts/
│   │   │       ├── uno_burgers_h128_l8_m24_adapt.log
│   │   │       └── uno_burgers_h128_l8_m24_adapt_best.npz
│   │   ├── 4644174c59f24c9c860d933d656e9a08/
│   │   │   └── artifacts/
│   │   │       ├── fno_h128_m26_l6.log
│   │   │       └── fno_h128_m26_l6_best.npz
│   │   ├── 46b495cc22844de1b6412382e2bba147/
│   │   │   └── artifacts/
│   │   │       ├── rfno_h128_m24_l8.log
│   │   │       └── rfno_h128_m24_l8_best.npz
│   │   ├── 49314515a0bf48f4b9ac34ab89058f77/
│   │   │   └── artifacts/
│   │   │       ├── rfno_h128_m24_l12.log
│   │   │       └── rfno_h128_m24_l12_best.npz
│   │   ├── 4ca8a3f087664cffb6e89dd89eb5f924/
│   │   │   └── artifacts/
│   │   │       ├── uno_h64_l2.log
│   │   │       └── uno_h64_l2_best.npz
│   │   ├── 4d95bbd7cc9447e2a98362d265f69b95/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_curriculum_ema_h128_l8_m24.log
│   │   │       └── fno_burgers_curriculum_ema_h128_l8_m24_best.npz
│   │   ├── 506ffc3293fd4f05a28ed5bd783b201e/
│   │   │   └── artifacts/
│   │   │       ├── latent_ode_burgers_h64_l4_adapt.log
│   │   │       └── latent_ode_burgers_h64_l4_adapt_best.npz
│   │   ├── 52096fa8b71f452883fb1533c72045b3/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_curriculum_h128_l8_m24.log
│   │   │       └── fno_burgers_curriculum_h128_l8_m24_best.npz
│   │   ├── 525ba9ad978d48728ebb7ff0add92e68/
│   │   │   └── artifacts/
│   │   │       ├── wno_h64_lvl3_l4_adapt.log
│   │   │       └── wno_h64_lvl3_l4_adapt_best.npz
│   │   ├── 52bef8df45b341258f7ef3b0283812b1/
│   │   │   └── artifacts/
│   │   │       ├── uno_h64_l2.log
│   │   │       └── uno_h64_l2_best.npz
│   │   ├── 54625101ce2f4fed8363ecf8c6f11b4e/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_onecycle_aug_h128_l8_m24.log
│   │   │       └── fno_burgers_onecycle_aug_h128_l8_m24_best.npz
│   │   ├── 5b0c5120f3164fb69e286941440c18c8/
│   │   │   └── artifacts/
│   │   │       └── validation_ensemble_uq_best.npz
│   │   ├── 5cd42e5777d548909041c2da816c4df6/
│   │   │   └── artifacts/
│   │   │       ├── fno_h256_m16_l4.log
│   │   │       └── fno_h256_m16_l4_best.npz
│   │   ├── 606d9c2740674f18aa79fc6e8ab6832e/
│   │   │   └── artifacts/
│   │   │       ├── s4no_burgers_h128_l8.log
│   │   │       └── s4no_burgers_h128_l8_best.npz
│   │   ├── 62a1f12a4ee949cc8ef091284af86e68/
│   │   │   └── artifacts/
│   │   │       ├── fno_h256_m16_l6.log
│   │   │       └── fno_h256_m16_l6_best.npz
│   │   ├── 652a8361bc16407b9c9152c8cd55251b/
│   │   │   └── artifacts/
│   │   │       ├── wno_burgers_h128_l8_adapt.log
│   │   │       └── wno_burgers_h128_l8_adapt_best.npz
│   │   ├── 66c386387e78416d92d9c2c5c439cba2/
│   │   │   └── artifacts/
│   │   │       ├── gnot_burgers_h128_l8_adapt.log
│   │   │       └── gnot_burgers_h128_l8_adapt_best.npz
│   │   ├── 6880651ca46848b4b9bd66bdafd142e8/
│   │   │   └── artifacts/
│   │   │       ├── afno_h128_m24_l8.log
│   │   │       └── afno_h128_m24_l8_best.npz
│   │   ├── 6961141d109a4a0889d076157be65b61/
│   │   │   └── artifacts/
│   │   │       ├── rfno_h128_m24_l12.log
│   │   │       └── rfno_h128_m24_l12_best.npz
│   │   ├── 6b478292309345dc826df07ce6f41195/
│   │   │   └── artifacts/
│   │   │       ├── mambano_burgers_h128_l8_adapt.log
│   │   │       └── mambano_burgers_h128_l8_adapt_best.npz
│   │   ├── 6b9c061cab004bb28ff20c4df3c0857d/
│   │   │   └── artifacts/
│   │   │       ├── wno_burgers_h128_l8_adapt.log
│   │   │       └── wno_burgers_h128_l8_adapt_best.npz
│   │   ├── 795692bf22524f1ca68cefc5331b185c/
│   │   │   └── artifacts/
│   │   │       ├── mambano_burgers_curriculum_adapt.log
│   │   │       └── mambano_burgers_curriculum_adapt_best.npz
│   │   ├── 7a4afee8db27401cb6d44d35137ce91b/
│   │   │   └── artifacts/
│   │   │       ├── mambano_burgers_curriculum.log
│   │   │       └── mambano_burgers_curriculum_best.npz
│   │   ├── 7f374f9cfe614796abb7b50e32abbad3/
│   │   │   └── artifacts/
│   │   │       ├── fno_h64_m16_l6.log
│   │   │       └── fno_h64_m16_l6_best.npz
│   │   ├── 82983239f8fe4ae889d7ab5f2ef9d183/
│   │   │   └── artifacts/
│   │   │       └── mambano_burgers_h128_l8_h1_best.npz
│   │   ├── 82b4d128456245f59ad3b98f098b7f1c/
│   │   │   └── artifacts/
│   │   │       ├── mambano_burgers_h128_l8.log
│   │   │       └── mambano_burgers_h128_l8_best.npz
│   │   ├── 8480426796384931a278d1a9637ebe40/
│   │   │   └── artifacts/
│   │   │       ├── latent_ode_burgers_h64_l4_adapt.log
│   │   │       └── latent_ode_burgers_h64_l4_adapt_best.npz
│   │   ├── 859bd38f672a4d4283e3cdd9ac2bb045/
│   │   │   └── artifacts/
│   │   │       ├── uno_burgers_h128_l8_m24.log
│   │   │       └── uno_burgers_h128_l8_m24_best.npz
│   │   ├── 85f086e93ea7436297756d936587b32c/
│   │   │   └── artifacts/
│   │   │       ├── rfno_h128_m24_l10.log
│   │   │       └── rfno_h128_m24_l10_best.npz
│   │   ├── 8c68e130491a4227ac647c5e2ec8c956/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_onecycle_aug_h128_l8_m24.log
│   │   │       └── fno_burgers_onecycle_aug_h128_l8_m24_best.npz
│   │   ├── 8fd9f1169f8b41d1bc2260f6e105e600/
│   │   │   └── artifacts/
│   │   │       ├── wno_h64_lvl3_l4.log
│   │   │       └── wno_h64_lvl3_l4_best.npz
│   │   ├── 926613ee40c3494a834f7556a5d8d265/
│   │   │   └── artifacts/
│   │   │       ├── wno_h64_lvl3_l4.log
│   │   │       └── wno_h64_lvl3_l4_best.npz
│   │   ├── 9410178ddfd7418b8dedb23f859c2d87/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_onecycle_aug_h128_l8_m24.log
│   │   │       └── fno_burgers_onecycle_aug_h128_l8_m24_best.npz
│   │   ├── 9552f9b438df415c8e6bc221628f3791/
│   │   │   └── artifacts/
│   │   │       ├── gnot_burgers_h128_l8.log
│   │   │       └── gnot_burgers_h128_l8_best.npz
│   │   ├── 96c5b66cac274f6cb37a5ea71e5577c6/
│   │   │   └── artifacts/
│   │   │       ├── mambano_burgers_curriculum.log
│   │   │       └── mambano_burgers_curriculum_best.npz
│   │   ├── 991723e02a734ae5bd462831d8a34dae/
│   │   │   └── artifacts/
│   │   │       ├── fno_h256_m24_l6_adapt.log
│   │   │       └── fno_h256_m24_l6_adapt_best.npz
│   │   ├── 9e86917f5b7b44aca58b5efe1ab25bd3/
│   │   │   └── artifacts/
│   │   │       ├── uno_burgers_h128_l8_m24.log
│   │   │       └── uno_burgers_h128_l8_m24_best.npz
│   │   ├── adb4db5fd26141379bd5ef8186fc5ef9/
│   │   │   └── artifacts/
│   │   │       ├── fno_h128_m24_l8.log
│   │   │       └── fno_h128_m24_l8_best.npz
│   │   ├── ae8edfc6c83f45bfa27079a1cc97a431/
│   │   │   └── artifacts/
│   │   │       ├── fno_h64_m16_l6.log
│   │   │       └── fno_h64_m16_l6_best.npz
│   │   ├── b2d3c64c24cc4903be3b8d57b218e23c/
│   │   │   └── artifacts/
│   │   │       ├── mambano_burgers_h128_l8.log
│   │   │       └── mambano_burgers_h128_l8_best.npz
│   │   ├── b48aca5c6f9c42d4bfe9f37e38dc1ab2/
│   │   │   └── artifacts/
│   │   │       └── ffno_burgers_test_best.npz
│   │   ├── b51a0599257d4b19be7c77004ec67380/
│   │   │   └── artifacts/
│   │   │       ├── uno_h64_l1.log
│   │   │       └── uno_h64_l1_best.npz
│   │   ├── ba23979f4408492f97ee446c2b7e2cf4/
│   │   │   └── artifacts/
│   │   │       ├── memno_burgers_h128_l8.log
│   │   │       └── memno_burgers_h128_l8_best.npz
│   │   ├── ba31c07677694be0b084f9524a8a8adc/
│   │   │   └── artifacts/
│   │   │       ├── uno_burgers_h128_l8_m24_adapt.log
│   │   │       └── uno_burgers_h128_l8_m24_adapt_best.npz
│   │   ├── bedf4839a9f949028f1201abe5a0e398/
│   │   │   └── artifacts/
│   │   │       ├── memno_burgers_h128_l8.log
│   │   │       └── memno_burgers_h128_l8_best.npz
│   │   ├── bf4a87c76f2a4107a71f1a281d7f27e5/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_onecycle_aug_h128_l8_m24.log
│   │   │       └── fno_burgers_onecycle_aug_h128_l8_m24_best.npz
│   │   ├── c1e1b2c04a4f458a8e3b5bc53d036045/
│   │   │   └── artifacts/
│   │   │       ├── fno_h128_m16_l4.log
│   │   │       └── fno_h128_m16_l4_best.npz
│   │   ├── c387b05fd92a42578ce10f466366233e/
│   │   │   └── artifacts/
│   │   │       ├── wno_h64_lvl3_l4_adapt.log
│   │   │       └── wno_h64_lvl3_l4_adapt_best.npz
│   │   ├── c67ad2eb9b654fcdb50323264258ce10/
│   │   │   └── artifacts/
│   │   │       ├── fno_h128_m16_l4.log
│   │   │       └── fno_h128_m16_l4_best.npz
│   │   ├── c9489185269e4fe9b69ebcb08a00ad4e/
│   │   │   └── artifacts/
│   │   │       ├── uno_burgers_h128_l8_m24_adapt.log
│   │   │       └── uno_burgers_h128_l8_m24_adapt_best.npz
│   │   ├── ca1b039eecd6409b839b073ab3eb404f/
│   │   │   └── artifacts/
│   │   │       └── afno_fix_v4_best.npz
│   │   ├── cb983bf1ac8745aba584d041471fea91/
│   │   │   └── artifacts/
│   │   │       ├── fno_h128_m20_l6.log
│   │   │       └── fno_h128_m20_l6_best.npz
│   │   ├── d82956e1080c471e9b8cb359e85fff32/
│   │   │   └── artifacts/
│   │   │       ├── uno_h64_l2.log
│   │   │       └── uno_h64_l2_best.npz
│   │   ├── ddbcbb5c342f4046947b2a1ceef88dbd/
│   │   │   └── artifacts/
│   │   │       ├── fno_h128_m24_l6_v2.log
│   │   │       └── fno_h128_m24_l6_v2_best.npz
│   │   ├── df2e9e358269465da4886814ad222d8d/
│   │   │   └── artifacts/
│   │   │       ├── mambano_burgers_h128_l8_adapt.log
│   │   │       └── mambano_burgers_h128_l8_adapt_best.npz
│   │   ├── e358f6af2af8473fab5e6482338859c7/
│   │   │   └── artifacts/
│   │   │       ├── uno_burgers_h128_l8_m24.log
│   │   │       └── uno_burgers_h128_l8_m24_best.npz
│   │   ├── e3cfd3a548a540619fdac20356427b84/
│   │   │   └── artifacts/
│   │   │       ├── fno_h256_m16_l6.log
│   │   │       └── fno_h256_m16_l6_best.npz
│   │   ├── e44104abf0fe40719a1b32a1f305b7fa/
│   │   │   └── artifacts/
│   │   │       ├── uno_h64_l1.log
│   │   │       └── uno_h64_l1_best.npz
│   │   ├── e4494c64a8af447695620f81bc4aae66/
│   │   │   └── artifacts/
│   │   │       ├── wno_h64_lvl3_l4.log
│   │   │       └── wno_h64_lvl3_l4_best.npz
│   │   ├── e583018f152f4544bb845bdd9e553d9d/
│   │   │   └── artifacts/
│   │   │       └── repro_afno_bias_best.npz
│   │   ├── e9eac498a39e469c83328d7e18835fbc/
│   │   │   └── artifacts/
│   │   │       ├── latent_ode_burgers_h64_l4.log
│   │   │       └── latent_ode_burgers_h64_l4_best.npz
│   │   ├── ec87c364298a43abb3835ae27d49bc75/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_ema_h128_l8_m24.log
│   │   │       └── fno_burgers_ema_h128_l8_m24_best.npz
│   │   ├── ef048e585233496aadb8b8101b0098f4/
│   │   │   └── artifacts/
│   │   │       ├── fno_h256_m24_l6.log
│   │   │       └── fno_h256_m24_l6_best.npz
│   │   ├── f1af9da58ad844fdb68fefa657a46ba1/
│   │   │   └── artifacts/
│   │   │       ├── fno_h256_m16_l4.log
│   │   │       └── fno_h256_m16_l4_best.npz
│   │   ├── f3256b4b2dd74473ae13761d79e50249/
│   │   │   └── artifacts/
│   │   │       ├── mambano_burgers_curriculum.log
│   │   │       └── mambano_burgers_curriculum_best.npz
│   │   ├── fa0d3b808c50481d8cb1dd266f9cd2b5/
│   │   │   └── artifacts/
│   │   │       ├── rfno_h128_m24_l10.log
│   │   │       └── rfno_h128_m24_l10_best.npz
│   │   ├── facea104fb684955ae97c8d213b26a7a/
│   │   │   └── artifacts/
│   │   │       ├── s4no_burgers_h128_l8.log
│   │   │       └── s4no_burgers_h128_l8_best.npz
│   │   ├── fbcd8ef7ffc64985bbd5e0e206f335fa/
│   │   │   └── artifacts/
│   │   │       ├── fno_h128_m16_l8.log
│   │   │       └── fno_h128_m16_l8_best.npz
│   │   ├── ff0b719f90d149d49d96bc8b9b27e78d/
│   │   │   └── artifacts/
│   │   │       └── afno_heavy_test_best.npz
│   │   └── ffd6c0e9ec104d73b19ed9e257ddd00b/
│   │       └── artifacts/
│   │           ├── fno_h128_m16_l8.log
│   │           └── fno_h128_m16_l8_best.npz
│   ├── 10/
│   │   ├── 6afdce8a982f4a9d8138c3e9f41513df/
│   │   │   └── artifacts/
│   │   │       ├── hybridfnodeeponet_wavebench_2d_h32_l2_m8_f1_r1.log
│   │   │       └── hybridfnodeeponet_wavebench_2d_h32_l2_m8_f1_r1_best.npz
│   │   └── e8483d3f69024373a35d9c997cec8135/
│   │       └── artifacts/
│   │           ├── fedonet_wavebench_2d_h32_l2_m8_f1_r1.log
│   │           └── fedonet_wavebench_2d_h32_l2_m8_f1_r1_best.npz
│   ├── 2/
│   │   ├── 101fe0b7688e4303a829a3dcf9d7bfa8/
│   │   │   └── artifacts/
│   │   │       ├── ns2d_fno_h32_l4_m8.log
│   │   │       └── ns2d_fno_h32_l4_m8_best.npz
│   │   ├── b92c9f53bb6e47898f52a956abeb1ef3/
│   │   │   └── artifacts/
│   │   │       ├── agent_rfno2d_ns2d_h32_l4_m8_f1_r3.log
│   │   │       └── agent_rfno2d_ns2d_h32_l4_m8_f1_r3_best.npz
│   │   ├── d6877b1b2654460aaae90343a0496d8e/
│   │   │   └── artifacts/
│   │   │       └── validation_ns2d_cleanup_best.npz
│   │   └── ee35c7503a604f2fa109f8f154abad94/
│   │       └── artifacts/
│   │           ├── rfno2d_ns2d_h32_m8_l4_480s_f1.log
│   │           └── rfno2d_ns2d_h32_m8_l4_480s_f1_best.npz
│   ├── 3/
│   │   ├── 2800e885ed3f406cbd1b857dfa1f40dd/
│   │   │   └── artifacts/
│   │   │       └── smoke_test_2d_best.npz
│   │   ├── 303d458e47d14da7b3387f5e7b79c978/
│   │   │   └── artifacts/
│   │   │       ├── rfno2d_darcy2d_h32_l4_m8.log
│   │   │       └── rfno2d_darcy2d_h32_l4_m8_best.npz
│   │   ├── 365d5e0de857483f977d9d3501148397/
│   │   │   └── artifacts/
│   │   │       ├── rfno2d_darcy2d_h32_l4_m8_adapt.log
│   │   │       └── rfno2d_darcy2d_h32_l4_m8_adapt_best.npz
│   │   ├── 569846757b1e4791a71c2ccb9801c281/
│   │   │   └── artifacts/
│   │   │       ├── fedonet2d_darcy_h32_l2_m8_h1.log
│   │   │       └── fedonet2d_darcy_h32_l2_m8_h1_best.npz
│   │   ├── 5bb14a75e85143e29f7f25087212e71a/
│   │   │   └── artifacts/
│   │   │       ├── fedonet2d_darcy_h32_l4.log
│   │   │       └── fedonet2d_darcy_h32_l4_best.npz
│   │   ├── 7b6b50efb79d4eefaba32ec67f79cfd1/
│   │   │   └── artifacts/
│   │   │       ├── fno2d_darcy_h32_l4_m12_h1_aug.log
│   │   │       └── fno2d_darcy_h32_l4_m12_h1_aug_best.npz
│   │   ├── 7fc7b67f422d48459b86cf9fda4de340/
│   │   │   └── artifacts/
│   │   │       ├── fno2d_darcy_h32_l4_m8_h1.log
│   │   │       └── fno2d_darcy_h32_l4_m8_h1_best.npz
│   │   ├── 968bf7023a124d728e9a769ebc59a12b/
│   │   │   └── artifacts/
│   │   │       ├── transolver2d_darcy_s16_h24_l3.log
│   │   │       └── transolver2d_darcy_s16_h24_l3_best.npz
│   │   ├── 9d956408609745a3a3c06a9b22a3cfc8/
│   │   │   └── artifacts/
│   │   │       ├── fno_darcy2d_h32_l4_m8_h1_f1_r1.log
│   │   │       └── fno_darcy2d_h32_l4_m8_h1_f1_r1_best.npz
│   │   ├── abc12a20d416466b95ec4e8cc2f02cc2/
│   │   │   └── artifacts/
│   │   │       ├── fedonet2d_darcy_h32_l2_m8_h1_adapt.log
│   │   │       └── fedonet2d_darcy_h32_l2_m8_h1_adapt_best.npz
│   │   ├── b0b654187e3f4bf2bc185394e871a6b3/
│   │   │   └── artifacts/
│   │   │       ├── fno2d_darcy_h48_l6_m12_aug.log
│   │   │       └── fno2d_darcy_h48_l6_m12_aug_best.npz
│   │   ├── b7cb04e62ee74c7fae9b5e8326476472/
│   │   │   └── artifacts/
│   │   │       ├── autogen_darcy_2d_fno2d_h1_4991.log
│   │   │       └── autogen_darcy_2d_fno2d_h1_4991_best.npz
│   │   ├── d26aaa4bbf92492896d0150798bd90ae/
│   │   │   └── artifacts/
│   │   │       ├── transolver2d_darcy_h32_l4_s32_h1.log
│   │   │       └── transolver2d_darcy_h32_l4_s32_h1_best.npz
│   │   └── de4538a2e33a4120a0d19c01fea7c051/
│   │       └── artifacts/
│   │           ├── transolver2d_darcy_h32_l4_s32_h1_adapt.log
│   │           └── transolver2d_darcy_h32_l4_s32_h1_adapt_best.npz
│   ├── 4/
│   │   └── 7379f0f26da342ba8b2cf5512f0e00de/
│   │       └── artifacts/
│   │           ├── autogen_rayleigh_benard_2d_fno2d_h1adapt_2746.log
│   │           └── autogen_rayleigh_benard_2d_fno2d_h1adapt_2746_best.npz
│   ├── 415139728503581214/
│   │   ├── f487c4bd48de4f4b98dd180188324a57/
│   │   │   ├── artifacts/
│   │   │   ├── metrics/
│   │   │   │   ├── training_seconds
│   │   │   │   └── val_l2_rel
│   │   │   ├── params/
│   │   │   │   ├── hidden_dim
│   │   │   │   ├── lr
│   │   │   │   └── n_layers
│   │   │   ├── tags/
│   │   │   │   ├── benchmark
│   │   │   │   ├── exp_name
│   │   │   │   ├── mlflow.runName
│   │   │   │   ├── mlflow.source.name
│   │   │   │   ├── mlflow.source.type
│   │   │   │   ├── mlflow.user
│   │   │   │   └── model
│   │   │   └── meta.yaml
│   │   └── meta.yaml
│   ├── 5/
│   │   ├── 08d3eb38ad3543e3b3ea72b543aa0fd3/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h128_m32_l8_f1.log
│   │   │       └── rfno_kdv_h128_m32_l8_f1_best.npz
│   │   ├── 12244133c4c440a0bbdd13e6c80a91f9/
│   │   │   └── artifacts/
│   │   │       ├── hnn_kdv_h64_l4_f1_adapt.log
│   │   │       └── hnn_kdv_h64_l4_f1_adapt_best.npz
│   │   ├── 16fba21f31fe40a1842e6ae9d8cee6fd/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h128_m24_l10_f1.log
│   │   │       └── rfno_kdv_h128_m24_l10_f1_best.npz
│   │   ├── 1bef46653d1948e98f6a7b7a89217f5d/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h128_l8_m24_aug_f1.log
│   │   │       └── rfno_kdv_h128_l8_m24_aug_f1_best.npz
│   │   ├── 206e3290c9a44f508b4308a471cd7f13/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h256_m24_l8_f1.log
│   │   │       └── rfno_kdv_h256_m24_l8_f1_best.npz
│   │   ├── 4422da2cb28a477caed1a058c7b8fd1b/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h128_l8_m24_aug_f1_adapt.log
│   │   │       └── rfno_kdv_h128_l8_m24_aug_f1_adapt_best.npz
│   │   ├── 49ccb8972c394db28edb90b58067b28f/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h128_m24_l12_f1.log
│   │   │       └── rfno_kdv_h128_m24_l12_f1_best.npz
│   │   ├── 67770ade185b4ca99461438e2db27ccc/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h128_m24_l8_aug_f1_adapt.log
│   │   │       └── rfno_kdv_h128_m24_l8_aug_f1_adapt_best.npz
│   │   ├── 6b087efc74dc408ab765f3f512c96f29/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h128_m24_l10_f1_adapt.log
│   │   │       └── rfno_kdv_h128_m24_l10_f1_adapt_best.npz
│   │   ├── 6ec949c9530e453e9cf8663b08968a99/
│   │   │   └── artifacts/
│   │   │       ├── tfno_kdv_h128_l8_m24_f1_adapt.log
│   │   │       └── tfno_kdv_h128_l8_m24_f1_adapt_best.npz
│   │   ├── 72998ac24aff4c4fa01764967d39643a/
│   │   │   └── artifacts/
│   │   │       ├── rtfno_kdv_h128_l10_m24_f1_r2.log
│   │   │       └── rtfno_kdv_h128_l10_m24_f1_r2_best.npz
│   │   ├── 796ed6d2a596418bbe5f9458136ea9e3/
│   │   │   └── artifacts/
│   │   │       ├── s4d_kdv_h128_l6_p1_f1_adapt.log
│   │   │       └── s4d_kdv_h128_l6_p1_f1_adapt_best.npz
│   │   ├── 8960ed99b4af4c91a9331248b58a3dab/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h128_m24_l8_aug_f1.log
│   │   │       └── rfno_kdv_h128_m24_l8_aug_f1_best.npz
│   │   ├── 977c50210e144f03be4b54556a353332/
│   │   │   └── artifacts/
│   │   │       ├── rtfno_kdv_h128_l10_m24_f1_adapt.log
│   │   │       └── rtfno_kdv_h128_l10_m24_f1_adapt_best.npz
│   │   ├── a52e0e23a9ed48308082fd721c02f38d/
│   │   │   └── artifacts/
│   │   │       ├── fno_kdv_h256_m32_l8_f1.log
│   │   │       └── fno_kdv_h256_m32_l8_f1_best.npz
│   │   ├── b475d5c9b9d045ff98bca3c676a3ef2e/
│   │   │   └── artifacts/
│   │   │       ├── hnn_kdv_h64_l4_f1.log
│   │   │       └── hnn_kdv_h64_l4_f1_best.npz
│   │   ├── c3d252ccc6844db3ae9feb04346dedc4/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h128_m24_l12_f1.log
│   │   │       └── rfno_kdv_h128_m24_l12_f1_best.npz
│   │   ├── c486e0920f374ab6aff09baa8041404f/
│   │   │   └── artifacts/
│   │   │       ├── ffno_kdv_h256_m32_l8_f1.log
│   │   │       └── ffno_kdv_h256_m32_l8_f1_best.npz
│   │   ├── c9bbe68c2b1749e8abdb2209e76e85c0/
│   │   │   └── artifacts/
│   │   │       ├── tfno_kdv_h128_l8_m24_f1.log
│   │   │       └── tfno_kdv_h128_l8_m24_f1_best.npz
│   │   ├── cbc7e7b893394cc0b47f047282caed66/
│   │   │   └── artifacts/
│   │   │       ├── energy_fno_kdv_h128_l8_m24_f1_r1.log
│   │   │       └── energy_fno_kdv_h128_l8_m24_f1_r1_best.npz
│   │   ├── d35710cea6a9470e881f4e8a985e27f2/
│   │   │   └── artifacts/
│   │   │       ├── s4d_kdv_h128_l6_p1_f1_r1.log
│   │   │       └── s4d_kdv_h128_l6_p1_f1_r1_best.npz
│   │   └── fa8e55dc92e746a68cf779fbc1e87755/
│   │       └── artifacts/
│   │           ├── energy_fno_kdv_h128_l8_m24_f1_r1.log
│   │           └── energy_fno_kdv_h128_l8_m24_f1_r1_best.npz
│   ├── 6/
│   │   ├── 139236511c4c457bb4aed3a9d4693c32/
│   │   │   └── artifacts/
│   │   │       ├── hnn_wave_stable_h128_l4_f1.log
│   │   │       └── hnn_wave_stable_h128_l4_f1_best.npz
│   │   ├── 41709d212b0d41b4a3fe1c6e07a94f22/
│   │   │   └── artifacts/
│   │   │       ├── ssno_wave_h64_l4_m16_f1.log
│   │   │       └── ssno_wave_h64_l4_m16_f1_best.npz
│   │   ├── 4930a7ed6016418c825e16a034ba57a1/
│   │   │   └── artifacts/
│   │   │       ├── energy_fno_wave_h64_l8_m24_f1.log
│   │   │       └── energy_fno_wave_h64_l8_m24_f1_best.npz
│   │   ├── 5a567c3696464eedb5c6d4672edf15de/
│   │   │   └── artifacts/
│   │   │       ├── time_deeponet_wave_h128_l4_f1.log
│   │   │       └── time_deeponet_wave_h128_l4_f1_best.npz
│   │   ├── 5ab268190e304f4eb68dcafe52af52a3/
│   │   │   └── artifacts/
│   │   │       ├── time_deeponet_wave_h64_l4_f1_adapt.log
│   │   │       └── time_deeponet_wave_h64_l4_f1_adapt_best.npz
│   │   ├── 6361061fb11342a496d8a0cab4ec3036/
│   │   │   └── artifacts/
│   │   │       ├── hnn_wave_stable_h128_l4_f1_adapt.log
│   │   │       └── hnn_wave_stable_h128_l4_f1_adapt_best.npz
│   │   ├── 6af3c131a637401ea11f8b11a89ad060/
│   │   │   └── artifacts/
│   │   │       ├── energy_fno_wave_h64_l8_m24_f1_adapt.log
│   │   │       └── energy_fno_wave_h64_l8_m24_f1_adapt_best.npz
│   │   ├── 780e711f9b3046bfaa31d67e892f7fa8/
│   │   │   └── artifacts/
│   │   │       ├── rfno_wave_h64_l8_m24_f1.log
│   │   │       └── rfno_wave_h64_l8_m24_f1_best.npz
│   │   ├── 7f91822f4c994a418658506c9ae5d312/
│   │   │   └── artifacts/
│   │   │       ├── fno_wave_h128_m24_l8_v2_f1_r1.log
│   │   │       └── fno_wave_h128_m24_l8_v2_f1_r1_best.npz
│   │   ├── 98684af02b9345c0a833a7255cdee090/
│   │   │   └── artifacts/
│   │   │       ├── fno_wave_h128_m24_l8_v2_f1_adapt.log
│   │   │       └── fno_wave_h128_m24_l8_v2_f1_adapt_best.npz
│   │   ├── a406f7d4587e4f9b8f608af03bcba888/
│   │   │   └── artifacts/
│   │   │       ├── ssno_wave_h64_l4_m16_f1_adapt.log
│   │   │       └── ssno_wave_h64_l4_m16_f1_adapt_best.npz
│   │   ├── a8f8bfc3822f4c578f935b0b0542d7d6/
│   │   │   └── artifacts/
│   │   │       ├── rfno_wave_h64_l8_m24_f1_adapt.log
│   │   │       └── rfno_wave_h64_l8_m24_f1_adapt_best.npz
│   │   ├── c562ff4728c746aba70fdd7285aa0984/
│   │   │   └── artifacts/
│   │   │       ├── energy_fno_wave_h64_l8_m24_f1.log
│   │   │       └── energy_fno_wave_h64_l8_m24_f1_best.npz
│   │   ├── df35f9fc2ab7479396b238721f0e1086/
│   │   │   └── artifacts/
│   │   │       ├── time_deeponet_wave_h64_l4_f1.log
│   │   │       └── time_deeponet_wave_h64_l4_f1_best.npz
│   │   ├── f4175ebb4d344ef6b87755649bdeabb2/
│   │   │   └── artifacts/
│   │   │       ├── energy_fno_wave_h64_l8_m24_f1_adapt.log
│   │   │       └── energy_fno_wave_h64_l8_m24_f1_adapt_best.npz
│   │   └── f7b10fc9a98e47888ad511c2fc8d485e/
│   │       └── artifacts/
│   │           ├── time_deeponet_wave_h128_l4_f1_adapt.log
│   │           └── time_deeponet_wave_h128_l4_f1_adapt_best.npz
│   ├── 7/
│   │   ├── 5fe811b91d044cada21a9a27c6bf1062/
│   │   │   └── artifacts/
│   │   │       ├── fno_swe2d_h32_l4_m8_f1_r3.log
│   │   │       └── fno_swe2d_h32_l4_m8_f1_r3_best.npz
│   │   └── 75bc509ea3124ec987db454097cd3ef8/
│   │       └── artifacts/
│   │           ├── rfno2d_swe2d_h32_l4_m8_f1_r3.log
│   │           └── rfno2d_swe2d_h32_l4_m8_f1_r3_best.npz
│   ├── 8/
│   │   ├── b17f68b7cfa4473d837392205283553c/
│   │   │   └── artifacts/
│   │   │       ├── euler1d_fno_mc_h128_l8_m24_f1.log
│   │   │       └── euler1d_fno_mc_h128_l8_m24_f1_best.npz
│   │   └── ff2d6801c2d64b23993bedfd1f3ae3bf/
│   │       └── artifacts/
│   │           ├── euler1d_fno_mc_h128_l8_m24_f1_adapt.log
│   │           └── euler1d_fno_mc_h128_l8_m24_f1_adapt_best.npz
│   └── 9/
│       ├── 05542a845be644e4924ccbcd40879d7e/
│       │   └── artifacts/
│       │       ├── bnu001_transolver_h64_l4.log
│       │       └── bnu001_transolver_h64_l4_best.npz
│       ├── 16f88f4d8b5841fd8008eb959b41bfc6/
│       │   └── artifacts/
│       │       ├── bnu001_rfno_h128_l8_m32.log
│       │       └── bnu001_rfno_h128_l8_m32_best.npz
│       ├── 1c7c0aee885c4e359640a9ca50f21730/
│       │   └── artifacts/
│       │       ├── bnu001_gnot_h64_l4_n4.log
│       │       └── bnu001_gnot_h64_l4_n4_best.npz
│       ├── 2de0b85b62b1499cb345a1bbe8903d3b/
│       │   └── artifacts/
│       │       ├── bnu001_rfno_h256_l6_m32_lowlr.log
│       │       └── bnu001_rfno_h256_l6_m32_lowlr_best.npz
│       ├── 3464039546ee47d5b382eda99cbc7621/
│       │   └── artifacts/
│       │       ├── bnu001_rfno_h256_l6_m32_lowlr.log
│       │       └── bnu001_rfno_h256_l6_m32_lowlr_best.npz
│       ├── 431b84df8cc542efb46491a9d2f01225/
│       │   └── artifacts/
│       │       ├── bnu001_wno_h128_l10_lvl6.log
│       │       └── bnu001_wno_h128_l10_lvl6_best.npz
│       ├── 434cd075c72444159d818906cd62f24a/
│       │   └── artifacts/
│       │       ├── bnu001_mambano_h64_l6.log
│       │       └── bnu001_mambano_h64_l6_best.npz
│       ├── 46ced2eb68be44ad980af62b4aaabbd9/
│       │   └── artifacts/
│       │       ├── bnu001_pino_h192_l8_m24_hi_lambda_adapt.log
│       │       └── bnu001_pino_h192_l8_m24_hi_lambda_adapt_best.npz
│       ├── 616ada9edab14541b133dc9d8867baff/
│       │   └── artifacts/
│       │       ├── bnu001_wno_h128_l10_lvl6.log
│       │       └── bnu001_wno_h128_l10_lvl6_best.npz
│       ├── 6316c0abac874773b1c5cf4a1548c79a/
│       │   └── artifacts/
│       │       ├── bnu001_uno_h128_l6.log
│       │       └── bnu001_uno_h128_l6_best.npz
│       ├── 6a012c1373114556ab26526b1974efc2/
│       │   └── artifacts/
│       │       ├── bnu001_pino_h192_l8_m24_hi_lambda_adapt.log
│       │       └── bnu001_pino_h192_l8_m24_hi_lambda_adapt_best.npz
│       ├── 744f00395a004984aa69f126857cc9e6/
│       │   └── artifacts/
│       │       ├── bnu001_transolver_h64_l4.log
│       │       └── bnu001_transolver_h64_l4_best.npz
│       ├── a03ea0773732424f8091bd27adcaced9/
│       │   └── artifacts/
│       │       ├── bnu001_gnot_h64_l4_n4_adapt.log
│       │       └── bnu001_gnot_h64_l4_n4_adapt_best.npz
│       ├── a58923dd83984459b26b2a95b4f16d73/
│       │   └── artifacts/
│       │       ├── bnu001_pino_h192_l8_m24_hi_lambda.log
│       │       └── bnu001_pino_h192_l8_m24_hi_lambda_best.npz
│       ├── b9c98aae78e04c758ef2673d4f7fb844/
│       │   └── artifacts/
│       │       ├── bnu001_wno_h128_l10_lvl6_adapt.log
│       │       └── bnu001_wno_h128_l10_lvl6_adapt_best.npz
│       ├── c4a8d4a1a305497696ce4662a9b3be39/
│       │   └── artifacts/
│       │       ├── bnu001_pino_h192_l8_m24_hi_lambda.log
│       │       └── bnu001_pino_h192_l8_m24_hi_lambda_best.npz
│       ├── c8f7176995cd490bb741b4eb7fac4196/
│       │   └── artifacts/
│       │       ├── bnu001_wno_h128_l10_lvl6.log
│       │       └── bnu001_wno_h128_l10_lvl6_best.npz
│       ├── d654372787d74395b263f0bd9e2b30ce/
│       │   └── artifacts/
│       │       ├── bnu001_mambano_h64_l6.log
│       │       └── bnu001_mambano_h64_l6_best.npz
│       ├── e1a71c9d563b4f49b6a673f4e4aca9eb/
│       │   └── artifacts/
│       │       ├── bnu001_transolver_h64_l4_adapt.log
│       │       └── bnu001_transolver_h64_l4_adapt_best.npz
│       ├── f2f1f30a637b4968b21bb7d7d1da67ec/
│       │   └── artifacts/
│       │       ├── bnu001_ffno_h96_l8_m32.log
│       │       └── bnu001_ffno_h96_l8_m32_best.npz
│       ├── f4028ea352f84b37894b1b279cc1ed5f/
│       │   └── artifacts/
│       │       ├── bnu001_ffno_h96_l8_m32.log
│       │       └── bnu001_ffno_h96_l8_m32_best.npz
│       └── f84f76634f49494b879265b4b21cd1e7/
│           └── artifacts/
│               ├── bnu001_transolver_h64_l4_adapt.log
│               └── bnu001_transolver_h64_l4_adapt_best.npz
├── models/
│   ├── __init__.py
│   ├── afno.py
│   ├── attention_fno.py
│   ├── axial_attention.py
│   ├── chebyshev_kan.py
│   ├── deeponet.py
│   ├── fedonet.py
│   ├── fno.py
│   ├── gnot.py
│   ├── hano.py
│   ├── hnn.py
│   ├── hybrid_decoder_deeponet.py
│   ├── hybrid_fno_deeponet.py
│   ├── kan.py
│   ├── mamba_no.py
│   ├── mem_no.py
│   ├── neural_ode.py
│   ├── pacmann.py
│   ├── pinn.py
│   ├── s4d.py
│   ├── sno.py
│   ├── ssno.py
│   ├── tfno.py
│   ├── time_deeponet.py
│   ├── transolver.py
│   ├── vsmno.py
│   └── wno.py
├── notebooks/
│   └── colab_experiments.ipynb
├── scripts/
│   ├── maintenance/
│   │   ├── backfill_model_registry.py
│   │   ├── gen_arch_nanobanana.py
│   │   └── gen_arch_viz.py
│   ├── dvc_train.py
│   └── stress_worker.py
├── README.md
├── RESEARCH_BRAIN.md
├── WIKI.md
├── agent_loop.py
├── analyze.py
├── auto_suggest.py
├── autorun.py
├── dvc.yaml
├── experiments.yaml
├── mlflow.db
├── model_registry.json
├── params.yaml
├── pyproject.toml
├── results.db
├── results.db-shm
├── results.db-wal
├── results.json
├── test_jobless.py
├── train.py
└── uv.lock
```
<!-- STRUCTURE_END -->
