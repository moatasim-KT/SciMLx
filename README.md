# autoresearch-sciml-mlx

Autonomous AI-driven research loop for **Scientific Machine Learning (SciML)** on Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx) — no PyTorch, no CUDA.

The agent explores PDE solver architectures (Neural Operators, PINNs, Neural ODEs) within a fixed 5-minute training budget per experiment, keeps improvements, discards regressions, and tracks everything in a reproducible `results.json` DAG.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

---

## Results vs SOTA (184+ experiments completed)

### Benchmarks that beat SOTA

| Benchmark | SOTA | Our Best | Ratio | Model | Note |
|---|---|---|---|---|---|
| `kdv_1d` | 0.0100 | **0.002023** | **5.0× better** | RFNO h=128 l=8 m=24 | Pre-LN residual stabilizes soliton dynamics |
| `wave_1d` | 0.0050 | **0.000992** | **5.0× better** | FNO h=64 l=4 m=16 | Smaller model = more steps in budget |
| `euler_1d` | 0.0150 | **0.002413** | **6.2× better** | FNO h=64 l=4 m=16 | FNO highly efficient on compressible Euler |

### Benchmarks in progress

| Benchmark | SOTA | Our Best | Gap | Priority |
|---|---|---|---|---|
| `burgers_1d` | 0.0031 | 0.1468 (FNO+aug) | 47.3× | **CRITICAL** — 135 experiments, gap remains huge |
| `darcy_2d` | 0.0041 | 0.1041 (FNO) | 25.4× | **HIGH** — 2D models need h≤32 l≤4 constraint |
| `ns_2d` | 0.0128 | 0.01428 (FNO 600s) | 1.12× | Medium — near SOTA, budget=600 key |
| `allen_cahn_2d` | 0.0200 | 0.0628 (FNO) | 3.14× | Medium — FNO h=64 l=4 is current best |
| `swe_2d` | 0.0020 | 0.0107 (FNO2D) | 5.36× | Medium — FNO2D h=32 l=4 is current best |
| `ns_hre_2d` | 0.0700 | — | — | Blocked — first-run ~70 min |

> SOTA targets are derived from `papers/*.yaml` (e.g., GNOT-2023 for Burgers/Darcy).

---

## Quick Start

**Requirements:** Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Pre-generate and disk-cache ALL PDE datasets
# (one-time, ~20 min — dominated by ns_hre_2d)
uv run python -m data.prefetch_data

# Run a single 5-minute experiment
uv run train.py --model FNO --hidden 128 --layers 8 --modes 24

# See current state vs SOTA
PYTHONPATH=. uv run python -m analyze --papers

# Run all pending priority-1 experiments autonomously
uv run autorun.py --priority 1 --commit

# Web dashboard
PYTHONPATH=. uv run python -m dashboard.app
# Open http://localhost:8000 in browser
```

---

## Orchestration Modes

### Mode A — External Agent (default)

A human or AI (Claude Code, Gemini) drives the loop manually:

```bash
# 1. Understand current state vs SOTA
PYTHONPATH=. uv run python -m analyze --papers

# 2. Get ranked next-step suggestions
PYTHONPATH=. uv run python -m auto_suggest --gaps

# 3. Edit experiments.yaml / models/*.py

# 4. Trigger runner
uv run autorun.py --priority 1 --commit
```

### Mode B — Automated In-Process Loop

Fully automated: `HypothesisEngine` + `BayesianHPO` generate and run new configs without human input.

```bash
PYTHONPATH=. uv run python -m agent_loop --dry-run   # preview only
PYTHONPATH=. uv run python -m agent_loop --top 5     # append top-5
PYTHONPATH=. uv run python -m agent_loop --run       # append + run top-3
```

---

## Model Zoo

| Category | Model Tags | Key Idea | Status |
|---|---|---|---|
| **Fourier Neural Operators** | `FNO`, `RFNO`, `FFNO`, `FNO2D`, `FNO_MC` | Global spectral conv; RFNO adds pre-LN residuals for stability at depth | ✓ |
| **Tensor-Factorized FNO** | `TFNO`, `RTFNO`, `CPFNO` | Tucker / CP decomposition to reduce spectral param count | ✓ |
| **U-Net Operator** | `UNO` | Encoder-decoder with FNO layers; good for multiscale | ✓ |
| **Attention-based** | `Transolver`, `Transolver2D`, `GNOT`, `GNOT2D`, `AFNO` | Physics Attention; Graph Neural Operator Transformer; Block-diagonal Fourier MLP | ✓/⚠ |
| **DeepONet Family** | `DeepONet`, `PODDeepONet`, `TimeDeepONet`, `DualDeepONet` | Branch/Trunk inner products; POD basis; time-marching variants | ✓ |
| **State-Space** | `S4NO`, `SSNO` | S4 structured state-space; SSNO adds adaptive S4D damping + spectral conv dual-branch | ✓ |
| **Physics-Biased** | `HNN`, `EnergyFNO`, `PINN` | Hamiltonian/Symplectic priors; soft energy conservation; PDE residual loss | ✓ |
| **Differential Eq** | `NeuralODE`, `UDE`, `LatentODE` | Continuous-time integration; Universal Differential Equations | ✓ |

> `PINO` is implemented but broken for endpoint-only formulations — never use.
> `AFNO` has wrong spectral bias (0.50–0.72 on Burgers) — skip.
> `Transolver2D` is unreliable on 2D benchmarks (stalls/crashes).
> **RFNO is 1D-only** — crashes on 2D input.

---

## File Structure

The repository is organized into a **minimalist root** structure to separate core research logic from simulation data and dev tools.

<!-- STRUCTURE_START -->
```text
autoresearch-mlx/
├── core/
│   ├── __init__.py
│   ├── diagnostics.py
│   ├── hypothesis.py
│   ├── losses.py
│   ├── research_plugins.py
│   ├── tracker.py
│   ├── trainer.py
│   └── utils.py
├── dashboard/
│   ├── ui/
│   │   └── dashboard.html
│   └── app.py
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
│   │   ├── AGENTS.md
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
│   │   ├── mppde_2022.yaml
│   │   ├── neural_ode_ude_2020.yaml
│   │   ├── physicsnemo_2024.yaml
│   │   ├── pino_2021.yaml
│   │   ├── rfno_2024.yaml
│   │   ├── ssm_s4_2022.yaml
│   │   ├── tfno_2022.yaml
│   │   ├── time_marching_deeponet_2025.yaml
│   │   ├── transolver_2024.yaml
│   │   ├── uno_2022.yaml
│   │   └── wno_2022.yaml
│   ├── AGENTS.md
│   ├── CLAUDE.md
│   ├── GEMINI.md
│   ├── LICENSE
│   ├── LITERATURE.md
│   ├── SOTA.md
│   ├── TERMINOLOGY.md
│   └── program.md
├── models/
│   ├── AGENTS.md
│   ├── __init__.py
│   ├── afno.py
│   ├── attention_fno.py
│   ├── axial_attention.py
│   ├── deeponet.py
│   ├── fedonet.py
│   ├── fno.py
│   ├── gnot.py
│   ├── hano.py
│   ├── hnn.py
│   ├── hybrid_decoder_deeponet.py
│   ├── hybrid_fno_deeponet.py
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
├── README.md
├── agent_loop.py
├── analyze.py
├── auto_suggest.py
├── autorun.py
├── bayesian_hpo.py
├── experiments.py
├── experiments.yaml
├── model_scaffold.py
├── monitor.py
├── paper_registry.py
├── pyproject.toml
├── results.json
├── results.tsv
├── train.py
├── update_readme.py
├── uv.lock
└── viz.py
```
<!-- STRUCTURE_END -->

## Empirical Findings (184 experiments)

**KdV 1D** (best: 0.0020 — **5× better than SOTA**):
- RFNO with pre-LN residuals stabilizes soliton dynamics; standard FNO diverges at depth ≥10

**Wave 1D** (best: 0.000992 — **5× better than SOTA**):
- Smaller/shallower models win (`h=64 l=4`): more gradient steps inside the 5-min budget dominate accuracy.

**Euler 1D** (best: 0.0024 — **6× better than SOTA**):
- FNO is highly efficient on compressible system dynamics.

**Burgers 1D** (best: 0.1468 — 47× gap to GNOT SOTA):
- `m=24` is the sweet spot; `h=128` wins over 64/256. Augmentation is critical.

**NS 2D** (best: 0.01428 — 1.12× gap):
- Extended budget (600s) is key. `n_modes=8` beats `n_modes=12`.

---

## Infrastructure Features

### 1. Unified REPO_ROOT
All file paths (results, logs, papers) are resolved from a single source of truth in `core/utils.py`, allowing scripts to run from any depth.

### 2. Bayesian HPO Auto-Loop
`tools/agent_loop.py` uses Gaussian Process surrogates and Expected Improvement (EI) to autonomously explore hyperparameter space.

### 3. Multi-Fix Crash Recovery
`autorun.py` automatically detects crash reasons (OOM, NaN, Loss Spike) and applies composite fixes (e.g. `batch_size//2` + `lr//10`) in real-time.

### 4. Interactive Dashboard
FastAPI backend (`apps/app.py`) serves a live React-based diagnostic dashboard for lineage tracking and VRAM monitoring.

---

## Hard Constraints

- **Never modify `data/prepare.py`** — defines the ground-truth metric.
- **`results.json` is SSoT** — never hand-edit; managed via `core/tracker.py`.
- **2D benchmarks**: use `h≤32 l≤4` to avoid OOM on unified memory.
- **Namespace consistency**: always import using `core.`, `data.`, or `tools.` prefixes.

---

## Acknowledgments

- Andrej Karpathy for the autonomous research concept
- [MLX](https://github.com/ml-explore/mlx) team at Apple
