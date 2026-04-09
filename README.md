# autoresearch-sciml-mlx

Autonomous AI-driven research loop for **Scientific Machine Learning (SciML)** on Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx) — no PyTorch, no CUDA.

The agent explores PDE solver architectures (Neural Operators, PINNs, Neural ODEs) within a fixed 5-minute training budget per experiment, keeps improvements, and discards regressions — all tracked in a reproducible `results.json` DAG.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

---

## Current Results (396 experiments)

| Benchmark | SOTA | Our Best | Gap | Status |
|---|---|---|---|---|
| `burgers_1d` | 0.0149 | **0.1468** (FNO+aug) | 9.8× | Priority target |
| `kdv_1d` | 0.0100 | **0.0020** (RFNO) | 0.2× | **Beat SOTA 5×** ✓ |
| `wave_1d` | 0.0050 | **0.000992** (FNO) | 0.2× | **Beat SOTA 5×** ✓ |
| `euler_1d` | 0.0150 | **0.0024** (FNO) | 0.16× | **Beat SOTA 6×** ✓ |
| `ns_2d_fix` | 0.0128 | **0.0152** (FNO) | 1.2× | Near SOTA |
| `darcy_2d_fix` | 0.0108 | **0.1041** (FNO) | 9.6× | Improving |
| `allen_cahn_2d` | 0.0200 | **0.0628** (FNO) | 3.1× | New |
| `swe_2d` | 0.0020 | **0.0107** (FNO2D) | 5.4× | New |
| `ns_hre_2d` | 0.0700 | — | — | Crashed |

---

## Quick Start

Requirements: **Apple Silicon Mac**, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies + generate PDE datasets (one-time)
uv sync
uv run prepare.py

# Run a single 5-minute experiment
uv run train.py --model FNO --hidden 128 --layers 8 --modes 24

# See current state vs SOTA
uv run analyze.py --papers

# Run all pending priority-1 experiments autonomously
uv run autorun.py --priority 1 --commit

# Web dashboard
uv run uvicorn app:app --reload --port 8000
# Open ui/dashboard.html in browser
```

---

## Model Zoo (30+ models implemented)

| Category | Models | Key Idea |
|---|---|---|
| **Neural Operators** | `FNO`, `RFNO`, `TFNO`, `FFNO`, `UNO` | Spectral convolutions (Fourier, Tensor-Factorized, U-Net) |
| **Attention-based** | `Transolver`, `GNOT`, `AFNO` | Physics Attention, Graph-based Transformers |
| **DeepONet Family** | `DeepONet`, `TimeDeepONet`, `DualDeepONet` | Branch/Trunk inner products, Time-marching variants |
| **State-Space** | `S4NO` | Structured State-Space Models (S4) |
| **Physics-Bias** | `HNN`, `EnergyFNO`, `PINN`, `PINO` | Hamiltonian/Symplectic priors, Soft energy conservation |
| **Differential Eq** | `NeuralODE`, `UDE`, `LatentODE` | Continuous-time integration, Universal Differential Equations |

---

## Orchestration

### Mode A — External Agent (default)

An external AI (Claude Code, Gemini, human) reads `program.md` and drives the loop:

```bash
uv run analyze.py --papers       # understand current state vs SOTA
uv run auto_suggest.py           # get ranked next-step suggestions
# edit experiments.py / models/*.py
uv run autorun.py --priority 1 --commit
```

### Mode B — Automated In-Process Loop

```bash
uv run agent_loop.py --dry-run   # preview proposals
uv run agent_loop.py --run       # generate + immediately run top-3
```

Mix freely: Mode A for novel ideas, Mode B for overnight saturation.

---

## Key Files

| File | Role |
|---|---|
| `prepare.py` | **Sacred** — generates PDE datasets, defines `evaluate_l2_rel`. Never modify. |
| `train.py` | Training harness — routes benchmarks, enforces 5-min budget, emits metrics |
| `experiments.py` | Declarative experiment queue (118+ `ExperimentConfig` entries) |
| `results.json` | Source of truth — DAG of all 396 completed experiments |
| `models/` | 30+ model implementations |
| `autorun.py` | Subprocess runner with auto-retry, crash classification, git commit |
| `agent_loop.py` | Mode B orchestrator (HypothesisEngine + Bayesian HPO) |
| `tracker.py` | DAG lineage engine, HP importance analysis |
| `auto_suggest.py` | Ranked next-step suggestion engine |
| `paper_registry.py` | 15+ papers with SOTA targets and gap tracking |
| `app.py` | FastAPI dashboard backend |
| `ui/dashboard.html` | Standalone React dashboard |
| `program.md` | Agent research protocol |

---

## Empirical Findings

- **Burgers 1D**: `m=24` is the sweet spot; augmentation is the single biggest win (+38% improvement); RFNO does not beat FNO here; H1 loss barely helps.
- **KdV 1D**: RFNO with pre-LN residual stabilizes soliton dynamics — 5× better than SOTA.
- **Wave 1D**: Smaller/shallower models win (`h=64 l=4`) within the fixed time budget.
- **Euler 1D**: FNO is highly efficient, beating SOTA by 6× (**0.0024** vs 0.0150).
- **Physics Priors**: `EnergyFNO` and `HNN` provide strong inductive biases for conservative systems (KdV, Wave) but are currently outpaced by pure FNO/RFNO efficiency within the 5-min budget.
- **2D benchmarks**: `darcy_2d_fix` and `ns_2d_fix` are moving toward SOTA; `ns_hre_2d` remains a challenge due to compute intensity.

---

## Acknowledgments

- Andrej Karpathy for the autonomous research concept
- [MLX](https://github.com/ml-explore/mlx) team at Apple
