# autoresearch-sciml-mlx

Autonomous AI-driven research loop for **Scientific Machine Learning (SciML)** on Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx) — no PyTorch, no CUDA.

The agent explores PDE solver architectures (Neural Operators, PINNs, Neural ODEs) within a fixed 5-minute training budget per experiment, keeps improvements, and discards regressions — all tracked in a reproducible `results.json` DAG.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

---

## Current Results (102 experiments)

| Benchmark | SOTA | Our Best | Gap | Status |
|---|---|---|---|---|
| `burgers_1d` | 0.0149 | **0.1468** (FNO+aug) | 9.8× | Priority target |
| `kdv_1d` | ~0.010 | **0.0020** (RFNO) | 0.2× | **Beat SOTA 5×** ✓ |
| `wave_1d` | ~0.005 | **0.000992** (FNO) | 0.2× | **Beat SOTA 5×** ✓ |
| `darcy_2d_fix` | 0.0108 | 0.1469 (FNO) | 13.6× | Scale up needed |
| `ns_2d_fix` | 0.0128 | **0.0152** (FNO) | 1.2× | Near SOTA |
| `euler_1d` | ~0.015 | — | — | Not yet run |
| `swe_2d` | ~0.002 | — | — | Not yet run |
| `allen_cahn_2d` | ~0.020 | — | — | Not yet run |
| `ns_hre_2d` | ~0.070 | — | — | ~70 min first run |

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

## Model Zoo (14 models, 12 implemented)

| Model | Key Idea | Best Result |
|---|---|---|
| `FNO` | Global Fourier spectral conv | 0.1468 (burgers), **0.000992** (wave) |
| `RFNO` | Pre-LN residual FNO | **0.0020** (kdv — beat SOTA) |
| `FNO2d` | 2D spectral conv | darcy_2d_fix baseline |
| `FFNO` | Factorized diagonal spectral conv | 0.2405 (burgers) |
| `AFNO` | Block-diagonal MLP in Fourier | 0.50–0.72 — skip |
| `UNO` | U-Net + FNO layers | step-limited |
| `WNO` | Haar wavelet conv | wrong for periodic BCs |
| `DeepONet` | Branch + Trunk operator | 0.808 |
| `PODDeepONet` | DeepONet + POD basis | — |
| `S4NO` | S4 state-space neural operator | — |
| `GNOT` | Graph Neural Operator Transformer | — |
| `PINN` | Physics-informed NN | — |
| `HNN` / `NeuralODE` | Hamiltonian / latent dynamics | — |

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
| `experiments.py` | Declarative experiment queue (118 `ExperimentConfig` entries) |
| `results.json` | Source of truth — DAG of all 102 completed experiments |
| `models/` | 12 model implementations |
| `autorun.py` | Subprocess runner with auto-retry, crash classification, git commit |
| `agent_loop.py` | Mode B orchestrator (HypothesisEngine + Bayesian HPO) |
| `tracker.py` | DAG lineage engine, HP importance analysis |
| `auto_suggest.py` | Ranked next-step suggestion engine |
| `paper_registry.py` | 15 papers with SOTA targets and gap tracking |
| `app.py` | FastAPI dashboard backend |
| `ui/dashboard.html` | Standalone React dashboard |
| `program.md` | Agent research protocol |

---

## Empirical Findings

- **Burgers 1D**: `m=24` is the sweet spot; augmentation is the single biggest win (+38% improvement); RFNO does not beat FNO here; H1 loss barely helps.
- **KdV 1D**: RFNO with pre-LN residual stabilizes soliton dynamics — 5× better than SOTA in just 8 experiments.
- **Wave 1D**: Smaller/shallower model wins (`h=64 l=4`) because more training steps fit in the budget.
- **2D benchmarks**: `darcy_2d` solver is broken — always use `darcy_2d_fix` and `ns_2d_fix`.

---

## Acknowledgments

- Andrej Karpathy for the autonomous research concept
- [MLX](https://github.com/ml-explore/mlx) team at Apple
