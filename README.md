# autoresearch-sciml-mlx

Autonomous AI-driven research loop for **Scientific Machine Learning (SciML)** on Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx) — no PyTorch, no CUDA.

The agent explores PDE solver architectures (Neural Operators, PINNs, Neural ODEs) within a fixed 5-minute training budget per experiment, keeps improvements, discards regressions, and tracks everything in a reproducible `results.json` DAG.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

---

## Results vs SOTA (173 experiments completed)

### Benchmarks that beat SOTA

| Benchmark | SOTA | Our Best | Ratio | Model | Note |
|---|---|---|---|---|---|
| `kdv_1d` | 0.0100 | **0.002023** | **5.0× better** | RFNO h=128 l=8 m=24 | Pre-LN residual stabilizes soliton dynamics |
| `wave_1d` | 0.0050 | **0.000992** | **5.0× better** | FNO h=64 l=4 m=16 | Smaller model = more steps in budget |
| `euler_1d` | 0.0150 | **0.002413** | **6.2× better** | FNO h=64 l=4 m=16 | FNO highly efficient on compressible Euler |

### Benchmarks in progress

| Benchmark | SOTA | Our Best | Gap | Priority |
|---|---|---|---|---|
| `burgers_1d` | 0.0149 | 0.1468 (FNO+aug) | 9.8× | High — 115 experiments, augmentation key |
| `darcy_2d_fix` | 0.0108 | 0.1041 (FNO) | 9.6× | Medium — need larger model |
| `allen_cahn_2d` | 0.0200 | 0.0628 (FNO) | 3.1× | Medium — 5 experiments so far |
| `swe_2d` | 0.0020 | 0.0107 (FNO2D) | 5.4× | Medium — 5 experiments so far |
| `ns_2d_fix` | 0.0128 | 0.0152 (FNO) | 1.2× | Low — near SOTA, needs tuning |
| `ns_hre_2d` | 0.0700 | — | — | Blocked — first-run ~70 min |

> `darcy_2d` (broken solver) is excluded. `ns_hre_2d` requires a one-time ~70min data generation.

---

## Quick Start

**Requirements:** Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies + generate PDE datasets (one-time, ~5 min)
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

## Orchestration Modes

### Mode A — External Agent (default)

A human or AI (Claude Code, Gemini) drives the loop manually:

```bash
uv run analyze.py --papers       # understand current state vs SOTA
uv run auto_suggest.py           # get ranked next-step suggestions
# edit experiments.py / models/*.py
uv run autorun.py --priority 1 --commit
```

### Mode B — Automated In-Process Loop

Fully automated: `HypothesisEngine` + `BayesianHPO` generate and run new configs without human input.

```bash
uv run agent_loop.py --dry-run   # preview proposals, write nothing
uv run agent_loop.py --top 5     # append top-5 new configs
uv run agent_loop.py --run       # append + immediately run top-3
```

Mix freely: Mode A for novel ideas, Mode B for overnight saturation.

---

## Model Zoo (27 registered models)

| Category | Model Tags | Key Idea |
|---|---|---|
| **Fourier Neural Operators** | `FNO`, `RFNO`, `FFNO`, `FNO2D`, `FNO_MC` | Global spectral conv; RFNO adds pre-LN residuals for stability at depth |
| **Tensor-Factorized FNO** | `TFNO`, `RTFNO`, `CPFNO` | Tucker / CP decomposition to reduce spectral param count |
| **U-Net Operator** | `UNO` | Encoder-decoder with FNO layers; good for multiscale |
| **Attention-based** | `Transolver`, `Transolver2D`, `GNOT`, `GNOT2D`, `AFNO` | Physics Attention; Graph Neural Operator Transformer; Block-diagonal Fourier MLP |
| **DeepONet Family** | `DeepONet`, `PODDeepONet`, `TimeDeepONet`, `DualDeepONet` | Branch/Trunk inner products; POD basis; time-marching variants |
| **State-Space** | `S4NO` | Structured State-Space Models (S4) for operator learning |
| **Physics-Biased** | `HNN`, `EnergyFNO`, `PINN` | Hamiltonian/Symplectic priors; soft energy conservation; PDE residual loss |
| **Differential Eq** | `NeuralODE`, `UDE`, `LatentODE` | Continuous-time integration; Universal Differential Equations |

> `PINO` is implemented but broken for endpoint-only formulations — never use.
> `AFNO` has wrong spectral bias (0.50–0.72 on Burgers) — skip.

---

## File Structure

```
autoresearch-mlx/
│
├── train.py              # Training harness — routes benchmarks, enforces 5-min budget
├── trainer.py            # AdamW, LR schedule (warmup→flat→cosine), Trainer class (JIT)
├── prepare.py            # SACRED — PDE dataset generation + evaluate_l2_rel. Never modify.
├── experiments.py        # Declarative experiment queue (169+ ExperimentConfig entries)
├── results.json          # Source of truth — DAG of all 173 completed experiments
├── results.tsv           # Append-only log, synced from results.json
│
├── models/               # All model implementations (12 files, 27 registered models)
│   ├── fno.py            # FNO, RFNO, FNO2d, FNO1dMC
│   ├── tfno.py           # TFNO, RTFNO, CPFNO, TFNO2d
│   ├── afno.py           # AFNO (block-diagonal Fourier MLP)
│   ├── deeponet.py       # DeepONet, PODDeepONet
│   ├── time_deeponet.py  # TimeDeepONet, DualDeepONet
│   ├── wno.py            # WNO (Haar wavelet conv — non-periodic BCs)
│   ├── s4d.py            # S4NO (structured state-space)
│   ├── gnot.py           # GNOT, GNOT2d (graph neural operator transformer)
│   ├── transolver.py     # Transolver, Transolver2D (physics attention)
│   ├── hnn.py            # HNN, EnergyFNO (Hamiltonian / energy-conserving)
│   ├── neural_ode.py     # NeuralODE, UDE, LatentODE
│   └── pinn.py           # PINN, PINO
│
├── simulations/          # High-fidelity PDE solvers (cached after first run)
│   ├── euler1d/          # Compressible Euler 1D
│   ├── shallow_water/    # 2D Shallow Water Equations
│   ├── allen_cahn/       # Allen-Cahn phase field 2D
│   └── ns_etdrk4/        # Navier-Stokes Re=1000 2D (~70 min first run)
│
├── papers/               # 20 papers as YAML (title, SOTA, suggested experiments, status)
├── docs/                 # SOTA.md, LITERATURE.md, TERMINOLOGY.md
├── figs/                 # Auto-generated plots and diagnostic inspect images
├── logs/                 # Per-experiment training logs (logs/<name>.log)
├── ui/                   # dashboard.html — standalone React app
│
├── autorun.py            # Subprocess runner: dedup, crash classification, auto-retry, git commit
├── agent_loop.py         # Mode B orchestrator (HypothesisEngine → BayesianHPO → append configs)
├── auto_suggest.py       # Ranked next-step suggestions (empirical wins + paper ideas + spectral diag)
├── analyze.py            # Full results report with SOTA gap analysis
├── tracker.py            # DAG lineage engine; HP importance (Pearson); trend detection
├── hypothesis.py         # Failure analysis: spectral bias → modes, shocks → UNO, collapse → RFNO
├── bayesian_hpo.py       # Gaussian Process surrogate (RBF + Expected Improvement)
├── diagnostics.py        # Spectral bias metrics; inspect_*.png generation
├── model_scaffold.py     # Gated code generation: stub → syntax → import → smoke test → register
├── paper_registry.py     # Loads papers/*.yaml; reports SOTA gaps; lists pending implementations
├── research_plugins.py   # ModelRegistry + BenchmarkRegistry (auto-routes train.py)
├── losses.py             # Loss factory: l2_rel, h1, h1_strong, h2, spectral, l1_rel, mse
├── utils.py              # Shared constants: SOTA targets, load_results(), done_names()
├── benchmarks_ext.py     # KdV, Wave, Darcy-fix, NS-fix extended benchmark runners
├── viz.py                # Leaderboard / training / spectral / architecture plots → figs/
├── app.py                # FastAPI dashboard backend (live-reloads results.json)
├── monitor.py            # VRAM telemetry monitor
├── program.md            # Agent research protocol (primary guide for Mode A)
└── CLAUDE.md             # AI agent instructions (Claude Code / external agents)
```

---

## Empirical Findings

**KdV 1D** (best: 0.0020 — **5× better than SOTA**):
- RFNO with pre-LN residuals stabilizes soliton dynamics; standard FNO diverges at depth ≥10

**Wave 1D** (best: 0.000992 — **5× better than SOTA**):
- Smaller/shallower models win (`h=64 l=4`): more gradient steps inside the 5-min budget dominate accuracy
- "Easy" for Fourier methods — the training-step count, not model size, is the bottleneck

**Euler 1D** (best: 0.0024 — **6× better than SOTA**):
- FNO is highly efficient; `FNO_MC` (multi-channel for 3-variable system) didn't outperform standard FNO

**Burgers 1D** (best: 0.1468 — 9.8× gap, 115 experiments):
- `m=24` is the sweet spot (m=32 hurts; m=16 too few modes)
- `h=128` wins over h=64 and h=256 (h=256 is step-time limited)
- Augmentation is the single biggest win: +aug → 0.1468 from 0.155
- RFNO does not beat FNO on Burgers; H1 loss barely helps

**NS 2D** (best: 0.0152 — 1.2× gap):
- FNO baseline is essentially at SOTA with corrected initial conditions (`ns_2d_fix`)

**General:**
- Fixed 5-min budget means shallow+wide models often beat deep+narrow (more steps)
- `darcy_2d` has a broken solver — always use `darcy_2d_fix`
- `ns_hre_2d` requires ~70 min first-run data generation, then is disk-cached

---

## Loss Library

| Flag | Formula | Best for |
|---|---|---|
| `l2_rel` | mean(‖pred−y‖/‖y‖) | Default — all PDEs |
| `h1` | L2 + 0.01·L2(∂u/∂x) | Shocks (Burgers, KdV) |
| `h1_strong` | H1 with α=1.0 | Strong gradient focus |
| `h2` | L2 + α·L2(∂²u/∂x²) | Smoothness regularization |
| `spectral` | Frequency-weighted L2 | High-frequency errors |
| `l1_rel` | mean(‖pred−y‖₁/‖y‖₁) | Outlier-robust |
| `mse` | Mean squared error | Debugging / baselines |

---

## Dashboard

```bash
uv run uvicorn app:app --reload --port 8000
# Open ui/dashboard.html in browser
```

Features: SOTA sidebar with progress bars, sortable/searchable results table, pending queue view, right inspector with config grid + training loss curve + log viewer + spectral diagnostics.

API endpoints: `/api/experiments`, `/api/sota`, `/api/queue`, `/api/logs/{name}`, `/api/status`, `/api/pause`, `/api/resume`, `/api/inject`, `/api/priority`, `/api/lineage`

---

## Hard Constraints

- **Never modify `prepare.py`** — it defines the ground-truth metric. Touching it invalidates all results.
- **No new packages** beyond `pyproject.toml` (mlx, numpy, scipy, matplotlib, pyyaml, fastapi, uvicorn).
- **`ExperimentConfig.name` must be globally unique** — it is the dedup key.
- **`results.json` is SSoT** — never hand-edit; all writes go through `tracker.py`.
- **`darcy_2d` is broken** — only use `darcy_2d_fix` and `ns_2d_fix`.
- **PINO is broken** for endpoint-only formulations — never queue PINO experiments.
- **Git hygiene** — stage only `train.py`, `models/`, `experiments.py`, `results.tsv`. Never `git add -A`.

---

## Acknowledgments

- Andrej Karpathy for the autonomous research concept
- [MLX](https://github.com/ml-explore/mlx) team at Apple
