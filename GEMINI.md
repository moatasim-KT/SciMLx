# GEMINI.md — SciML AutoResearch (Apple Silicon / MLX)

## What this project does

An autonomous research platform for **Scientific Machine Learning (SciML)** on Apple Silicon. The loop trains neural PDE solvers within a fixed 5-minute budget and logs whether each run improves `val_l2_rel` (relative L2 error — lower is better). You (the external agent) can drive experiments manually or let `agent_loop.py` propose and run them automatically.

**Primary goal:** minimize `val_l2_rel` on fixed validation sets. The ground-truth evaluator lives in `prepare.py` — never modify it.

---

## Setup

```bash
uv sync           # install all dependencies (mlx, numpy, scipy, matplotlib, pyyaml, fastapi)
uv run prepare.py # one-time: generate + cache PDE datasets
```

---

## Two ways to drive experiments

### Mode A — External agent (you)

```bash
uv run analyze.py --papers          # 1. understand current state vs SOTA
uv run auto_suggest.py              # 2. get ranked suggestions with CLI commands
# 3. edit experiments.py or models/*.py
uv run autorun.py --priority 1 --commit   # 4. run the new queue
```

### Mode B — Autonomous loop (optional)

```bash
uv run agent_loop.py --dry-run      # preview without writing
uv run agent_loop.py --top 5        # propose + append 5 new configs
uv run agent_loop.py --run          # propose + run top-3 immediately
```

Mix both: external agent for novel ideas; `agent_loop.py` for overnight saturation.

---

## Key commands reference

```bash
# Single experiment (5-min budget)
uv run train.py --model FNO  --hidden 128 --layers 8  --modes 24
uv run train.py --model RFNO --hidden 128 --layers 8  --modes 24
uv run train.py --benchmark kdv_1d  --model RFNO --hidden 128 --layers 8 --modes 24
uv run train.py --benchmark wave_1d --model FNO  --hidden 64  --layers 4 --modes 16

# Run the pending queue
uv run autorun.py --priority 1 --commit          # run all priority-1 pending
uv run autorun.py --auto --commit                # run + re-suggest until queue empty
uv run autorun.py --dry-run                      # preview without running

# Analysis
uv run analyze.py --papers                       # results vs SOTA
uv run auto_suggest.py                           # ranked next-step suggestions
uv run auto_suggest.py --generate                # output ready-to-paste ExperimentConfig snippets
uv run auto_suggest.py --gaps                    # SOTA gap table
uv run paper_registry.py --pending               # unimplemented paper ideas

# Bayesian HPO (adaptive hyperparameter search)
uv run bayesian_hpo.py --benchmark burgers_1d --top 5
uv run bayesian_hpo.py --benchmark burgers_1d --multi --pareto  # acc + memory tradeoff

# Model scaffolding (gated: syntax → import → smoke test before registering)
uv run model_scaffold.py --stub MyModel --base FNO
uv run model_scaffold.py --register MyModel models/mymodel.py --benchmarks burgers_1d

# Dashboard (open ui/dashboard.html after starting)
uv run uvicorn app:app --reload --port 8000

# Visualization → figs/
uv run viz.py --mode leaderboard
```

---

## File map

| File | Role |
|------|------|
| `prepare.py` | **READ-ONLY.** PDE data generation + ground-truth evaluator. |
| `train.py` | Training harness. Routes benchmark → dataloader, enforces 5-min budget, emits `val_l2_rel`. |
| `trainer.py` | `AdamW`, `get_lr_schedule()` (warmup→flat→cosine), `Trainer` (JIT step). |
| `models/` | 14 model files, 23 exports (FNO, RFNO, AFNO, FFNO, UNO, WNO, DeepONet, S4NO, GNOT, PINN…). |
| `losses.py` | `get_loss_fn(name)` — l2_rel, h1, h1_strong, h2, spectral, l1_rel, mse. |
| `experiments.py` | 118 `ExperimentConfig` entries (declarative queue). Fields: `name`, `benchmark`, `model`, `hidden_dim`, `n_layers`, `n_modes`, `budget_s` (default 300s, 480s for 2D), `priority`, `parent_name`, `rationale`. |
| `results.json` | **SSoT.** DAG experiment tree. Never hand-edit — use `tracker.py`. |
| `results.tsv` | Append-only legacy log, synced from results.json. |
| `autorun.py` | Runs pending configs as subprocesses. Deduplicates, classifies crashes (OOM/NaN-Inf/ImportError/…), auto-retries once with halved params. `--commit` git-commits each kept result. |
| `agent_loop.py` | Mode B orchestrator. HypothesisEngine + BayesianHPO → new ExperimentConfig entries appended to experiments.py. |
| `bayesian_hpo.py` | GP surrogate (RBF kernel + EI). `BayesianHPO(bm, model, objectives=[...])`. Single-obj (`tell`/`ask`) or multi-obj (`tell_multi`/`pareto_front`). Seeds from results.json. |
| `model_scaffold.py` | Gated code generation: 3-gate validation (syntax → import → smoke) before registering new models. |
| `tracker.py` | `log_experiment(…, config={}, diag={}, parent_name="")`. `analyze_lineage()` for HP importance + trends. |
| `hypothesis.py` | `HypothesisEngine.analyze_benchmark(bm)` + `suggest_intervention(bm, best_val)` → concrete config dict. Detects spectral bias, shock errors, gradient collapse. |
| `diagnostics.py` | `calculate_spectral_bias(pred, target)` → low/mid/high-freq error. PNG to `figs/inspect_{id}.png`. |
| `auto_suggest.py` | Combines empirical wins + paper ideas + spectral diagnostic feedback + cross-benchmark transfer → ranked CLI suggestions. |
| `research_plugins.py` | `ModelRegistry` (14 models) + `BenchmarkRegistry` (10 benchmarks). Add models/benchmarks here. |
| `utils.py` | `REPO_ROOT`, `LOGS_DIR`, `FIGS_DIR`, `SOTA`, `load_results()`, `best_per_benchmark()`, `done_names()`. |
| `app.py` | FastAPI. Re-loads results.json on every request. Endpoints: `/api/experiments`, `/api/sota`, `/api/queue`, `/api/logs/{name}`, `/api/status`, `/api/pause`, `/api/resume`, `/api/inject`. |
| `ui/dashboard.html` | Standalone React dashboard. SOTA sidebar, Results + Queue tabs, right inspector with log viewer + training loss curve + diagnostics. |
| `benchmarks_ext.py` | KdV, Wave, Darcy-fix, NS-fix. |
| `simulations/` | 4 high-fidelity solvers: euler1d, shallow_water, allen_cahn, ns_etdrk4. |
| `papers/*.yaml` | 15 papers with key idea, reported results, our results, suggested experiments. |
| `docs/SOTA.md` | SOTA targets per benchmark. |
| `program.md` | Detailed research protocol for Mode A agents. |

---

## Model Zoo

| Model | Key idea | Status | Best |
|-------|----------|--------|------|
| `FNO` | Global Fourier spectral conv | ✓ | 0.1468 (burgers+aug) |
| `RFNO` | Pre-LN residual FNO (stable at depth) | ✓ | **0.0020** (kdv) — beats SOTA |
| `FNO2d` | 2D spectral conv | ✓ | darcy_2d |
| `FNO_MC` | Multi-channel [B,N,C]→[B,N,C] | ✓ | euler_1d only |
| `FFNO` | Factorized diagonal spectral conv | ~ | 0.24 (burgers) |
| `AFNO` | Block-diag MLP in Fourier | ✗ skip | 0.50–0.72 — wrong bias |
| `UNO` | U-Net encoder-decoder + FNO | ✓ | 0.715 (step-limited) |
| `WNO` | Haar wavelet conv | ✓ | wrong for periodic PDEs |
| `DeepONet` | Branch + Trunk inner product | ✓ | 0.808 |
| `PODDeepONet` | DeepONet + POD basis | ✓ | not benchmarked |
| `S4NO` | S4 state-space model | ✓ | not benchmarked |
| `GNOT` | Graph Neural Operator Transformer | ✓ | not benchmarked |
| `PINO` | Physics residual loss | ✗ broken | **never retry** |
| `PINN` | Standard physics-informed NN | ✓ | not benchmarked |

---

## Benchmark Catalog

| Benchmark | PDE | SOTA | Our best | Priority |
|-----------|-----|------|----------|----------|
| `burgers_1d` | 1D viscous Burgers | 0.0149 | 0.1468 | **HIGH** (10× gap) |
| `kdv_1d` | KdV soliton | 0.010 | **0.0020** ✓ | maintain lead |
| `wave_1d` | 1D wave | 0.005 | **0.000992** ✓ | maintain lead |
| `darcy_2d_fix` | 2D Darcy | 0.0108 | 0.1469 | HIGH (scale up) |
| `ns_2d_fix` | 2D NS vorticity | 0.0128 | 0.0152 | MEDIUM (near SOTA) |
| `euler_1d` | Compressible Euler | ~0.015 | not run | use `FNO_MC` |
| `swe_2d` | 2D Shallow Water | ~0.002 | not run | — |
| `allen_cahn_2d` | Allen-Cahn | ~0.020 | not run | — |
| `ns_hre_2d` | NS Re=1000 | ~0.070 | not run | ~70 min first gen |
| `darcy_2d` | 2D Darcy (broken) | — | 0.9986 | **DO NOT USE** |

---

## Loss functions

`--loss l2_rel` (default) · `h1` · `h1_strong` · `h2` · `spectral` · `l1_rel` · `mse`

H1 loss (α=0.01) marginally helps on Burgers; default l2_rel is fine for KdV and Wave.

---

## Empirical findings (102 experiments)

**Burgers (best 0.1468 — still 10× from SOTA 0.0149):**
- Optimal config: FNO h=128 l=8 m=24 + augmentation
- m=32 hurts (0.631); l>8 reduces training steps and hurts
- RFNO does not beat FNO here

**KdV (best 0.0020 — 5× better than SOTA):**
- RFNO h=128 l=8 m=24 is best; pre-LN residual helps soliton dynamics
- Wide models (h=256) hurt — fewer training steps

**Wave (best 0.000992 — 5× better than SOTA):**
- FNO h=64 l=4 m=16: smaller model → more steps within budget
- Don't use large models here

**Darcy / NS (2D):**
- `darcy_2d` is broken — use `darcy_2d_fix` only
- `darcy_2d_fix` h=32 is too small; try h=128, m=24
- `ns_2d_fix` at 0.0152 is close to SOTA 0.0128

---

## Hard constraints

1. **Never modify `prepare.py`** — it defines the ground-truth metric.
2. **No new packages** — only what's in `pyproject.toml`.
3. **`ExperimentConfig.name` must be unique** — it's the dedup key.
4. **Never hand-edit `results.json`** — use `tracker.py` API.
5. **`darcy_2d` is broken** — always use `darcy_2d_fix`.
6. **PINO is broken** — endpoint-only formulation always fails; never add PINO entries.
7. **Budget**: 1D benchmarks use `budget_s=300`; 2D benchmarks use `budget_s=480`.

---

## Current research priorities

1. **Close the Burgers gap** — best is 0.1468, SOTA is 0.0149 (10× gap). Try: curriculum training, ensemble, PINN variants, stronger augmentation, cross-architecture HPO.
2. **Scale up darcy_2d_fix** — current FNO h=32 gets 0.1469; need h=128+ with m=24.
3. **Benchmark unrun models** — S4NO, GNOT, PODDeepONet on burgers_1d and kdv_1d.
4. **Run simulation benchmarks** — euler_1d (FNO_MC), swe_2d, allen_cahn_2d.
5. **Push ns_2d_fix below SOTA** — 0.0152 vs 0.0128; try RFNO or wider FNO.

---

## How to add a new experiment (minimal example)

```python
# In experiments.py, append to EXPERIMENTS:
ExperimentConfig(
    name="rfno_burgers_curriculum_h128",
    benchmark="burgers_1d",
    model="RFNO",
    hidden_dim=128, n_layers=8, n_modes=24,
    budget_s=300,
    priority=1,
    rationale="Curriculum + RFNO to close Burgers gap from 0.1468 toward SOTA 0.0149",
)
```

Then run: `uv run autorun.py --priority 1 --commit`
