# CLAUDE.md

Guidance for Claude Code and any external AI agent working in this repository.

---

## Quick Start

```bash
uv sync                    # install dependencies
uv run prepare.py          # one-time: generate + cache PDE datasets

# Run a single experiment (~6-8 min on Apple Silicon)
uv run train.py --model FNO --hidden 128 --layers 8 --modes 24

# See current state vs SOTA
uv run analyze.py --papers
uv run auto_suggest.py --gaps

# Run all pending priority-1 experiments
uv run autorun.py --priority 1 --commit

# Dashboard (open ui/dashboard.html in browser after starting)
uv run uvicorn app:app --reload --port 8000
```

---

## Orchestration Modes

There are two ways to drive the research loop. Both write to the same
`results.json` / `results.tsv` and share the `experiments.py` queue.

### Mode A — External Agent (this file, original, always supported)

An external AI (Claude Code, Gemini, human) reads this file and drives
the loop manually.

```bash
1. uv run analyze.py --papers       # understand current state vs SOTA
2. uv run auto_suggest.py           # get ranked suggestions
3. # Edit experiments.py / models/*.py as needed
4. uv run autorun.py --priority 1 --commit
5. Repeat from step 1
```

### Mode B — agent_loop.py (optional, automated)

A fully automated in-process loop that generates new `ExperimentConfig`
entries using the HypothesisEngine + Bayesian HPO, without any human or
external-agent input.

```bash
uv run agent_loop.py --dry-run           # preview proposals, write nothing
uv run agent_loop.py --top 5             # generate 5 new configs + append
uv run agent_loop.py --run               # generate + immediately run top-3
uv run agent_loop.py --benchmark kdv_1d  # focus on one benchmark
```

Mix freely: use Mode A for novel ideas in the morning, Mode B for
overnight saturation.

---

## All Commands

```bash
# ── Training ──────────────────────────────────────────────────────────────────
uv run train.py --model FNO  --hidden 128 --layers 8  --modes 24
uv run train.py --model RFNO --hidden 128 --layers 10 --modes 24
uv run train.py --model AFNO --hidden 128 --layers 8  --modes 24
uv run train.py --model FFNO --hidden 256 --layers 8  --modes 24
uv run train.py --model FNO  --hidden 128 --layers 8  --modes 24 --loss h1

# Extended benchmarks
uv run train.py --benchmark kdv_1d       --model RFNO --hidden 128 --layers 8  --modes 24
uv run train.py --benchmark wave_1d      --model FNO  --hidden 64  --layers 4  --modes 16
uv run train.py --benchmark darcy_2d_fix --model FNO  --hidden 64  --layers 4  --modes 12
uv run train.py --benchmark ns_2d_fix    --model FNO  --hidden 64  --layers 4  --modes 12

# High-fidelity simulations (ns_hre_2d first run ~70 min, cached after)
uv run train.py --benchmark euler_1d      --model FNO    --hidden 128 --layers 8 --modes 24
uv run train.py --benchmark swe_2d        --model FNO    --hidden 64  --layers 4 --modes 12
uv run train.py --benchmark allen_cahn_2d --model FNO    --hidden 64  --layers 4 --modes 12
uv run train.py --benchmark ns_hre_2d     --model FNO    --hidden 64  --layers 4 --modes 12

# Check result from a log
grep "^val_l2_rel:\|^peak_vram_mb:" logs/<name>.log
tail -n 50 logs/<name>.log   # if grep is empty (crash)

# ── Autonomous runner ─────────────────────────────────────────────────────────
uv run autorun.py --priority 1 --commit        # run priority-1 pending experiments
uv run autorun.py --priority 2 --commit        # run priority-1 + 2
uv run autorun.py --auto --commit              # run + re-suggest until queue empty
uv run autorun.py --dry-run                    # preview without running

# ── In-process agent loop (Mode B) ───────────────────────────────────────────
uv run agent_loop.py --dry-run                 # plan without writing
uv run agent_loop.py --top 5                   # append top-5 new configs
uv run agent_loop.py --run                     # append + run top-3
uv run agent_loop.py --benchmark burgers_1d    # focus on one benchmark

# ── Analysis ──────────────────────────────────────────────────────────────────
uv run analyze.py                              # full results report
uv run analyze.py --papers                     # results + SOTA gap
uv run auto_suggest.py                         # ranked next-step suggestions
uv run auto_suggest.py --generate              # output ExperimentConfig snippets
uv run auto_suggest.py --gaps                  # SOTA gap table only
uv run paper_registry.py                       # full paper registry
uv run paper_registry.py --gaps                # SOTA gap by paper
uv run paper_registry.py --pending             # unimplemented paper ideas

# ── Bayesian HPO ──────────────────────────────────────────────────────────────
uv run bayesian_hpo.py --benchmark burgers_1d --top 5
uv run bayesian_hpo.py --benchmark burgers_1d --multi --pareto   # multi-objective

# ── Model scaffolding ─────────────────────────────────────────────────────────
uv run model_scaffold.py --stub MyModel --base FNO     # generate stub
uv run model_scaffold.py --register MyModel models/mymodel.py --benchmarks burgers_1d kdv_1d
uv run model_scaffold.py --validate MyModel models/mymodel.py
uv run model_scaffold.py --list                        # show registered models

# ── Visualization ─────────────────────────────────────────────────────────────
uv run viz.py                                  # all plots → figs/
uv run viz.py --mode leaderboard
uv run viz.py --mode arch --benchmark burgers_1d --model FNO

# ── Dashboard ─────────────────────────────────────────────────────────────────
uv run uvicorn app:app --reload --port 8000
# Then open ui/dashboard.html in a browser
# Endpoints: /api/experiments  /api/experiment/{id}  /api/sota  /api/queue
#            /api/logs/{name}  /api/status  /api/pause  /api/resume
#            /api/priority  /api/inject  /api/lineage

# ── Smoke tests ───────────────────────────────────────────────────────────────
uv run benchmarks_ext.py       # verify KdV/Wave/Darcy-fix/NS-fix work
uv run simulations             # smoke test all 4 simulation modules
```

---

## File Roles

### Sacred (never modify)
- `prepare.py` — generates PDE datasets, defines `evaluate_l2_rel`. Modifying this invalidates all results.

### Core experiment files
- `train.py` — training harness. Routes benchmarks → dataloaders, enforces 5-min budget, emits `val_l2_rel` / `peak_vram_mb` / `inspect_id` / `diag_*` metrics to stdout.
- `trainer.py` — `AdamW` (MLX-native), `get_lr_schedule()` (warmup→flat→cosine), `clip_grad_norm()`, `Trainer` class (JIT step).
- `models/` — all model implementations. 14 files, 23 exports. Add new `.py` files here.
- `losses.py` — loss factory: `get_loss_fn(name)`. Available: `l2_rel`, `h1`, `h1_strong`, `h2`, `spectral`, `l1_rel`, `mse`.
- `experiments.py` — 118 `ExperimentConfig` entries (declarative queue). `autorun.py` reads this. Key fields:
  ```python
  ExperimentConfig(
      name="unique_id",          # dedup key — must be unique
      benchmark="burgers_1d",    # see Benchmark Catalog below
      model="FNO",               # see Model Zoo below
      hidden_dim=128,
      n_layers=8,
      n_modes=24,
      budget_s=300,              # training budget in seconds (default 300, use 480 for 2D)
      priority=1,                # 1=first; autorun runs lowest number first
      parent_name="",            # DAG lineage: name of parent experiment
      rationale="Why this?",
  )
  ```
- `results.json` — **source of truth**. DAG experiment tree with parent_id, config, metrics, diag. Never edit by hand.
- `results.tsv` — append-only legacy log, synced from results.json.

### Automation infrastructure
- `autorun.py` — subprocess runner. Deduplicates against `done_names()`, writes log to `logs/<name>.log`, classifies crashes (OOM / NaN-Inf / ImportError / ValueError / Timeout / UnknownError / NoOutput), auto-retries once with halved hidden_dim+n_layers on non-fatal crashes, optionally commits with `--commit`.
- `agent_loop.py` — Mode B orchestrator. Calls `tracker.analyze_lineage()` → `HypothesisEngine` → `BayesianHPO` → appends new `ExperimentConfig` entries to `experiments.py`. Respects `.autorun_pause` sentinel.
- `bayesian_hpo.py` — Gaussian Process surrogate with RBF kernel + Expected Improvement. `BayesianHPO(benchmark, model, objectives=[...])`. Single-obj: `tell(cfg, val)` + `ask()`. Multi-obj: `tell_multi(cfg, {metric: val})` + `pareto_front()`. Seeded from `results.json` via `load_history()`.
- `model_scaffold.py` — gated code generation. `generate_stub(name, base, notes, two_d)` → `.py` template. `ModelGate.validate(name, path)` runs 3 gates: syntax → import → smoke test. `register_and_queue()` copies to `models/`, updates `__init__.py`, adds to `MODEL_REGISTRY`, appends `ExperimentConfig` entries.
- `auto_suggest.py` — suggestion engine. Combines empirical wins, paper ideas, spectral diagnostic feedback, and cross-benchmark transfer (e.g. kdv_1d winner → suggested on wave_1d). Run `uv run auto_suggest.py` for ranked CLI commands.
- `paper_registry.py` — loads `papers/*.yaml` (15 papers), reports SOTA gaps, lists pending implementations.
- `research_plugins.py` — `ModelRegistry` (14 models) + `BenchmarkRegistry` (10 benchmarks). Add new models/benchmarks here; `train.py` routing is automatic.
- `utils.py` — shared constants: `REPO_ROOT`, `LOGS_DIR`, `FIGS_DIR`, `SOTA`, `load_results()`, `best_per_benchmark()`, `done_names()`. Always import from here.

### Experiment tracking & science
- `tracker.py` — DAG lineage engine. `Tracker.log_experiment(benchmark, model, val_l2_rel, ..., config={}, diag={}, parent_name="")`. `analyze_lineage(benchmark=None)` → HP importance (Pearson correlation), model ranking, trend detection. `get_experiment(id)` for detail.
- `diagnostics.py` — `calculate_spectral_bias(pred, target)` → `{low_freq_error, mid_freq_error, high_freq_error, spectral_gap}`. `generate_experiment_comparison()` → `figs/inspect_{id}.png`.
- `hypothesis.py` — data-driven failure analysis. `HypothesisEngine.analyze_benchmark(bm)` reads `results.json` + `papers/*.yaml`. `suggest_intervention(bm, best_val)` returns a concrete config dict (model, hidden_dim, n_layers, n_modes, rationale). Detects: spectral bias → increase modes, shock errors → switch UNO, gradient collapse → RFNO.

### Extended benchmarks
- `benchmarks_ext.py` — KdV, Wave, Darcy-fix, NS-fix using solvers from `prepare.py`.
- `simulations/` — 4 high-fidelity PDE solvers (euler1d, shallow_water, allen_cahn, ns_etdrk4). `ns_hre_2d` first-run generation ~70 min, disk-cached after.

### Visualization
- `viz.py` — `--mode leaderboard|training|data|solver|spectral|concerns|arch`. Saves to `figs/`.

### Web dashboard
- `app.py` — FastAPI backend. Re-loads `results.json` on every request (no stale data). Key endpoints:
  - `GET /api/experiments` — all completed runs
  - `GET /api/experiment/{id}` — detail + inspect_url
  - `GET /api/sota` — SOTA targets + our best + ratio per benchmark
  - `GET /api/queue` — pending experiments with full config
  - `GET /api/logs/{name}?tail=N` — last N lines of a log file
  - `GET /api/status` — VRAM, count, paused flag
  - `POST /api/pause` / `POST /api/resume` — control autorun
  - `POST /api/inject` — inject an experiment into the queue
  - `POST /api/priority` — override priority for a named experiment
- `ui/dashboard.html` — standalone React app. Features: SOTA sidebar with progress bars, Results tab (sortable/searchable/filterable table), Queue tab (all 38 pending), right inspector panel with config grid + training loss curve + log viewer + spectral diagnostics.

### Knowledge base
- `papers/*.yaml` — 15 papers. Each: title, key idea, reported results, our results, suggested experiments, status (pending/implemented), verdict.
- `docs/SOTA.md` — SOTA targets per benchmark.
- `docs/LITERATURE.md` — annotated bibliography.
- `docs/TERMINOLOGY.md` — SciML glossary.
- `program.md` — research protocol (this is the primary agent guide for Mode A).

---

## Model Zoo

| MODEL_TYPE    | Key idea                                      | Status      | Best val_l2_rel               |
|---------------|-----------------------------------------------|-------------|-------------------------------|
| `FNO`         | Global Fourier spectral conv                  | ✓           | 0.1468 (burgers+aug)          |
| `RFNO`        | Pre-LN residual FNO (stable at l≥10)          | ✓           | **0.0020** (kdv_1d) — SOTA   |
| `FNO2d`       | 2D spectral conv (x,y modes)                  | ✓           | darcy_2d                      |
| `FNO_MC`      | Multi-channel [B,N,C]→[B,N,C]                 | ✓           | euler_1d only                 |
| `FFNO`        | Factorized diagonal spectral conv             | ~           | 0.2405 (burgers)              |
| `AFNO`        | Block-diagonal MLP in Fourier + softshrink    | ✗ skip      | 0.50–0.72 — wrong bias        |
| `UNO`         | U-Net encoder-decoder + FNO layers            | ✓           | 0.715 (step-limited)          |
| `WNO`         | Haar wavelet conv (non-periodic BCs)          | ✓           | ✗ wrong for periodic PDEs     |
| `DeepONet`    | Branch + Trunk inner product                  | ✓           | 0.808                         |
| `PODDeepONet` | DeepONet with POD low-rank basis              | ✓           | not benchmarked               |
| `S4NO`        | S4 state-space model for operators            | ✓           | not benchmarked               |
| `GNOT`        | Graph Neural Operator Transformer             | ✓           | not benchmarked               |
| `PINO`        | Physics residual (PDE constraint)             | ✗ broken    | never retry — endpoint-only   |
| `PINN`        | Standard physics-informed NN                  | ✓           | not benchmarked               |

---

## Benchmark Catalog

| Benchmark         | PDE                     | SOTA     | Our best                        | Note                        |
|-------------------|-------------------------|----------|---------------------------------|-----------------------------|
| `burgers_1d`      | 1D viscous Burgers      | 0.0149   | 0.1468 (FNO+aug)                | 10× gap — priority target   |
| `kdv_1d`          | KdV soliton             | ~0.010   | **0.0020** (RFNO)               | 5× better than SOTA ✓       |
| `wave_1d`         | 1D wave u_tt=c²u_xx     | ~0.005   | **0.000992** (FNO h=64 l=4)     | 5× better than SOTA ✓       |
| `darcy_2d_fix`    | 2D Darcy (corrected)    | 0.0108   | 0.1469 (FNO h=32)               | too small — scale up        |
| `ns_2d_fix`       | 2D NS vorticity         | 0.0128   | 0.0152 (FNO)                    | near SOTA — push further    |
| `euler_1d`        | Compressible Euler 1D   | ~0.015   | not run                         | use FNO_MC (3-channel)      |
| `swe_2d`          | 2D Shallow Water        | ~0.002   | not run                         | —                           |
| `allen_cahn_2d`   | Allen-Cahn phase field  | ~0.020   | not run                         | —                           |
| `ns_hre_2d`       | NS 2D Re=1000           | ~0.070   | not run                         | first run ~70 min           |
| `darcy_2d`        | 2D Darcy (broken)       | —        | 0.9986                          | broken solver — never use   |

---

## Loss Library

| `--loss`     | Formula                                       | Best for                  |
|--------------|-----------------------------------------------|---------------------------|
| `l2_rel`     | mean(‖pred−y‖/‖y‖)                            | Default; all PDEs         |
| `h1`         | L2 + 0.01·L2(∂u/∂x)                          | Shocks (Burgers, KdV)     |
| `h1_strong`  | H1 with α=1.0                                | Strong gradient focus     |
| `h2`         | L2 + α·L2(∂²u/∂x²)                           | Smoothness regularization |
| `spectral`   | Frequency-weighted L2                         | High-frequency errors     |
| `l1_rel`     | mean(‖pred−y‖₁/‖y‖₁)                         | Outlier-robust            |
| `mse`        | Mean squared error                            | Debugging/baselines       |

---

## Empirical Findings (from 102 completed experiments)

**Burgers 1D (best: 0.1468 — still 10× from SOTA):**
- m=24 is the sweet spot (m=32 hurts: 0.631; m=16: 0.185)
- h=128 wins over h=64 and h=256 (h=256 is step-time limited)
- l=8 is the depth limit for FNO (l=10: 0.169, l=12: 0.217)
- Augmentation is the single biggest win: +aug → 0.1468
- RFNO does NOT beat FNO on Burgers
- H1 loss barely helps: 0.1559 vs 0.1553 baseline

**KdV 1D (best: 0.0020 — 5× better than SOTA):**
- RFNO h=128 l=8 m=24 is best; pre-LN residual stabilizes soliton dynamics
- Wide models (h=256) hurt due to fewer training steps

**Wave 1D (best: 0.000992 — 5× better than SOTA):**
- Smaller/shallower model wins: FNO h=64 l=4 m=16 → more steps in budget
- "Easy" for Fourier methods — the training-step budget dominates

**Darcy 2D / NS 2D:**
- `darcy_2d` is broken (wrong solver) — always use `darcy_2d_fix`
- darcy_2d_fix: FNO h=32 too small; scale to h=128 m=24
- ns_2d_fix FNO baseline 0.0152 ≈ SOTA 0.0128 — promising, needs tuning

---

## Hard Constraints

- **Never modify `prepare.py`** — it defines the ground-truth metric for all results.
- **No new packages** beyond `pyproject.toml` (mlx, numpy, scipy, matplotlib, pyyaml, fastapi, uvicorn).
- **`ExperimentConfig.name` must be unique** across all entries — it's the dedup key.
- **`results.json` is SSoT** — never hand-edit; all writes go through `tracker.py`.
- **`darcy_2d` is broken** — only use `darcy_2d_fix` and `ns_2d_fix`.
- **PINO is broken** for endpoint-only formulation — never add PINO experiments.
- **Git hygiene** — stage only `train.py`, `models/`, `experiments.py`, `results.tsv`. Never `git add -A`.
- **MLX unified memory** — large models share CPU/GPU memory; watch `peak_vram_mb`.

---

## How to Add Things

### New experiment (fastest path)
```python
# In experiments.py, append to EXPERIMENTS list:
ExperimentConfig(
    name="rfno_burgers_h128_l8_h1",
    benchmark="burgers_1d",
    model="RFNO",
    hidden_dim=128, n_layers=8, n_modes=24,
    budget_s=300,
    priority=1,
    rationale="H1 loss + RFNO on Burgers — hypothesis-driven from spectral bias analysis",
)
# Then: uv run autorun.py --priority 1 --commit
```

### New model (gated path via model_scaffold)
```bash
uv run model_scaffold.py --stub MyModel --base FNO --notes "Add attention after each block"
# Edit models/mymodel.py
uv run model_scaffold.py --register MyModel models/mymodel.py --benchmarks burgers_1d kdv_1d
# Gates: syntax → import → smoke test → auto-registers + queues experiments
```

### New model (manual path)
1. Implement `models/<name>.py` (follow `afno.py` pattern)
2. Export from `models/__init__.py`
3. Add to `research_plugins.py` `ModelRegistry._register_defaults()` — no `train.py` edit needed
4. Add `ExperimentConfig` entries in `experiments.py`

### New benchmark
1. Add solver to `benchmarks_ext.py`; register in `EXT_BENCHMARKS` + `EXT_BENCHMARK_INFO` + `EXT_SOTA`
2. Register in `research_plugins.py` `BenchmarkRegistry`
3. Add `ExperimentConfig` entries; `train.py` routing is automatic

### New paper
1. Create `papers/<id>.yaml` (follow existing format)
2. Set `status: pending`, add `suggested_experiments:` list
3. `uv run paper_registry.py --pending` shows it; `auto_suggest.py` incorporates it
