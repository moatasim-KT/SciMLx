# GEMINI.md — SciML AutoResearch (Apple Silicon / MLX)

## What this project does

An autonomous research platform for **Scientific Machine Learning (SciML)** on Apple Silicon. The loop trains neural PDE solvers within a fixed 5-minute budget and logs whether each run improves `val_l2_rel` (relative L2 error — lower is better). You (the external agent) can drive experiments manually or let `agent_loop.py` propose and run them automatically.

**Primary goal:** minimize `val_l2_rel` on fixed validation sets. The ground-truth evaluator lives in `prepare.py` — never modify it.

---

## Setup

```bash
uv sync                      # install all dependencies
uv run prefetch_data.py      # one-time: pre-generate + disk-cache all PDE datasets (~20 min)
uv run prefetch_data.py --skip-slow  # faster: skip ns_hre_2d (~2 min)
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
uv run agent_loop.py --no-hpo       # skip Bayesian HPO, use heuristics only
```

The `--auto` flag wires both modes: when queue exhausts, `autorun.py` calls `agent_loop.py --top 5` and recurses if new experiments are added.

```bash
# Guarded overnight run (stops at 20 experiments or 3 hours)
uv run autorun.py --auto --commit --max-auto-experiments 20 --max-auto-time 10800
```

---

## Key commands reference

```bash
# Single experiment (5-min budget)
uv run train.py --model FNO  --hidden 128 --layers 8  --modes 24
uv run train.py --model RFNO --hidden 128 --layers 8  --modes 24
uv run train.py --benchmark kdv_1d  --model RFNO --hidden 128 --layers 8 --modes 24
uv run train.py --benchmark wave_1d --model FNO  --hidden 64  --layers 4 --modes 16
# 2D benchmarks — use small models + extended budget
uv run train.py --benchmark darcy_2d --model FNO  --hidden 32 --layers 4 --modes 12 --budget 480
uv run train.py --benchmark ns_2d_fix    --model FNO  --hidden 32 --layers 4 --modes 8  --budget 600

# Run the pending queue
uv run autorun.py --priority 1 --commit          # run all priority-1 pending
uv run autorun.py --auto --commit                # run + re-suggest until queue empty
uv run autorun.py --auto --commit --max-auto-experiments 20 --max-auto-time 10800
uv run autorun.py --dry-run                      # preview without running

# Kill a running experiment (terminates within 2s)
curl -X POST http://localhost:8000/api/kill/<name>

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
| `train.py` | Training harness. Routes benchmark → dataloader, enforces 5-min budget, emits `val_l2_rel`. `WARMDOWN_RATIO=0.2` (cosine decay starts at 80%). |
| `trainer.py` | `AdamW`, `get_lr_schedule()` (warmup→flat→cosine), `Trainer` (JIT step). Writes rolling loss history to `.vram_telemetry`. |
| `models/` | 13 model files, 28+ exports (FNO, RFNO, SSNO, AFNO, FFNO, UNO, WNO, DeepONet, S4NO, GNOT, PINN…). |
| `losses.py` | `get_loss_fn(name)` — l2_rel, h1, h1_strong, h2, spectral, l1_rel, mse. |
| `experiments.py` | 205+ `ExperimentConfig` entries (declarative queue). Fields: `name`, `benchmark`, `model`, `hidden_dim`, `n_layers`, `n_modes`, `budget_s` (default 300s, 480s for 2D), `priority`, `parent_name`, `rationale`. |
| `results.json` | **SSoT.** DAG experiment tree. Never hand-edit — use `tracker.py`. IDs include UUID suffix to avoid collisions. |
| `results.tsv` | Append-only legacy log, synced from results.json. |
| `autorun.py` | Runs pending configs as subprocesses. Deduplicates, classifies crashes (OOM/NaN-Inf/ImportError/…), **multi-fix** retry (OOM+NaN → both `batch//2` AND `lr//10`), sentinel kill (`.kill_{name}`), `--max-auto-experiments`, `--max-auto-time` guards. `--commit` git-commits each kept result. |
| `agent_loop.py` | Mode B orchestrator. HypothesisEngine + BayesianHPO → new ExperimentConfig entries. `--no-hpo` skips HPO. |
| `bayesian_hpo.py` | GP surrogate (RBF kernel + EI). `BayesianHPO(bm, model, objectives=[...])`. Single-obj (`tell`/`ask`) or multi-obj (`tell_multi`/`pareto_front`). Seeds from results.json. |
| `model_scaffold.py` | Gated code generation: 3-gate validation (syntax → import → smoke) before registering new models. |
| `tracker.py` | `log_experiment(…, config={}, diag={}, parent_name="")`. UUID-stable IDs. `analyze_lineage()` for HP importance + trends. |
| `hypothesis.py` | `HypothesisEngine.analyze_benchmark(bm)` + `suggest_intervention(bm, best_val)` → concrete config dict. |
| `diagnostics.py` | `calculate_spectral_bias(pred, target)` → low/mid/high-freq error. PNG to `figs/inspect_{id}.png`. |
| `auto_suggest.py` | Dynamic KNOWN_WINS (reads results.json at import). Ranked suggestions from empirical wins + paper ideas + spectral feedback + cross-benchmark transfer. |
| `research_plugins.py` | `ModelRegistry` (28+ models) + `BenchmarkRegistry` (10 benchmarks). Add models/benchmarks here. |
| `utils.py` | `REPO_ROOT`, `LOGS_DIR`, `FIGS_DIR`, `SOTA`, `load_results()`, `best_per_benchmark()`, `done_names()`. |
| `app.py` | FastAPI. Re-loads results.json on every request. Endpoints: `/api/experiments`, `/api/sota`, `/api/queue`, `/api/logs/{name}`, `/api/status` (includes live `progress`, `loss`, `loss_history`), `/api/pause`, `/api/resume`, `/api/inject`, `/api/kill/{name}`. |
| `ui/dashboard.html` | Standalone React dashboard. SOTA sidebar, Results + Queue + **Lineage DAG** tabs, right inspector with **parent comparison**, **SparkLine** in active strip, **Kill** button. 3s poll interval. |
| `benchmarks_ext.py` | KdV, Wave, Darcy-fix, NS-fix. |
| `simulations/` | 4 high-fidelity solvers: euler1d, shallow_water, allen_cahn, ns_etdrk4. |
| `papers/*.yaml` | 20 papers with key idea, reported results, our results, suggested experiments. |
| `docs/SOTA.md` | SOTA targets per benchmark. |
| `program.md` | Detailed research protocol for Mode A agents. |

---

## Model Zoo

| Model | Key idea | Status | Best |
|-------|----------|--------|------|
| `FNO` | Global Fourier spectral conv | ✓ | 0.1468 (burgers+aug) |
| `RFNO` | Pre-LN residual FNO (stable at depth) | ✓ | **0.0020** (kdv) — beats SOTA |
| `FNO2d` | 2D spectral conv | ✓ | darcy_2d |
| `FNO_MC` | Multi-channel [B,N,C]→[B,N,C] | ✓ | euler_1d |
| `FFNO` | Factorized diagonal spectral conv | ~ | 0.24 (burgers) |
| `AFNO` | Block-diag MLP in Fourier | ✗ skip | 0.50–0.72 — wrong bias |
| `UNO` | U-Net encoder-decoder + FNO | ✓ | 0.715 (step-limited) |
| `WNO` | Haar wavelet conv | ✓ | wrong for periodic PDEs |
| `DeepONet` | Branch + Trunk inner product | ✓ | 0.808 |
| `S4NO` | S4 state-space model | ✓ | not benchmarked |
| `SSNO` | Adaptive S4D damping + spectral conv dual-branch | ✓ | not benchmarked yet |
| `GNOT` | Graph Neural Operator Transformer | ✓ | not benchmarked |
| `PINO` | Physics residual loss | ✗ broken | **never retry** |
| `PINN` | Standard physics-informed NN | ✓ | not benchmarked |

---

## Benchmark Catalog

| Benchmark | PDE | SOTA | Our best | Notes |
|-----------|-----|------|----------|-------|
| `burgers_1d` | 1D viscous Burgers | 0.0149 | 0.1468 | **HIGH** (9.8× gap) |
| `kdv_1d` | KdV soliton | 0.010 | **0.0020** ✓ | 5× better than SOTA |
| `wave_1d` | 1D wave | 0.005 | **0.000992** ✓ | 5× better than SOTA |
| `euler_1d` | Compressible Euler | ~0.015 | **0.002413** ✓ | 6.2× better than SOTA |
| `darcy_2d` | 2D Darcy | 0.0108 | 0.1041 | HIGH — need h≤32 l≤4 |
| `ns_2d_fix` | 2D NS vorticity | 0.0128 | 0.01428 | MEDIUM — 1.12× gap, budget=600 |
| `swe_2d` | 2D Shallow Water | ~0.002 | 0.0107 | 5.4× gap |
| `allen_cahn_2d` | Allen-Cahn | ~0.020 | 0.0628 | 3.1× gap |
| `ns_hre_2d` | NS Re=1000 | ~0.070 | not run | ~70 min first gen |
| `darcy_2d` | 2D Darcy (legacy) | — | 0.9986 | **REMOVED** |

---

## Loss functions

`--loss l2_rel` (default) · `h1` · `h1_strong` · `h2` · `spectral` · `l1_rel` · `mse`

H1 loss (α=0.01) marginally helps on Burgers; default l2_rel is fine for KdV and Wave.

---

## Empirical findings (184 experiments)

**Burgers (best 0.1468 — still 9.8× from SOTA 0.0149):**
- Optimal config: FNO h=128 l=8 m=24 + augmentation
- m=32 hurts (0.631); l>8 reduces training steps and hurts
- RFNO does not beat FNO here

**KdV (best 0.0020 — 5× better than SOTA):**
- RFNO h=128 l=8 m=24 is best; pre-LN residual helps soliton dynamics
- Wide models (h=256) hurt — fewer training steps

**Wave (best 0.000992 — 5× better than SOTA):**
- FNO h=64 l=4 m=16: smaller model → more steps within budget

**Euler 1D (best 0.002413 — 6.2× better than SOTA):**
- FNO h=64 l=4 highly efficient; FNO_MC didn't outperform standard FNO

**NS 2D (best 0.01428 — 1.12× from SOTA 0.0128):**
- Extended budget (600s) beats 480s by 5.8% — same config, more steps
- n_modes=8 beats n_modes=12; H1 loss hurts

**2D benchmarks (critical):**
- Models with h≥64 or l≥8 crash on all 2D benchmarks (OOM/broadcast errors on Apple Silicon)
- Safe config: h≤32, l≤4, m≤8, budget_s=480 (600s for ns_2d_fix)
- **RFNO is 1D-only** — crashes on all 2D benchmarks; never add RFNO to 2D experiments
- **SSNO is unstable** at h=128 l=8 (val=80.5 on Burgers); needs smaller config or lr tuning

---

## Hard constraints

1. **Never modify `prepare.py`** — it defines the ground-truth metric.
2. **No new packages** — only what's in `pyproject.toml`.
3. **`ExperimentConfig.name` must be unique** — it's the dedup key.
4. **Never hand-edit `results.json`** — use `tracker.py` API.
5. **darcy_2d is now consolidated**: Legacy flawed version removed; corrected Richardson solver promoted to primary name.
6. **PINO is broken** — endpoint-only formulation always fails; never add PINO entries.
7. **2D model size**: h≤32, l≤4, m≤8, budget_s=480 (ns_2d_fix: 600) — larger configs crash.
8. **RFNO is 1D-only** — crashes on all 2D benchmarks; never add RFNO to 2D experiments.
9. **SSNO is unstable** at h≥128 — start with h≤64 l≤4 and lower lr if exploring.

---

## Current research priorities

1. **Close the Burgers gap** — best is 0.1468, SOTA is 0.0149 (9.8× gap). SSNO claims 0.007 in the paper.
2. **Scale up darcy_2d** — current best 0.1041 from FNO; try h=32 l=4 m=12 480s (small models only).
3. **Push ns_2d_fix below SOTA** — 0.0152 vs 0.0128; try RFNO h=32 m=8 or H1 loss.
4. **Benchmark unrun models** — SSNO on burgers_1d (paper target: 0.007).
5. **Run simulation benchmarks** — swe_2d, allen_cahn_2d need more experiments.

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
