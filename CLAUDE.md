# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) or any AI agent when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# One-time PDE data generation (synthesizes + caches validation set)
uv run prepare.py

# Run one experiment (~6-8 minutes on Apple Silicon)
uv run train.py --model FNO --hidden 128 --layers 8 --modes 24

# Run with new model types
uv run train.py --model RFNO --hidden 128 --layers 10 --modes 24
uv run train.py --model AFNO --hidden 128 --layers 8  --modes 24
uv run train.py --model FNO  --hidden 128 --layers 8  --modes 24 --loss h1

# New benchmarks (KdV, Wave)
uv run train.py --benchmark kdv_1d  --model FNO --hidden 128 --layers 8 --modes 24
uv run train.py --benchmark wave_1d --model FNO --hidden 64  --layers 4 --modes 24

# Check results after a run
grep "^val_l2_rel:\|^peak_vram_mb:" run.log

# Tail the log if grep returns empty (crash diagnosis)
tail -n 50 run.log

# Autonomous experiment runner
uv run autorun.py --priority 2 --commit          # run pending experiments
uv run autorun.py --auto --commit                # fully autonomous: run + suggest loop
uv run autorun.py --dry-run                      # preview queue without running

# Analysis and next-step suggestions
uv run analyze.py                                # results report
uv run analyze.py --papers                       # results + SOTA gap from papers
uv run auto_suggest.py                           # autonomous next-experiment suggestions
uv run auto_suggest.py --generate                # output ExperimentConfig code snippets
uv run auto_suggest.py --gaps                    # SOTA gap analysis

# Paper registry
uv run paper_registry.py                         # full paper registry report
uv run paper_registry.py --gaps                  # SOTA gap table
uv run paper_registry.py --pending               # list unimplemented papers

# New benchmark smoke test
uv run benchmarks_ext.py                         # verify KdV and Wave benchmarks work
```

## Architecture

This is an Apple Silicon (MLX) platform for autonomous **Scientific Machine Learning (SciML)** research. The autonomous loop trains a neural PDE solver on a fixed 5-minute budget and logs whether each experiment improved `val_l2_rel` (relative L2 error, lower is better).

### File Roles

**Core (read-only):**
- `prepare.py` — **never modify**. Generates benchmark datasets, defines `evaluate_l2_rel`. Exports: `GRID_SIZE`, `TIME_BUDGET`, `make_dataloader`, `solve_burgers_batch`, `solve_wave_batch`, `solve_kdv_batch`.

**Core (editable):**
- `train.py` — main experiment script. Add model types, change loss, tune hyperparams.
- `models/` — model implementations (add new `.py` files here).
- `experiments.py` — declarative experiment queue. Add `ExperimentConfig` entries.
- `results.tsv` — append-only experiment log.

**Automation infrastructure:**
- `autorun.py` — subprocess runner with dedup, baseline tracking, git integration. `--auto` flag runs indefinitely.
- `analyze.py` — results analysis, convergence plots, hyperparameter correlations.
- `auto_suggest.py` — **autonomous next-step suggester**. Reads results + papers → generates ranked experiment suggestions with CLI commands.
- `paper_registry.py` — loads `papers/*.yaml`, reports SOTA gaps, suggests pending paper implementations.

**Extended benchmarks:**
- `benchmarks_ext.py` — KdV and Wave equation benchmarks using existing solvers from `prepare.py`. Same dataloader/eval interface.

**Loss library:**
- `losses.py` — modular loss functions: `l2_rel`, `h1`, `h1_strong`, `spectral`, `l1_rel`. Use via `--loss` CLI flag.

**Knowledge base:**
- `papers/*.yaml` — one YAML per paper: title, key idea, reported results, our results, suggested experiments, verdict.
- `docs/SOTA.md` — SOTA targets per benchmark.
- `docs/LITERATURE.md` — annotated paper bibliography.
- `docs/TERMINOLOGY.md` — SciML glossary.
- `program.md` — research protocol for agent-driven loops.

### Model Zoo

| MODEL_TYPE   | Key idea                                      | Status      | Best val_l2_rel |
|--------------|-----------------------------------------------|-------------|-----------------|
| `FNO`        | Global Fourier spectral conv (linear)         | ✓ working   | 0.1553 (burgers) |
| `RFNO`       | Pre-LN residual FNO (unlocks l≥10)            | ✓ working   | running (s6)    |
| `AFNO`       | Block-diagonal MLP in Fourier + softshrink    | ✓ new       | not yet run     |
| `UNO`        | U-shaped encoder-decoder with FNO layers      | ✓ working   | 0.715 (step-limited) |
| `WNO`        | Haar wavelet conv (non-periodic BCs)          | ✓ working   | 0.896 (wrong inductive bias for periodic Burgers) |
| `DeepONet`   | Branch + Trunk inner product                  | ✓ working   | 0.808 (high-dim branch input) |

### Loss Library

| LOSS_TYPE   | Formula                                   | Best for                    |
|-------------|-------------------------------------------|-----------------------------|
| `l2_rel`    | mean(‖pred−y‖/‖y‖)                        | Default; all PDEs           |
| `h1`        | L2 + α·L2(∂pred/∂x − ∂y/∂x)              | Shock fronts (Burgers, KdV) |
| `h1_strong` | H1 with α=1.0                             | Strong gradient focus       |
| `spectral`  | Frequency-weighted L2                     | High-frequency errors       |
| `l1_rel`    | mean(‖pred−y‖₁/‖y‖₁)                     | Outlier-robust              |

### Benchmark Catalog

| Benchmark          | PDE                    | Solver     | SOTA     | Our best |
|--------------------|------------------------|------------|----------|----------|
| `burgers_1d`       | 1D viscous Burgers     | IMEX-Euler | 0.0149   | 0.1553   |
| `darcy_2d`         | 2D steady Darcy        | Spectral   | 0.0108   | 0.9986   |
| `kdv_1d`           | KdV soliton            | ETDRK4     | ~0.010   | not run  |
| `wave_1d`          | 1D wave u_tt=c²u_xx    | Störmer-V  | ~0.005   | not run  |
| `navier_stokes_2d` | 2D incompressible NS   | Spectral   | 0.0128   | crashes  |

### Empirical Findings (Burgers 1D)

From 60+ experiments:
- **m=24** is the mode sweet spot (m=16: 0.185, m=22: 0.164, m=24: 0.155, m=32: 0.631)
- **h=128** wins over h=64 and h=256 (step-time limited at h=256)
- **l=8** is FNO depth limit (l=10: 0.169, l=12: 0.217 — fewer training steps)
- **lr=1e-3** optimal; bs=32 better than bs=16
- **PINO is broken** for endpoint-only formulation — never retry
- **WNO is wrong** for periodic Burgers — only try on darcy_2d or kdv_1d
- **Pre-LN residuals (RFNO)** should unlock l>8 — session 6 running

### Key Constraints

- `prepare.py` must not be modified — it defines the ground-truth metric.
- No new packages beyond `pyproject.toml` (mlx, numpy, scipy, matplotlib, pyyaml optional).
- All experiments compare against a hardware-local baseline (Apple Silicon throughput differs from CUDA).
- MLX uses unified memory; large models share CPU/GPU memory.
- `ExperimentConfig.name` must be unique — it's the dedup key against results.tsv.

## Autonomous Research Loop

### Fully autonomous (hands-off)
```bash
uv run autorun.py --auto --priority 2 --commit 2>&1 | tee logs/autorun_auto.log
```
This runs all priority≤2 experiments, then prints `auto_suggest.py` recommendations, and loops until the queue is exhausted.

### Semi-autonomous (agent-guided)
1. `uv run auto_suggest.py` — see what to try next
2. Edit `experiments.py` with new configs (use `--generate` output as template)
3. `uv run autorun.py --priority N --commit` — run the new queue
4. `uv run analyze.py --papers` — review results vs literature

### Manual entry
Add any `ExperimentConfig` to `EXPERIMENTS` in `experiments.py` and it will be picked up by `autorun.py`. Every field has a default; minimal entry:
```python
ExperimentConfig(
    name="my_exp_name",        # unique, used for dedup
    benchmark="burgers_1d",
    model="FNO",               # or RFNO, AFNO, UNO, WNO, DeepONet
    hidden_dim=128,
    n_layers=8,
    n_modes=24,
    priority=1,                # 1 = run first
    rationale="Why this?",
)
```

### Adding a new paper
1. Create `papers/<id>.yaml` following the template in existing files
2. Set `status: pending` for unimplemented ideas
3. Add `suggested_experiments:` list with name, rationale, expected
4. `uv run paper_registry.py --pending` will show it in suggestions
5. `uv run auto_suggest.py` will incorporate it into recommendations

### Adding a new benchmark
1. Add solver to `benchmarks_ext.py` (or reuse existing `solve_*_batch` functions)
2. Register in `EXT_BENCHMARKS` set and `EXT_BENCHMARK_INFO` dict
3. Add `EXT_SOTA` entry with paper-reported SOTA
4. Add `ExperimentConfig` entries with `benchmark="new_benchmark_name"`
5. The `train.py` routing (`if BENCHMARK in EXT_BENCHMARKS`) handles it automatically

### Adding a new model
1. Implement in `models/<name>.py` (see `afno.py` for the pattern)
2. Export from `models/__init__.py`
3. Import in `train.py` and add to the model factory (`elif MODEL_TYPE == "NAME":`)
4. Add to `_parse_args()` choices list
5. Add `short()` method support in `experiments.py`
6. Add `ExperimentConfig` entries and a `papers/<name>.yaml`
