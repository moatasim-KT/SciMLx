# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) or any AI agent when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# One-time PDE data generation (synthesizes + caches validation set)
uv run prepare.py

# Run one experiment (~6-8 minutes on Apple Silicon)
uv run train.py

# Run with output redirected (required during autonomous experiment loops)
uv run train.py > run.log 2>&1

# Check results after a run
grep "^val_l2_rel:\|^peak_vram_mb:" run.log

# Tail the log if grep returns empty (crash diagnosis)
tail -n 50 run.log
```

## Architecture

This is an Apple Silicon (MLX) port of Karpathy's autoresearch, repurposed for
**Scientific Machine Learning (SciML)**.  The autonomous research loop trains a
neural PDE solver on a fixed 5-minute budget and logs whether each experiment
improved `val_l2_rel` (relative L2 error, lower is better).

**File roles:**
- `prepare.py` — **read-only**. Generates the 1D Burgers benchmark dataset via a
  pseudo-spectral IMEX solver, caches the fixed validation set, and defines
  `evaluate_l2_rel` — the ground-truth metric. Also exports `GRID_SIZE`,
  `TIME_BUDGET`, `make_dataloader`, `solve_burgers_batch`, `solve_wave_batch`,
  and `solve_kdv_batch` for use by `train.py`.
- `train.py` — **the only file you edit**. Defines the neural operator (baseline:
  FNO-1D), the optimiser, and the training loop. Hyperparameters are module-level
  constants (no CLI flags).
- `program.md` — the autonomous experiment protocol for agent-driven research loops.
- `results.tsv` — tab-separated experiment log (`commit`, `val_l2_rel`,
  `memory_gb`, `status`, `description`).

**Baseline model (`train.py`):**
- `SpectralConv1d` — 1-D Fourier spectral convolution. Multiplies the `N_MODES`
  lowest Fourier coefficients by learned complex weights (stored as `wr`, `wi`).
  Uses `mx.fft.rfft` / `mx.fft.irfft` and `mx.view` to pack real/imag pairs into
  complex64 for the inverse FFT.
- `FNOBlock` — spectral conv + pointwise linear + GELU.
- `FNO1d` — lifts (u₀ ‖ grid) → hidden_dim, stacks N_LAYERS FNOBlocks, projects to output.
- `AdamW` — hand-rolled AdamW with runtime LR control (no mlx.optimizers dependency).
- Loss: per-sample relative L2, averaged over the batch.

**Eight SciML research directions** (see `program.md` for details):
1. Fluid Mechanics / Navier–Stokes Solvers
2. In-Context PDE Solving
3. Learned Preconditioners
4. Uncertainty Quantification via Architecture
5. Latent Dynamics Models (Neural ODEs / CDEs)
6. Multi-Scale / Hierarchical PDE Solvers
7. Physics-Informed Neural Network (PINN) Optimizer Search
8. Neural Operator Architecture Search

**Key constraints:**
- `prepare.py` must not be modified — it defines the ground-truth metric.
- No new packages beyond `pyproject.toml` (mlx, numpy, scipy, matplotlib).
- All experiments compare against a hardware-local baseline (Apple Silicon
  throughput differs from CUDA).
- MLX uses unified memory; large models share CPU/GPU memory.

## Experiment loop (for autonomous agent use)

See `program.md` for the full protocol. In brief:

1. Create branch `autoresearch/<tag>`
2. Run `uv run train.py` once to establish the hardware-local baseline
3. Edit `train.py`, commit, run, read `val_l2_rel`
4. If improved: amend commit to include `results.tsv` and keep
5. If not improved: log as `discard`, then `git reset --hard <last-kept-commit>`
6. **Never use `git add -A`** — stage only `train.py` and `results.tsv` explicitly
