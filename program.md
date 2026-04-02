# SciML Autoresearch Experiment Protocol

This repository is an Apple Silicon (MLX) platform for autonomous **Scientific
Machine Learning (SciML)** research.  An AI agent modifies `train.py`, runs a
5-minute experiment, and keeps or discards each result based on `val_l2_rel`.

---

## Knowledge Base

Before starting experiments, consult:
- `docs/LITERATURE.md` – foundational + recent SciML papers
- `docs/TERMINOLOGY.md` – standardised terms
- `docs/SOTA.md`        – performance targets per benchmark

---

## Setup & Benchmarks

| Benchmark           | PDE                      | Metric target (SOTA) |
|---------------------|--------------------------|----------------------|
| `burgers_1d`        | 1D viscous Burgers        | ≤ 0.05 rel-L2        |
| `darcy_2d`          | 2D steady Darcy flow      | ≤ 0.02 rel-L2        |
| `navier_stokes_2d`  | 2D incompressible NS      | ≤ 0.02 rel-L2        |

```bash
uv run prepare.py          # one-time: generates + caches val set
uv run train.py            # run experiment (5-min budget)
uv run train.py > run.log 2>&1   # redirect for autonomous loops
grep "^val_l2_rel:" run.log
```

---

## Experiment Loop

Each experiment has a **5-minute training budget** (`TIME_BUDGET = 300`).

### Permitted edits
- `train.py`   – hyperparameters, architecture selection, loss design
- `models/*.py` – add or modify model classes
- `results.tsv` – append result rows

### Forbidden edits
- `prepare.py` – defines the ground-truth metric; must not be changed

### Protocol

1. Create branch `autoresearch/<tag>`
2. Run baseline once to confirm hardware speed
3. Edit `train.py`, commit, run, read `val_l2_rel`
4. **Keep** if improved: `git add train.py results.tsv && git commit --amend`
5. **Discard** if not: log as `discard`, then `git reset --hard <last-kept>`
6. Never use `git add -A`

---

## Architecture Zoo

All models share the same interface: `model(u0)` where `u0: [B, N]` (1D) or
`[B, N, N]` (2D).  Switch via `MODEL_TYPE` constant or `--model` flag.

| MODEL_TYPE     | Key idea                              | Best for               |
|----------------|---------------------------------------|------------------------|
| `FNO`          | Global Fourier spectral conv          | Periodic, smooth PDEs  |
| `UNO`          | U-Net encoder-decoder + FNO layers    | Multi-scale phenomena  |
| `WNO`          | Haar wavelet conv (multi-level)       | Non-periodic, shocks   |
| `DeepONet`     | Branch + Trunk inner product          | Irregular geometries   |
| `PODDeepONet`  | Shared POD basis + branch coeff net   | Low-dim solution space |

---

## Research Directions (8 Advanced Topics)

Work through these in order; earlier ones are higher expected yield.

---

### Direction 1 · Architecture Depth & Width Sweep

**Hypothesis:** The current baseline (FNO, 4 layers, hidden=64) may under-use
the 5-minute budget on Apple Silicon.  Wider/deeper models could converge to
better solutions.

**What to try:**
```python
# Vary one at a time; record val_l2_rel for each
HIDDEN_DIM in [32, 64, 128, 256]
N_LAYERS   in [2, 4, 6, 8]
N_MODES    in [8, 12, 16, 24, 32]  # ≤ GRID_SIZE//2 = 32
```

**Expected outcome:** hidden=128, layers=6 should outperform the hidden=64,
layers=4 baseline without exceeding memory.

**Stop condition:** if step time exceeds 200ms (model too large for budget).

---

### Direction 2 · U-shaped Neural Operator (UNO)

**Hypothesis:** UNO's encoder-decoder structure captures multi-scale features
that a flat FNO misses, especially important for Burgers' equation where the
shock lives at a different scale from the smooth background.

**What to try:**
```python
MODEL_TYPE = "UNO"
N_MODES    = 16
HIDDEN_DIM = 64   # per-level; bottleneck uses 4×HIDDEN_DIM
N_LAYERS   = 2    # FNO blocks per level (6 total across 3 levels)
```

**Expected outcome:** 10–30% improvement over FNO baseline on `burgers_1d`.
Reference: Rahman et al. (2022) report ~20% error reduction vs flat FNO.

---

### Direction 3 · Wavelet Neural Operator (WNO)

**Hypothesis:** Haar wavelets have compact support and handle non-periodic
boundary conditions better than global Fourier modes.  Better suited for
`darcy_2d` (inhomogeneous domain).

**What to try:**
```python
MODEL_TYPE = "WNO"
N_LEVELS   = 3    # 3 decomposition levels → N/8 = 8-point approximation
HIDDEN_DIM = 64
N_LAYERS   = 4
```

**Expected outcome:** Comparable to FNO on `burgers_1d`; meaningfully better
on non-periodic problems.

---

### Direction 4 · Physics-Informed Neural Operator (PINO)

**Hypothesis:** Adding the Burgers PDE residual as an auxiliary loss encourages
the model to learn physically consistent solutions, especially in regions poorly
covered by training data.

**What to try:**
```python
MODEL_TYPE  = "FNO"   # or UNO
PINO_LAMBDA = 0.01    # start small; try 0.001, 0.01, 0.1
HIDDEN_DIM  = 64
```

The `burgers_residual()` function in `train.py` computes `u·∂u/∂x − ν·∂²u/∂x²`
spectrally.  The combined loss is:
```
L = L_data + λ · mean(residual²)
```

**Key pitfall:** λ too large → physics loss dominates, data loss increases.
Binary-search λ: start at 0.01, halve/double based on whether val_l2_rel
improves.

**Expected outcome:** 5–15% improvement over data-only baseline.
Reference: Li et al. (2021) PINO paper reports consistent improvements.

---

### Direction 5 · Learning Rate & Schedule Tuning

**Hypothesis:** The default LR=1e-3 and 40% cosine warmdown may not be optimal
for the fixed 5-minute budget.

**What to try:**
```python
LR             in [3e-4, 1e-3, 3e-3]
WARMUP_RATIO   in [0.02, 0.05, 0.10]
WARMDOWN_RATIO in [0.20, 0.40, 0.60]
FINAL_LR_FRAC  in [0.001, 0.01, 0.1]
```

**Note:** higher LR can diverge without GRAD_CLIP.  Keep `GRAD_CLIP = 1.0`.

**Expected outcome:** 5–10% improvement from better schedule.

---

### Direction 6 · Gradient Clipping Sensitivity

**Hypothesis:** The default GRAD_CLIP=1.0 was chosen conservatively.  Looser
or tighter clipping affects convergence speed and stability.

**What to try:**
```python
GRAD_CLIP in [0.1, 0.5, 1.0, 5.0, 0.0]  # 0 = disabled
```

Check `max_grad_norm` in the log to see how often clipping fires.  If
`max_grad_norm` is consistently < 0.5, clipping is unnecessary; try disabling.

---

### Direction 7 · Sobolev / H1 Loss Function

**Hypothesis:** Standard L2 loss treats all spatial frequencies equally.  A
Sobolev H1 norm penalises high-frequency errors more, which can improve
smoothness of the predicted solution and overall accuracy.

**What to try** (implement in `loss_fn`):
```python
# H1 loss: L2 of (pred - y) + α * L2 of (∂pred/∂x - ∂y/∂x)
def h1_loss(pred, y, alpha=0.1):
    N = pred.shape[-1]
    k = mx.arange(N // 2 + 1, dtype=mx.float32)
    def grad_fft(u):
        u_ft = mx.fft.rfft(u, axis=-1)
        return mx.fft.irfft(mx.complex(-u_ft.imag * k, u_ft.real * k),
                            n=N, axis=-1)
    diff   = pred - y
    grad_d = grad_fft(diff)
    axes   = tuple(range(1, y.ndim))
    return mx.mean(mx.mean(diff ** 2, axis=axes) +
                   alpha * mx.mean(grad_d ** 2, axis=axes))
```

Try α in [0.01, 0.1, 1.0].

---

### Direction 8 · DeepONet Architecture Search

**Hypothesis:** The current DeepONet uses 4 hidden layers with LayerNorm.
Changing width, depth, or the branch/trunk asymmetry could unlock better
performance on this benchmark.

**What to try:**
```python
MODEL_TYPE = "DeepONet"
HIDDEN_DIM in [64, 128, 256]    # branch/trunk width
N_LAYERS   in [3, 4, 6]         # layers in each net
```

Also try `PODDeepONet`:
```python
MODEL_TYPE = "PODDeepONet"
HIDDEN_DIM = 64                 # also controls n_basis
```

**Expected target:** val_l2_rel < 0.15 (vs current 0.808).  SOTA for DeepONet
on Burgers is 0.05–0.10.

---

### Direction 9 · Cross-PDE Generalisation

**Hypothesis:** A single model trained simultaneously on multiple PDEs can
learn shared structure (e.g., advection, diffusion) and achieve better
generalisation than benchmark-specific models.

**How to implement** (requires changes to `loss_fn` and data loading):
```python
# Interleave batches from two benchmarks:
loader_b = make_dataloader("burgers_1d",  "train", BATCH_SIZE // 2)
loader_d = make_dataloader("darcy_2d",    "train", BATCH_SIZE // 2)
# Concatenate along batch dim; use same FNO1d/FNO2d for respective shapes
# or embed both into a shared latent space
```

---

### Direction 10 · Resolution-Invariant Evaluation

**Hypothesis:** FNO is theoretically resolution-invariant.  Verify by training
at GRID_SIZE=64 and manually evaluating at N=128 or N=256.

**How to implement:**
```python
# After training, create a new eval set at higher resolution:
from prepare import solve_burgers_batch
import numpy as np
# Generate high-res ICs and solutions, evaluate using trained model
# (The model's FFT layers handle variable N automatically)
```

This tests whether learned operators truly generalise across resolutions.

---

## Logging

`results.tsv` schema:
```
commit  benchmark  model  val_l2_rel  memory_gb  status  description
```

Statuses: `keep` | `discard` | `crash`

Example:
```
a1b2c3d  burgers_1d  FNO   0.043210  0.1  keep     FNO1d hidden=128 layers=6
e5f6a7b  burgers_1d  UNO   0.038500  0.2  keep     UNO1d hidden=64 n_layers=2
f9g0h1i  burgers_1d  WNO   0.051200  0.1  discard  WNO1d n_levels=3 - worse
```

---

## Useful one-liners

```bash
# Quick result check
grep "^val_l2_rel:" run.log

# Diagnose a crash
tail -n 50 run.log

# Show current best
sort -t$'\t' -k4 -n results.tsv | head -5

# Run with explicit model/hyperparams via CLI
uv run train.py --model UNO --hidden 128 --layers 2 > run.log 2>&1
uv run train.py --model WNO --levels 3 --hidden 64  > run.log 2>&1
uv run train.py --model FNO --pino_lambda 0.01      > run.log 2>&1
```
