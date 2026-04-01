# SciML Autoresearch Experiment Protocol

This repository is an Apple Silicon (MLX) platform for autonomous **Scientific Machine Learning (SciML)** research. It supports multiple benchmarks and architectures grounded in SOTA literature.

---

## Knowledge Base
Before starting experiments, consult the `docs/` directory:
- `docs/LITERATURE.md`: Foundational and recent SciML papers.
- `docs/TERMINOLOGY.md`: Standardized terms for SciML.
- `docs/SOTA.md`: Performance targets for each benchmark.

---

## Setup & Benchmarks

1. **Benchmarks Available:**
   - `burgers_1d`: 1D viscous Burgers equation (Advection-Diffusion).
   - `darcy_2d`: 2D steady-state Darcy Flow in porous media.
   - `navier_stokes_2d`: 2D incompressible fluid dynamics (vorticity form).

2. **Generate Data:**
   ```bash
   uv run prepare.py --benchmark all
   ```

3. **Establish Baselines:**
   Run `train.py` with the default settings for each benchmark and record in `results.tsv`.

---

## Experimentation Loop

Each experiment has a **5-minute training budget**.

### What you CAN do
- **Architecture Search:** Modify models in `models/` or add new ones.
- **Multi-Benchmark Optimization:** Improve a model's performance across multiple PDEs.
- **Physics-Informed Training:** Incorporate PDE residuals into the loss function (see `docs/TERMINOLOGY.md`).
- **Hyperparameter Tuning:** Optimize LR, Weight Decay, Batch Size, etc.

### Goal
**Minimize `val_l2_rel`** for the selected benchmark. Competitive results should approach SOTA levels documented in `docs/SOTA.md`.

---

## Research Directions (Advanced)

### 1 · Cross-PDE Generalization
Can a single architecture (e.g., a universal DeepONet) solve both `burgers_1d` and `darcy_2d` with high accuracy?

### 2 · Spectral vs. Real-Space Operators
Compare the efficiency and accuracy of `FNO` (spectral) vs. `DeepONet` (real-space) on non-periodic or irregular benchmarks.

### 3 · Physics-Informed Neural Operators (PINO)
Add a physical consistency loss to an FNO:
$L = L_{data} + \lambda L_{physics}$
Where $L_{physics}$ is the PDE residual computed via automatic differentiation (`mx.grad`).

### 4 · Resolution-Invariant Scaling
Train on `GRID_SIZE=64` but evaluate on higher resolutions (requires updating `evaluate_l2_rel` in a local experiment).

---

## Logging

`results.tsv` schema:
```
commit	benchmark	model	val_l2_rel	memory_gb	status	description
```

Example:
```
a1b2c3d	burgers_1d	FNO	0.043210	2.0	keep	baseline FNO1d
e5f6a7b	darcy_2d	DeepONet	0.021500	2.5	keep	initial DeepONet test
```
