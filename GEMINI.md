# GEMINI.md - autoresearch-sciml-mlx

## Project Overview
This project is an Apple Silicon (MLX) port of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), specifically repurposed for **Scientific Machine Learning (SciML)**. It implements an autonomous research loop where an AI agent iteratively modifies a training script (`train.py`) to optimize neural PDE solvers (like FNO-1D) within a fixed 5-minute training budget.

The primary goal is to minimize the **Relative L2 Error (`val_l2_rel`)** on a fixed validation set of PDE solutions.

### Core Technologies
- **Framework:** [MLX](https://github.com/ml-explore/mlx) (Natively optimized for Apple Silicon)
- **Language:** Python 3.10+
- **Dependency Management:** [uv](https://docs.astral.sh/uv/)
- **Libraries:** NumPy, SciPy, Matplotlib

## Building and Running

### Setup
```bash
# Install dependencies
uv sync

# One-time data synthesis for PDEs (caches to ~/.cache/sciml_autoresearch/)
uv run prepare.py
```

### Running Experiments
```bash
# Run a single 5-minute training experiment
uv run train.py

# Run with output redirected for logging (standard practice for agent loops)
uv run train.py > run.log 2>&1
```

### Analysis
```bash
# Extract key metrics from the log
grep "^val_l2_rel:\|^peak_vram_mb:" run.log

# Diagnose crashes
tail -n 50 run.log
```

## Project Architecture

- **`prepare.py` (READ-ONLY):** The fixed research harness. It handles data generation (Burgers, Wave, KdV equations) and provides the ground-truth evaluation metric (`evaluate_l2_rel`).
- **`train.py` (MUTABLE):** The core file for experimentation. Contains the model architecture (baseline: FNO-1D), optimizer (custom AdamW), and the training loop. This is the only file the AI agent should edit.
- **`program.md`:** Detailed autonomous experiment protocol and research directions.
- **`results.tsv`:** Tab-separated log of experiment history (`commit`, `val_l2_rel`, `memory_gb`, `status`, `description`).
- **`CLAUDE.md`:** Operational guidance for AI agents interacting with this repository.

## Development Conventions

### The Research Loop
1. **Branching:** Create a dedicated branch for each research run: `autoresearch/<tag>`.
2. **Iteration:**
   - Edit `train.py` with a focused improvement idea.
   - Commit the change.
   - Run the experiment (`uv run train.py`).
   - Log the results in `results.tsv`.
   - **Keep** the change if `val_l2_rel` improves; otherwise, **discard** (reset to last kept commit).
3. **Budget:** Fixed 5-minute wall-clock training budget per experiment.

### Constraints
- **NO changes to `prepare.py`:** It defines the ground-truth metric.
- **NO new dependencies:** Use only the packages specified in `pyproject.toml`.
- **Target Metric:** Minimize `val_l2_rel`.
- **Git Hygiene:** Only stage `train.py` and `results.tsv`. Never use `git add -A`.

## Research Directions
1. **Fluid Mechanics:** 2D Navier-Stokes solvers.
2. **In-Context Learning:** Adapting to unseen PDE parameters at inference.
3. **Learned Preconditioners:** Optimizing convergence for stiff problems.
4. **Uncertainty Quantification:** Architecture-based UQ (MC Dropout, Ensembles).
5. **Latent Dynamics:** Neural ODEs / CDEs.
6. **Multi-Scale Solvers:** Hierarchical coarse-to-fine pathways.
7. **PINN Search:** Physics-informed residual loss optimization.
8. **Operator Search:** Alternatives like DeepONet, Wavelet Operators, AFNO.
