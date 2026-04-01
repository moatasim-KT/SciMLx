# autoresearch-sciml-mlx

Apple Silicon (MLX) port of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), repurposed for **Scientific Machine Learning (SciML)**.

This repository runs autonomous AI-driven research loops to explore and optimize PDE solvers, Neural Operators, and Physics-Informed Neural Networks. The AI agent modifies a mutable `train.py` within a fixed 5-minute budget and keeps or reverts changes based on a fixed metric (`val_loss`, usually Relative L2 Error), leveraging natively on Apple Silicon through [MLX](https://github.com/ml-explore/mlx) without PyTorch or CUDA dependencies.

## SciML Research Objectives

This repository is designed to encourage AI agents to explore advanced SciML topics:
- **Fluid Mechanics & Navier-Stokes Solvers**
- **In-Context PDE Solving**
- **Learned Preconditioners**
- **Uncertainty Quantification (UQ) via Architecture**
- **Latent Dynamics Models (Neural ODEs / CDEs)**
- **Multi-Scale / Hierarchical PDE Solvers**
- **Physics-Informed Neural Network (PINN) Optimizer Search**
- **Neural Operator Architecture Search**

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync

# one-time data synthesis for PDEs
uv run prepare.py

# run one 5-minute training experiment
uv run train.py
```

Then point an AI coding agent at `program.md` and let it run the loop.

## What matters

- `prepare.py` - Synthesizes / loads PDE data and handles evaluation (`evaluate_pde`). Treat as fixed.
- `train.py` - The model (e.g., FNO, PINN), optimizer, and training loop. This is the file the agent edits.
- `program.md` - The autonomous experiment protocol.
- `results.tsv` - Logged experiment history.

## Acknowledgments
- Andrej Karpathy for the autonomous research concept.
- MLX team.
