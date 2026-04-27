# SciMLx (CUDA Optimized)

**Autonomous neural operator research loop for PDE solving on NVIDIA GPUs.**  
Queue an experiment, go to sleep — the system trains, evaluates, diagnoses failures,
proposes follow-ups, and updates itself overnight.

[![PyTorch](https://img.shields.io/badge/Platform-PyTorch%20%28CUDA%29-orange.svg)](https://pytorch.org/)
[![NVIDIA](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Active%20Research-green.svg)](#project-status)

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Architecture](#architecture)
5. [Model Zoo](#model-zoo)
6. [PDE Benchmarks](#pde-benchmarks)
7. [Configuration](#configuration)
8. [Training a Single Model](#training-a-single-model)
9. [Running the Autonomous Loop](#running-the-autonomous-loop)
10. [Loss Functions](#loss-functions)
11. [Bayesian HPO](#bayesian-hpo)
12. [Cloud Deployment (GCP)](#cloud-deployment-gcp)
13. [Dashboard](#dashboard)
14. [Agentic Scientist Ideation Loop (ASIL)](#agentic-scientist-ideation-loop-asil)
15. [Results & Tracking](#results--tracking)
16. [Troubleshooting](#troubleshooting)
17. [Project Status](#project-status)

---

## Overview

SciMLx is a high-performance, self-driving experiment harness for **neural operator** research. Originally developed for MLX, it has been fully migrated and optimized for **NVIDIA CUDA** using PyTorch.

**What it does:**

- **30+ Neural Operator Architectures**: FNO, MambaNO, Transolver, Neural ODEs, KANs, PINNs, and more — all ported to PyTorch/CUDA.
- **GPU-Accelerated Solvers**: 15+ PDE benchmarks (Navier-Stokes, Allen-Cahn, etc.) with solvers running directly on the GPU.
- **Autonomous Research**: Orchestrates overnight campaigns: train → evaluate → diagnose → propose next experiment → repeat.
- **CUDA Optimizations**: Utilizes `torch.compile`, Mixed Precision (AMP), and high-throughput data loading with Pinned Memory.
- **Lineage Tracking**: Records every experiment in `results.json` and manages champions in a model registry.

**Who this is for:** Researchers and engineers who want to systematically explore neural PDE solvers at scale on NVIDIA hardware.

---

## Quick Start

### Prerequisites

- **NVIDIA GPU** (Ampere, Hopper, or Blackwell recommended)
- **CUDA Toolkit** 12.1+
- **Python** 3.10–3.13
- [`uv`](https://github.com/astral-sh/uv) package manager

```bash
# Clone and install
git clone <repo-url> scimlx
cd scimlx
uv sync
```

### Your First Training Run (~2 minutes)

```bash
uv run train.py \
  --benchmark burgers_1d \
  --model FNO \
  --hidden 64 \
  --layers 4 \
  --modes 16 \
  --lr 1e-3 \
  --budget 120
```

### Running the Autonomous Loop

```bash
# Process the experiments.yaml queue with auto-commit
uv run autorun.py --auto --commit
```

---

## Architecture

SciMLx follows a modular design optimized for CUDA data throughput:

1.  **Unified Trainer**: Built-in support for `torch.compile` (kernel fusion) and `torch.amp` (mixed precision).
2.  **GPU Solvers**: All PDE simulations (in `data/simulations/`) are implemented in PyTorch and execute on the `DEVICE`.
3.  **Data Pipeline**: Uses `torch.utils.data.DataLoader` with `num_workers > 0` and `pin_memory=True` to eliminate I/O bottlenecks.
4.  **Autonomous Loop**: A hierarchy of retry logic and failure analysis that adapts hyperparameters based on GPU logs.

---

## Model Zoo

30+ neural operator architectures, all optimized for CUDA:

### Fourier & Spectral Operators
- **FNO / FNO2D**: Fourier Neural Operators (1D & 2D)
- **AFNO / FFNO**: Adaptive and Factorized variants
- **TFNO**: Tucker/CP-factorized FNO for parameter efficiency
- **WNO**: Wavelet Neural Operator (Haar decomposition)
- **SNO2D**: Spectral Neural Operator

### State-Space & Attention
- **MambaNO**: Mamba-based operator learning
- **Transolver**: Physics-slice attention transformer
- **GNOT**: Graph Neural Operator Transformer
- **S4NO / SSNO**: State-space neural operators

### Hybrid & Physics-Informed
- **PINN / PINO**: Physics-Informed Neural Networks and Operators
- **DeepONet / PODDeepONet**: Branch-Trunk and Basis-based DeepONets
- **NeuralODE / UDE**: Continuous-time dynamics
- **cPIKAN**: Chebyshev polynomial KANs

---

## Cloud Deployment (GCP)

SciMLx is pre-configured for **Google Cloud Platform** (Project `gdpr-494411`):

- **Vertex AI**: Use the provided `Dockerfile` and `scripts/gcp/submit_vertex.py` for serverless GPU training.
- **Compute Engine**: Use `scripts/gcp/setup_vm.sh` to provision a Deep Learning VM with L4/A100/H100 GPUs.

---

## Agentic Scientist Ideation Loop (ASIL)

SciMLx features an autonomous research pipeline that moves from literature review to code implementation without manual intervention.

### Workflow
1.  **ArXiv Scan**: `asil_ideate.py` fetches recent papers and identifies SOTA gaps.
2.  **Autonomous Synthesis**: The AI synthesizes novel architectural ideas based on the knowledge base in `RESEARCH_BRAIN.md`.
3.  **Research Proposal**: A structured markdown proposal is generated in `docs/proposals/`.
4.  **Human Approval**: Researchers review and approve proposals.
5.  **Automated Scaffolding**: `asil_scaffold.py` parses the proposal, generates PyTorch code, registers the model, and updates the experiment queue.

### Usage Examples

**Generate a new idea:**
```bash
python scripts/asil_ideate.py --keywords "Fourier Neural Operator" --novelty high
```

**Scaffold an approved proposal:**
```bash
python scripts/asil_scaffold.py --proposal docs/proposals/2026-04-27-hybrid-mamba-fno.md
```

---

## Project Status

**Current Performance:**
- **9 of 14 SOTA targets beaten** in initial benchmarks.
- Full support for large-scale 2D simulations on A100/H100.
- Structurally ready for multi-GPU training (Phase 15 direction).

---

*For detailed research findings and SOTA gaps, see [`RESEARCH_BRAIN.md`](./RESEARCH_BRAIN.md).*
