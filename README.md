# SciMLx (Hardware Agnostic)

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

---

## Multi-Backend Support

SciMLx is designed for cross-platform research, automatically optimizing for your hardware using a hardware-agnostic design. The system supports a dual-backend architecture where models are implemented for both PyTorch (CUDA) and MLX (Apple Silicon).

- **NVIDIA CUDA (PyTorch)**: Leverages `torch.compile`, mixed precision (AMP), and high-throughput data loading. Optimized for high-performance training on A100/H100/L4 GPUs.
- **Apple Silicon (MLX)**: Uses Apple's unified memory architecture and native MLX framework for efficient training on M-series chips.

### Usage
The system automatically detects your environment. You can explicitly override the backend using the `SCIMLX_BACKEND` environment variable:

```bash
# Force PyTorch/CUDA
export SCIMLX_BACKEND=torch
uv run train.py --model FNO ...

# Force MLX (on Apple Silicon)
export SCIMLX_BACKEND=mlx
uv run train.py --model FNO ...
```

### Dual-Backend Model Dispatch
New models in `models/` follow a dispatcher pattern. For example, `models/mff.py` automatically imports and returns the appropriate implementation from `mff_torch.py` or `mff_mlx.py` based on the active backend. This ensures code portability without sacrificing hardware-specific optimizations.

---

## Scientific Implementation (SI) Layer

SciMLx features a 3-tier production-grade Scientific Implementation layer that ensures physical consistency and mathematical rigor:

1.  **Foundations**: Core math and device abstractions.
    - `device.py`: Backend-agnostic tensor dispatch (CUDA, MLX, MPS).
    - `units.py`: Unit-aware `SciMLTensor` with dimensional analysis.
    - `lie_math.py` & `heat_kernels.py`: Lie Algebra and Geometric foundations.
    - `oracle_constants.py`: Buckingham Pi Theorem analyzer.
2.  **Models**: Production-grade neural operators.
    - `models/`: Dual-backend operator implementations.
    - `scaffold.py`: Automated multi-backend model generation.
    - `losses.py`: Physics-informed loss functions (H1, H2, Spectral).
    - `spectral_governor.py`: Frequency-aware loss modulation.
3.  **Production**: Deployment and autonomous scaling.
    - `deployment.py`: Serverless training on Vertex AI / GCP.
    - `model_versioning.py`: Registry and lineage tracking.
    - `hpo.py` & `dp_federated.py`: Bayesian HPO and Secure Federated Learning.
    - `arxiv_agent.py`: Agentic research automation (ASIL pipeline).

---

**Who this is for:** Researchers and engineers who want to systematically explore neural PDE solvers at scale on NVIDIA or Apple Silicon hardware.

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


<!-- STRUCTURE_START -->
```text
autoresearch-mlx/
├── agents/
│   └── skills/
│       └── SciMLx/
│           └── SKILL.md
├── artifacts/
│   └── presentation/
│       ├── assets/
│       │   └── gen/
│       └── src/
├── core/
│   ├── __init__.py
│   ├── arxiv_agent.py
│   ├── audio.py
│   ├── brain_distiller.py
│   ├── deployment.py
│   ├── device.py
│   ├── diagnostics.py
│   ├── dp_federated.py
│   ├── heat_kernels.py
│   ├── hpo.py
│   ├── hypothesis.py
│   ├── lie_math.py
│   ├── loader.py
│   ├── losses.py
│   ├── mlflow_integration.py
│   ├── model_versioning.py
│   ├── oracle_constants.py
│   ├── paper_registry.py
│   ├── readme_hook.py
│   ├── research_plugins.py
│   ├── results_store.py
│   ├── scaffold.py
│   ├── spectral_governor.py
│   ├── tracker.py
│   ├── trainer.py
│   ├── units.py
│   ├── utils.py
│   └── viz.py
├── dashboard/
│   ├── ui/
│   │   └── dashboard.html
│   ├── app.py
│   └── monitor.py
├── data/
│   ├── simulations/
│   │   ├── AGENTS.md
│   │   ├── __init__.py
│   │   ├── allen_cahn.py
│   │   ├── elasticity.py
│   │   ├── euler1d.py
│   │   ├── multiphysics.py
│   │   ├── ns_etdrk4.py
│   │   ├── pdebench.py
│   │   ├── radiative.py
│   │   ├── shallow_water.py
│   │   └── wavebench.py
│   ├── benchmarks_ext.py
│   ├── prefetch_data.py
│   └── prepare.py
├── docs/
│   ├── benchmarks/
│   │   ├── allen_cahn_2d.md
│   │   ├── burgers_1d.md
│   │   ├── burgers_nu_001.md
│   │   ├── burgers_nu_01.md
│   │   ├── darcy_2d.md
│   │   ├── elasticity_2d.md
│   │   ├── euler_1d.md
│   │   ├── kdv_1d.md
│   │   ├── mhd_2d.md
│   │   ├── multiphysics_2d.md
│   │   ├── ns_2d.md
│   │   ├── ns_hre_2d.md
│   │   ├── pdebench_2d.md
│   │   ├── poisson_2d.md
│   │   ├── radiative_2d.md
│   │   ├── reionization_1d.md
│   │   ├── swe_2d.md
│   │   ├── wave_1d.md
│   │   └── wavebench_2d.md
│   ├── maestro/
│   │   ├── plans/
│   │   │   └── archive/
│   │   │       ├── 2026-04-27-asil-pipeline-design.md
│   │   │       ├── 2026-04-27-asil-pipeline-impl-plan.md
│   │   │       ├── 2026-04-27-multi-backend-integration.md
│   │   │       ├── 2026-04-27-scimlx-vision-refactor-design.md
│   │   │       └── 2026-04-27-scimlx-vision-refactor-impl-plan.md
│   │   └── state/
│   ├── papers/
│   │   ├── afno_2022.yaml
│   │   ├── augmentation_2023.yaml
│   │   ├── cosmic_reionization_pinn_2023.yaml
│   │   ├── curriculum_2009.yaml
│   │   ├── deeponet_2021.yaml
│   │   ├── ensemble_uq_2023.yaml
│   │   ├── fbpinn_2023.yaml
│   │   ├── ffno_2023.yaml
│   │   ├── fno_2020.yaml
│   │   ├── gnot_2023.yaml
│   │   ├── h1_loss.yaml
│   │   ├── hnn_2019.yaml
│   │   ├── inverse_pinn_2023.yaml
│   │   ├── mambano_2024.yaml
│   │   ├── memno_2025.yaml
│   │   ├── modal_pinn_2024.yaml
│   │   ├── mppde_2022.yaml
│   │   ├── neural_ode_ude_2020.yaml
│   │   ├── neural_pde_solver_2023.yaml
│   │   ├── packed_ensemble_2023.yaml
│   │   ├── pdebench_2024.yaml
│   │   ├── physicsnemo_2024.yaml
│   │   ├── pikan_2025.yaml
│   │   ├── pinnacle_2024.yaml
│   │   ├── pino_2021.yaml
│   │   ├── pod_dl_rom_2023.yaml
│   │   ├── rfno_2024.yaml
│   │   ├── sar_2026.yaml
│   │   ├── spline_pinn_2024.yaml
│   │   ├── ssm_s4_2022.yaml
│   │   ├── tfno_2022.yaml
│   │   ├── time_marching_deeponet_2025.yaml
│   │   ├── transolver_2024.yaml
│   │   ├── uno_2022.yaml
│   │   └── wno_2022.yaml
│   ├── proposals/
│   │   ├── TEMPLATE.md
│   │   ├── test_proposal.md
│   │   └── test_refactor_scaffold.md
│   ├── ARCHITECTURE.md
│   ├── BENCHMARKS.md
│   ├── CONTRIBUTING.md
│   ├── LICENSE
│   ├── LITERATURE.md
│   ├── SOTA.md
│   ├── TERMINOLOGY.md
│   └── VISION_2026.md
├── models/
│   ├── layers/
│   │   └── equivariant.py
│   ├── __init__.py
│   ├── afno.py
│   ├── attention_fno.py
│   ├── axial_attention.py
│   ├── chebyshev_kan.py
│   ├── deeponet.py
│   ├── dualmodeltest_mlx.py
│   ├── dualmodeltest_torch.py
│   ├── fedonet.py
│   ├── fno.py
│   ├── gato.py
│   ├── gnot.py
│   ├── hano.py
│   ├── hnn.py
│   ├── hybrid_decoder_deeponet.py
│   ├── hybrid_fno_deeponet.py
│   ├── kan.py
│   ├── kan_refiner.py
│   ├── mamba_no.py
│   ├── mambafno.py
│   ├── mambafno_mlx.py
│   ├── mambafno_torch.py
│   ├── mem_no.py
│   ├── mff.py
│   ├── mff_mlx.py
│   ├── mff_torch.py
│   ├── neural_ode.py
│   ├── pacmann.py
│   ├── pinn.py
│   ├── s4d.py
│   ├── sar.py
│   ├── sno.py
│   ├── ssno.py
│   ├── testnet_mlx.py
│   ├── testnet_torch.py
│   ├── tfno.py
│   ├── time_deeponet.py
│   ├── transolver.py
│   ├── vsmno.py
│   └── wno.py
├── notebooks/
├── sciml_mlx/
│   ├── core/
│   ├── dashboard/
│   ├── data/
│   │   └── simulations/
│   ├── models/
│   └── results.db
├── scripts/
│   ├── gcp/
│   │   ├── setup_vm.sh
│   │   └── submit_vertex.py
│   ├── maintenance/
│   │   ├── backfill_model_registry.py
│   │   ├── gen_arch_nanobanana.py
│   │   └── gen_arch_viz.py
│   ├── asil_ideate.py
│   ├── asil_scaffold.py
│   └── dvc_train.py
├── tests/
│   ├── integration/
│   │   ├── test_asil_loop.py
│   │   ├── test_federated.py
│   │   ├── test_gato.py
│   │   ├── test_mff.py
│   │   ├── test_parity.py
│   │   └── test_si_modules.py
│   ├── smoke/
│   └── unit/
│       ├── test_arxiv_agent.py
│       ├── test_heat_kernels.py
│       ├── test_lie_math.py
│       ├── test_oracle_constants.py
│       ├── test_spectral_governor.py
│       └── test_units.py
├── Dockerfile
├── README.md
├── RESEARCH_BRAIN.md
├── agent_loop.py
├── analyze.py
├── auto_suggest.py
├── autorun.py
├── dvc.yaml
├── experiments.yaml
├── mlflow.db
├── model_registry.json
├── params.yaml
├── pyproject.toml
├── results.db
├── results.json
├── train.py
└── uv.lock
```
<!-- STRUCTURE_END -->
