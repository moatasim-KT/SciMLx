# SciML AutoResearch Wiki

Welcome to the **SciML AutoResearch** platform wiki. This document serves as a comprehensive technical guide for both human researchers and autonomous agents (Mode A/B).

---

## 🔬 Core Concept: The "5-Minute" Research Protocol

This platform is designed for **high-velocity iterative research** on Scientific Machine Learning (SciML) using Apple Silicon (MLX). 

- **Budgeted Training:** Most 1D experiments are hard-limited to **300 seconds (5 minutes)**. 2D experiments use **480-600 seconds**.
- **The Metric:** Success is measured by `val_l2_rel` (Relative L2 Error) on a fixed validation set. Lower is better.
- **The Loop:** Observation → Hypothesis → Experiment → Result → Next Hypothesis.

---

## 🏗️ System Architecture

The platform follows a decoupled, registry-driven architecture to allow for rapid extension.

### High-Level Data Flow
```text
[ experiments.yaml ]  <───  [ agent_loop.py ] (Mode B)
      │                            ^
      │ (loads queue)              │ (analyzes results)
      v                            │
[ autorun.py ]  ───────>  [ train.py ]  ───────>  [ results.json ]
      │ (spawns subprocess)        │ (emits metrics)       (SSoT)
      v                            v
[ logs/<name>.log ]       [ checkpoints/ ]
```

### Component Roles
- **`train.py`:** The training harness. Routes benchmarks to loaders and models to the trainer.
- **`core/research_plugins.py`:** The central registry. Maps strings like `"FNO"` to model classes and `"burgers_1d"` to data generators.
- **`core/trainer.py`:** Handles the MLX-specific JIT-compiled training loop and `AdamW` optimization.
- **`data/prepare.py`:** The **Read-Only** authority on data generation and evaluation.
- **`core/tracker.py`:** Manages the lineage-aware logging of experiments into `results.json`.

---

## 📊 Data Structures

### 1. `ExperimentConfig` (defined in `experiments.yaml`)
Each experiment is a declarative entry in `experiments.yaml`.
- `name`: Unique identifier (UUID suffix recommended).
- `benchmark`: The target PDE (e.g., `burgers_1d`, `ns_2d`).
- `model`: The architecture to use (e.g., `FNO`, `RFNO`, `UNO`).
- `hidden_dim`, `n_layers`, `n_modes`: Primary hyperparameters.
- `priority`: 1 (High) to 5 (Low).
- `parent_name`: Name of the parent run in the lineage DAG.
- `rationale`: Natural language explanation of why this run was created.

### 2. `results.json` (The Experiment DAG)
A JSON array of completed runs. Each entry contains:
- `id`, `parent_id`: Defines the lineage graph.
- `val_l2_rel`: The primary success metric.
- `diag_*`: Spectral diagnostics (error per frequency band).
- `status`: `keep` (successful), `discard` (worse than parent), or `error` (crash).

---

## 🤖 Model Zoo

The platform currently supports 28+ model exports across several families:

| Family | Models | Strengths |
|---|---|---|
| **Fourier (FNO)** | `FNO1d`, `FNO2d`, `RFNO1d`, `UNO1d` | Global spectral dependencies, very efficient for periodic PDEs. |
| **DeepONet** | `DeepONet`, `PODDeepONet`, `TimeDeepONet` | General operator learning, handles complex geometries. |
| **State Space (SSM)** | `S4NO1d`, `SSNO1d` | Long-range temporal dependencies, adaptive damping. |
| **Physics-Informed** | `PINN`, `PINO1d` | Incorporates PDE residuals into the loss function. |
| **Attention** | `Transolver`, `GNOT`, `Transformer` | High-fidelity resolution, physics-aware attention. |

---

## 📈 Benchmark Catalog

| Benchmark | Type | SOTA | Our Best | Constraints |
|---|---|---|---|---|
| `burgers_1d` | 1D Viscous | 0.0031 | 0.1468 | High gap; needs curriculum/augmentation. |
| `kdv_1d` | 1D Soliton | 0.010 | 0.0020 | **Winner.** 5x better than SOTA. |
| `wave_1d` | 1D Wave | 0.005 | 0.0009 | **Winner.** 5x better than SOTA. |
| `darcy_2d` | 2D Darcy | 0.0041 | 0.1041 | **OOM Sensitive.** Max h=32, l=4. |
| `ns_2d` | 2D NS | 0.0128 | 0.0142 | Near SOTA; budget=600s. |

---

## 🛠️ Operational Guide for Agents

### 1. Indexing & Navigation
Use `CODE_INDEX.json` for a machine-readable map of all symbols and docstrings.
- **Models:** `models/` (One file per family).
- **Core Logic:** `core/` (Utility modules).
- **Data Logic:** `data/` (PDE solvers).

### 2. Research Workflow (Mode A)
1.  **Analyze:** `uv run analyze.py --papers` to see the current gaps.
2.  **Suggest:** `uv run auto_suggest.py` to get ranked suggestions from the `HypothesisEngine`.
3.  **Propose:** Append new `ExperimentConfig` to `experiments.yaml`.
4.  **Execute:** `uv run autorun.py --priority 1 --commit`.

### 3. Automated HPO (Mode B)
Run `uv run agent_loop.py --top 5` to let the system automatically propose 5 new experiments based on Bayesian Optimization (via `core/hpo.py`).

---

## 🔍 Diagnostics & Troubleshooting

| Observation | Meaning | Suggested Action |
|---|---|---|
| `diag_high_freq_error` > 0.1 | Model missing fine details. | Increase `n_modes` or use `SpectralLoss`. |
| `NaN` in loss | Gradient explosion. | Decrease `lr`, use `grad_clip`, or check `pino_lambda`. |
| `OOM` (2D runs) | Model too large for VRAM. | Decrease `hidden_dim` (<=32) or `n_layers` (<=4). |
| `l2_rel` plateaus | Optimization bottleneck. | Change `lr` schedule or try `RFNO` for stable depth. |

---

## 🔗 External Resources
- `AGENTS.md`: Detailed field guide for external agents.
- `GEMINI.md`: Full setup and reference guide.
- `program.md`: Registry of papers and SOTA targets.
- `CODE_INDEX.json`: Machine-readable symbol mapping.
