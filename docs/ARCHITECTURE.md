# System Architecture (CUDA Optimized)

Deep technical reference for the SciMLx autonomous research loop components, optimized for NVIDIA GPUs.

---

## Two-Mode Operation

The system supports two operating modes that can be mixed within a session:

**Mode A — Human-Guided**  
A human (or AI agent) reads `RESEARCH_BRAIN.md`, interprets results, edits `experiments.yaml` directly, and invokes `autorun.py` to execute the queue. The system handles execution, retry, and logging; the human handles strategy.

**Mode B — Fully Autonomous (`agent_loop.py`)**  
`agent_loop.py` performs one full autonomous cycle:
1. Calls `tracker.analyze_lineage()` to build per-benchmark summaries.
2. Calls `HypothesisEngine.analyze_benchmark()` for each priority benchmark.
3. Calls `BayesianHPO.ask()` to sample hyperparameters.
4. Generates new `ExperimentConfig` entries and appends them to `experiments.yaml`.
5. Triggers `autorun.py` to process the queue on NVIDIA GPUs.

---

## Unified Trainer (`core/trainer.py`)

The trainer is rewritten in PyTorch and optimized for NVIDIA hardware:

### CUDA Optimizations
- **`torch.compile()`**: Used by default to fuse kernels and optimize graph execution. Provides significant speedups for complex operators like Transolver and AFNO.
- **Mixed Precision (AMP)**: Leverages `torch.amp.autocast` and `GradScaler` to utilize Tensor Cores (FP16/BF16) without losing numerical stability.
- **Tensor Core Precision**: `torch.set_float32_matmul_precision('high')` is enabled globally.

### Training Logic
- **EMA (Exponential Moving Average)**: Maintains a shadow copy of model weights for more stable evaluation.
- **Dynamic Budget Extension**: Automatically extends training time by 20% if the loss is still decreasing significantly at the end of the budget.
- **Snapshot Ensembling**: Optionally saves and averages multiple model states throughout the run.

---

## GPU-Accelerated PDE Solvers (`data/simulations/`)

Unlike original CPU-bound implementations, all solvers are now PyTorch-based:
- **Spectral Methods**: Use `torch.fft` for high-speed spectral derivatives and integration.
- **Zero-Copy Data**: Solvers execute directly on the `DEVICE` (CUDA), producing tensors that never leave GPU memory during training.
- **Batch Processing**: All simulations are vectorized to solve multiple initial conditions in parallel.

---

## High-Throughput Data Pipeline

The I/O bottleneck is eliminated through:
1. **`PDEDataset`**: An `IterableDataset` that interfaces with cached `.npz` files or on-the-fly solvers.
2. **`DataLoader`**: Standard PyTorch implementation with:
   - `pin_memory=True`: For faster Host-to-Device transfer.
   - `num_workers > 0`: For multi-process data pre-fetching.
   - `prefetch_factor`: To keep the GPU saturated.

---

## HypothesisEngine (`core/hypothesis.py`)

The engine classifies experiment outcomes to guide follow-up logic:

| Mode | Detection Logic |
|---|---|
| `gradient_collapse` | `val_l2_rel ≥ 1.0`, NaN loss, or CUDA launch errors. |
| `spectral_bias` | High-frequency error > 0.3 in spectral diagnostic. |
| `capacity_limited` | Small model (`hidden_dim < 64`) with high error. |
| `cuda_oom` | Log analysis detects "Out of Memory" on GPU. |

---

## Retry Escalation (`autorun.py`)

Escalates through recovery levels for NVIDIA environments:
- **r1 — `smart_fix()`**: Detects CUDA OOM and automatically halves `hidden_dim`.
- **r2**: Aggressive reduction of `hidden_dim`, `n_layers`, and `n_modes`.
- **r3**: Minimal viable fallback (`h=32, l=2, lr=1e-4`).

---

## Cloud Infrastructure (GCP)

Configured for project `gdpr-494411`:
- **Vertex AI**: Custom container execution using the project's Artifact Registry.
- **Compute Engine**: G2-standard instances with NVIDIA L4 GPUs for development.
