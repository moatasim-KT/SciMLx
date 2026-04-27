# System Architecture (Hardware Agnostic)

Deep technical reference for the SciMLx autonomous research loop components, optimized for both NVIDIA GPUs (PyTorch) and Apple Silicon (MLX).

---

## 3-Tier Scientific Implementation (SI) Layer

SciMLx utilizes a modular SI layer in `core/` to decouple scientific logic from underlying hardware and compute frameworks.

### Tier 1: Hardware Agnostic Tier (`device.py`)
Provides a unified abstraction for tensor operations. It automatically detects the best available backend (CUDA, MLX, MPS, or CPU) and provides a single API for:
- **`to_array()`**: Framework-agnostic tensor creation.
- **`to_device()`**: Unified device placement.
- **Backend Switching**: Controlled via the `SCIMLX_BACKEND` environment variable.

### Tier 2: Physical Tier (`units.py` & `oracle_constants.py`)
Ensures that the "Sci" in SciML is mathematically and physically grounded.
- **Unit Registry**: Uses `pint` to manage physical units (m, s, kg, etc.).
- **`SciMLTensor`**: A wrapper that performs dimensional analysis on every operation, raising errors for physically impossible calculations (e.g., adding meters to seconds).
- **Buckingham Pi Theorem**: The `OracleOfConstants` identifies dimensionless groups (like Reynolds or Péclet numbers) to aid in feature discovery and similarity analysis.

### Tier 3: Mathematical Operator Tier (`losses.py` & `spectral_governor.py`)
High-level scientific operators that guide the training process.
- **Physics-Informed Losses**: Implementations of Sobolev ($H^1$, $H^2$) and Spectral losses that penalize unphysical oscillations.
- **Spectral Bias Governor**: Dynamically monitors the Fourier spectrum of residuals across backends and adjusts loss weighting to ensure high-frequency features are captured.

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

The trainer is designed to be high-performance while remaining flexible across backends:

### Compute Optimizations
- **NVIDIA/PyTorch**: Utilizes `torch.compile()` for kernel fusion and `torch.amp` for mixed precision training.
- **Apple/MLX**: Leverages MLX's lazy evaluation and unified memory for efficient processing on M-series chips.
- **Precision Management**: Configurable precision levels (float32, bfloat16) mapped to hardware-specific best practices.

### Training Logic
- **EMA (Exponential Moving Average)**: Maintains a shadow copy of model weights for more stable evaluation.
- **Dynamic Budget Extension**: Automatically extends training time by 20% if the loss is still decreasing significantly at the end of the budget.
- **Snapshot Ensembling**: Optionally saves and averages multiple model states throughout the run.

---

## Hardware-Accelerated PDE Solvers (`data/simulations/`)

All PDE solvers are implemented using framework-native spectral methods to ensure high-speed simulation on the active device:
- **Spectral Methods**: Utilize fast Fourier transforms (`torch.fft` or `mlx.fft`) for high-speed spectral derivatives and integration.
- **Zero-Copy Data**: Solvers execute directly on the `DEVICE`, producing tensors that never leave high-speed device memory during training.
- **Batch Processing**: All simulations are vectorized to solve multiple initial conditions in parallel, maximizing device throughput.

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
