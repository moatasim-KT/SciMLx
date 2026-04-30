# System Architecture (Hardware Agnostic)

Deep technical reference for the SciMLx autonomous research loop components, optimized for both NVIDIA GPUs (PyTorch) and Apple Silicon (MLX).

---

## 3-Tier Scientific Implementation (SI) Layer

SciMLx utilizes a modular SI layer in `core/` to decouple scientific logic from underlying hardware and compute frameworks. The layer is organized into three tiers:

### Tier 1: Foundations
Core math and device abstractions that provide a stable, hardware-agnostic base.
- **`device.py` (Hardware Agnostic Dispatch)**: Automatically detects the best available backend (CUDA, MLX, MPS, or CPU) and provides a single API for framework-agnostic tensor creation (`to_array()`) and device placement (`to_device()`).
- **`units.py` (`SciMLTensor`)**: Ensures mathematical and physical groundedness by performing dimensional analysis on every operation using the `pint` unit registry.
- **`lie_math.py` & `heat_kernels.py` (Geometric Math)**: Implements Lie Algebra foundations and mesh-based heat kernel signatures for geometry-aware operators.
- **`oracle_constants.py` (Buckingham Pi Theorem)**: Identifies dimensionless groups (like Reynolds or Péclet numbers) to aid in feature discovery and similarity analysis.

### Tier 2: Models
Production-grade neural operators and the infrastructure to build them.
- **Dual-Backend Operators**: All new models are implemented with both PyTorch (`_torch.py`) and MLX (`_mlx.py`) backends, managed by a central dispatcher (e.g., `models/mff.py`).
- **`scaffold.py` (Automated Scaffolding)**: Generates dual-backend model stubs from research proposals, ensuring feature parity across hardware.
- **`losses.py` (Physics-Informed Operators)**: Framework-agnostic implementations of Sobolev ($H^1$, $H^2$) and Spectral losses that penalize unphysical oscillations.
- **`spectral_governor.py` (Frequency Governance)**: Dynamically monitors the Fourier spectrum of residuals across backends and adjusts loss weighting to ensure high-frequency features are captured.

### Tier 3: Production
Systems for scaling, deploying, and automating the research cycle.
- **`deployment.py`**: Integration with Google Cloud Platform (Vertex AI, Compute Engine) for serverless GPU training.
- **`model_versioning.py`**: A centralized model registry and lineage tracking system for managing champion models.
- **`hpo.py` (Bayesian Optimization)**: Automated hyperparameter search that adapts to hardware-specific constraints.
- **`dp_federated.py` (Differential Privacy)**: Logic for secure, privacy-preserving federated training of scientific models.
- **`arxiv_agent.py` (ASIL Pipeline)**: Orchestrates the Agentic Scientist Ideation Loop, from literature review to automated model scaffolding.

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
