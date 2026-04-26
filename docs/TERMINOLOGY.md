# SciML Terminology

## Core Training Concepts (CUDA Optimized)

- **Mixed Precision (AMP):** Automatically switching between FP16 and FP32 during training to utilize Tensor Cores. Provides significant speedup on NVIDIA hardware without sacrificing accuracy.

- **`torch.compile`:** A JIT compiler that fuses operations and optimizes the compute graph. In SciMLx, this is used to accelerate complex spectral and attention-based operators.

- **Pinned Memory:** Memory allocated in "page-locked" RAM, allowing for faster transfer between the CPU (Host) and the NVIDIA GPU (Device).

- **Relative L2 Error:** `||u_pred − u_gt||_2 / ||u_gt||_2`. The standard metric in this repository.

## Model Architectures

- **Spectral Convolution:** Convolution performed in the frequency domain via `torch.fft`. Filters to the lowest Fourier coefficients.

- **FNO Block:** SpectralConv + pointwise Linear + GELU activation.

- **WNO (Wavelet Neural Operator):** Uses Haar wavelet decomposition instead of Fourier modes. Better spatial localization for shocks.

- **Transolver:** Physics-slice attention transformer that captures multi-scale features through learnable physical slices.

## Training Concepts

- **AdamW:** Adam optimiser with weight decay. Standard for operator learning.

- **Gradient Clipping:** Normalizing gradients to prevent instability (common in high-Reynolds Navier-Stokes).

- **H1 / Sobolev Loss:** `||u||_{H1}² = ||u||_{L2}² + ||∂u/∂x||_{L2}²`. Penalises high-frequency errors.

## PDE Solvers

- **GPU-Accelerated Simulation:** PDE solvers (Navier-Stokes, Allen-Cahn, etc.) implemented in PyTorch/CUDA to enable high-throughput data generation during training.

- **Spectral Derivative:** Computing spatial derivatives in the frequency domain using `torch.fft` for exponential accuracy.

## Metrics & Benchmarking

- **Step time (dt):** Wall-clock time per gradient step. On NVIDIA L4/A100 GPUs, expect 5–50 ms for 1D models and 50–200 ms for 2D models using `torch.compile`.
