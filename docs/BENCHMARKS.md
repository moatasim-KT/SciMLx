# SciML Benchmark Index & Tuning Guide (CUDA Optimized)

This document serves as a master index for the PDE benchmarks in the SciMLx project. All benchmarks are now fully GPU-accelerated via PyTorch/CUDA.

---

## 📊 Benchmark Index

| ID | Dimension | Category | Status | Details |
|---|---|---|---|---|
| `burgers_1d` | 1D | Fluid (Shock) | [STABLE] | [Spec](benchmarks/burgers_1d.md) |
| `burgers_nu_001` | 1D | Fluid (Strong Shock) | [STABLE] | [Spec](benchmarks/burgers_nu_001.md) |
| `kdv_1d` | 1D | Fluid (Soliton) | [STABLE] | [Spec](benchmarks/kdv_1d.md) |
| `wave_1d` | 1D | Waves | [STABLE] | [Spec](benchmarks/wave_1d.md) |
| `darcy_2d` | 2D | Elliptic Flow | [FIXED (EXT)] | [Spec](benchmarks/darcy_2d.md) |
| `ns_2d` | 2D | Incompressible Flow | [FIXED (EXT)] | [Spec](benchmarks/ns_2d.md) |
| `ns_hre_2d` | 2D | Turbulence (Re=1000) | [FIXED (EXT)] | [Spec](benchmarks/ns_hre_2d.md) |
| `swe_2d` | 2D | Gravity Waves | [STABLE] | [Spec](benchmarks/swe_2d.md) |
| `allen_cahn_2d` | 2D | Phase Field | [STABLE] | [Spec](benchmarks/allen_cahn_2d.md) |

---

## GPU-Accelerated Solvers

Previously CPU-bound (NumPy), all simulation solvers (in `data/simulations/`) now run directly on the **NVIDIA GPU**. This provides:
- **Instant Data Generation**: Eliminates the bottleneck where the GPU waits for the CPU to solve the PDE.
- **Batched Simulation**: Hundreds of initial conditions can be solved in parallel.
- **Higher Fidelity**: Enables larger grids and longer temporal integrations within the same time budget.

---

## 1D vs 2D: CUDA Capacity

| Aspect | 1D Benchmarks | 2D Benchmarks |
|---|---|---|
| Input shape | `(B, 64, C)` | `(B, 64, 64, C)` |
| Typical `hidden_dim` | 128–256 | 64–128 (on L4/A100) |
| Memory usage | Minimal | High (requires Pinned Memory) |
| **`torch.compile`** | Fast (1.5x) | Massive (2-3x speedup) |

### Memory Guidelines for NVIDIA GPUs
- **L4 (24 GB)**: Safely supports `hidden_dim=64`, `n_layers=8` for most 2D benchmarks.
- **A100/H100 (40/80 GB)**: Can scale to `hidden_dim=128+`.
- **T4 (16 GB)**: Recommended to stay at `hidden_dim=32` for 2D to avoid OOM.

---

## Budget Recommendations (NVIDIA L4)

Estimated training times to reach reasonable convergence:

| Category | Fast Check | Convergence | Full Champion Run |
|---|---|---|---|
| 1D (FNO/WNO) | 5 min | 15–30 min | 1 hr |
| 2D (FNO2D) | 10 min | 30–60 min | 2 hr |
| 2D (Transolver/GNOT) | 20 min | 1–2 hr | 4 hr |

> **Note:** `torch.compile` adds a 1–2 minute overhead at the start of the run but provides much higher throughput thereafter.

---

## Loss Function × Benchmark Matrix

| Benchmark | Recommended Loss | Reason |
|---|---|---|
| `burgers_1d` | `h1` | Gradient penalty sharpens shock fronts. |
| `ns_2d` | `spectral` | Helps recover energy in high-frequency turbulent modes. |
| `darcy_2d` | `h1` | Captures discontinuities in conductivity fields. |
| `wave_1d` | `l2_rel` | Simple L2 is sufficient for smooth wave propagation. |

---

## Benchmark Tuning Cards

### `burgers_1d`
**Strategy**: Use `h1_loss` with a small `alpha` (0.1). If shocks are extremely sharp (`nu=0.001`), switch to **WNO** (Wavelet Neural Operator).

### `darcy_2d`
**Strategy**: Use **Transolver2D** or **GNOT**. These models handle the variable-coefficient nature of Darcy flow better than standard FNOs. Ensure `augment: true` is enabled for better spatial generalization.

### `ns_2d`
**Strategy**: Use **FNO2D** with `spectral_loss`. The turbulent energy cascade requires accurate modeling of high-wavenumber components.
