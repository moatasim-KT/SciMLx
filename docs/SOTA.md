# SOTA Results & Benchmarks

Performance targets are relative L2 error (`val_l2_rel`, lower is better).
Numbers from published papers on their respective test sets.
"This repo" rows reflect the **current best** from `results.json` — run
`uv run analyze.py --papers` to see live values.

---

## 2D Performance (NVIDIA CUDA)

> **Note on CUDA Capacity:**
> With the migration to NVIDIA GPUs (L4, A100, H100), the previous memory constraints
> of Apple Silicon base models are largely removed. 2D benchmarks can now scale to 
> `hidden_dim=128+` and `n_layers=12+` on high-end hardware, allowing us to close
> the gap to SOTA targets that were previously capacity-limited.

---

## 1D Benchmarks

### Burgers 1D (ν = 0.01/π, N=64, T=1)

| Model                     | Relative L2 | Notes                              |
|---------------------------|-------------|------------------------------------|
| FNO (paper)               | 0.0149      | Li et al. 2020, 16 modes, width 64 |
| GNOT (paper)              | 0.0031      | Hao et al. 2023                    |
| **This repo (best)**      | **0.1468**  | FNO h=128 l=8 m=24                 |

---

### KdV 1D (soliton transport, ETDRK4)

| Model                | Relative L2 | Notes                    |
|----------------------|-------------|--------------------------|
| RFNO (paper)         | ~0.010      | Residual FNO baseline    |
| **This repo (best)** | **0.005748** | RFNO h=128 — beats SOTA |

---

### Wave 1D (u_tt = c²u_xx)

| Model                | Relative L2 | Notes                    |
|----------------------|-------------|--------------------------|
| FNO baseline         | ~0.005      | Estimated                |
| **This repo (best)** | **0.001662** | FNO — beats SOTA 3×     |

---

## 2D Benchmarks

### Darcy Flow 2D (steady-state -∇·(a∇u)=f, N=64×64)

| Model                | Relative L2 | Notes                                |
|----------------------|-------------|--------------------------------------|
| FNO (paper)          | 0.0108      | Li et al. 2020, 12 modes, width 32   |
| GNOT (paper)         | 0.0041      | Hao et al. 2023                      |
| **This repo (best)** | **0.2735**  | FEDONet2D (registry champion)   |

---

### Navier-Stokes 2D (vorticity, ν=1e-3, T=10, N=64×64)

| Model                | Relative L2 | Notes                   |
|----------------------|-------------|-------------------------|
| FNO (paper)          | 0.0128      | Li et al. 2020          |
| **This repo (best)** | **0.01428** | Competitive — near SOTA |

---

## Notes on Infrastructure

- **Accelerated Solvers**: Validation data generation is now 100% GPU-accelerated.
- **Mixed Precision**: Benchmarks utilize AMP for faster training without precision loss.
- **Compilation**: `torch.compile` is used to optimize the execution of complex operator graphs.
