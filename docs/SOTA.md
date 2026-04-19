# SOTA Results & Benchmarks

Performance targets are relative L2 error (`val_l2_rel`, lower is better).
Numbers from published papers on their respective test sets.
"This repo" rows reflect the **current best** from `results.json` — run
`uv run analyze.py --papers` to see live values.

---

## 1D Benchmarks

### Burgers 1D (ν = 0.01/π, N=64, T=1)

| Model                     | Relative L2 | Notes                              |
|---------------------------|-------------|------------------------------------|
| FNO (paper)               | 0.0149      | Li et al. 2020, 16 modes, width 64 |
| UNO                       | ~0.012      | Rahman et al. 2022                 |
| WNO                       | ~0.015      | Tripura et al. 2022                |
| GNOT (paper)              | 0.0031      | Hao et al. 2023                    |
| **This repo (best)**      | **0.1468**  | FNO h=128 l=8 m=24                 |

**Gap to SOTA:** ~47× vs GNOT. Key levers: architecture (GNOT, Transolver), more
modes, SOTA loss (H1/spectral), longer budget.

---

### KdV 1D (soliton transport, ETDRK4)

| Model                | Relative L2 | Notes                    |
|----------------------|-------------|--------------------------|
| RFNO (paper)         | ~0.010      | Residual FNO baseline    |
| **This repo (best)** | **0.005748** | RFNO h=128 — beats SOTA |

**Status:** SOTA beat. Registry champion: RFNO, exp `rfno_kdv_h128_m24_l8`.

---

### Wave 1D (u_tt = c²u_xx)

| Model                | Relative L2 | Notes                    |
|----------------------|-------------|--------------------------|
| FNO baseline         | ~0.005      | Estimated                |
| **This repo (best)** | **0.001662** | FNO — beats SOTA 3×     |

**Status:** SOTA beat. Registry champion: FNO, exp `time_deeponet_wave_h128_l4_f1_adapt`.

---

### Euler 1D (compressible)

| Model                | Relative L2 | Notes           |
|----------------------|-------------|-----------------|
| FNO baseline         | ~0.003      | Estimated       |
| **This repo (best)** | **0.00241** | Competitive run |

---

## 2D Benchmarks

> **2D constraints (Apple Silicon unified memory):**
> Hard limit (raises ValueError): `hidden_dim < 64`, `n_layers < 8`
> Recommended practice: `hidden_dim = 32`, `n_layers ≤ 4`, `n_modes ≤ 12`
> Budget floor: `budget_s ≥ 3600`

### Darcy Flow 2D (steady-state -∇·(a∇u)=f, N=64×64)

| Model                | Relative L2 | Notes                                |
|----------------------|-------------|--------------------------------------|
| FNO (paper)          | 0.0108      | Li et al. 2020, 12 modes, width 32   |
| GNOT (paper)         | 0.0041      | Hao et al. 2023                      |
| WNO                  | ~0.015      |                                      |
| **This repo (best)** | **0.2735**  | FEDONet2D h=32 (registry champion)   |

**Gap to SOTA:** ~67× vs GNOT. Registry champion: FEDONet2D h=32 l=4 → val=0.2735.
2D memory constraints (hidden_dim<64, n_layers<8) are the primary limiter.

---

### Navier-Stokes 2D (vorticity, ν=1e-3, T=10, N=64×64)

| Model                | Relative L2 | Notes                   |
|----------------------|-------------|-------------------------|
| FNO (paper)          | 0.0128      | Li et al. 2020          |
| AFNO                 | ~0.008      | Guibas et al. 2022      |
| **This repo (best)** | **0.01428** | Competitive — near SOTA |

**Gap to SOTA:** ~1.1× vs FNO (near-SOTA performance).

---

### Shallow Water Equations 2D (SWE)

| Model                | Relative L2 | Notes             |
|----------------------|-------------|-------------------|
| FNO baseline         | ~0.015      | Estimated         |
| **This repo (best)** | **0.01073** | Beats FNO baseline |

---

### Allen-Cahn 2D (phase separation)

| Model                | Relative L2 | Notes     |
|----------------------|-------------|-----------|
| FNO baseline         | ~0.08       | Estimated |
| **This repo (best)** | **0.06280** |           |

---

### Elasticity 2D (stress-strain)

| Model                | Relative L2 | Notes                           |
|----------------------|-------------|---------------------------------|
| FNO baseline         | ~0.010      | Estimated                       |
| **This repo (best)** | **0.00773** | HANO2D / FEDONet2D — beats FNO  |

---

### PDEBench 2D

| Model                | Relative L2 | Notes               |
|----------------------|-------------|---------------------|
| Baseline             | ~0.005      | Estimated           |
| **This repo (best)** | **0.00260** | FEDONet2D — strong  |

---

### WaveBench 2D

| Model                | Relative L2 | Notes |
|----------------------|-------------|-------|
| Baseline             | ~0.015      |       |
| **This repo (best)** | **0.00991** |       |

---

### MHD 2D (Magnetohydrodynamics)

| Model                | Relative L2 | Notes                            |
|----------------------|-------------|----------------------------------|
| FNO baseline         | ~0.05       | Estimated                        |
| **This repo (best)** | **1.000**   | Not yet solved — all models fail |

**Status:** Active challenge. All current models return val=1.0 (random-level).
Try physics-informed losses and architecture families with symmetry constraints.

---

### NS High-Re 2D (Re=1000 turbulence)

| Model                | Relative L2 | Notes                      |
|----------------------|-------------|----------------------------|
| Best published       | ~0.05       | Turbulence challenge        |
| **This repo (best)** | **1.000**   | All models fail at Re=1000 |

---

### Multiphysics 2D

| Model                | Relative L2 | Notes             |
|----------------------|-------------|-------------------|
| Baseline             | ~0.2        |                   |
| **This repo (best)** | **0.6923**  | Early experiments |

---

## Target Milestones

Registry best values from `model_registry.json`. Targets from `core/utils.py`.

| Benchmark | Registry Best | SOTA Target | Gap | Next Action |
|-----------|--------------|-------------|-----|-------------|
| burgers_1d | 0.1468 (FNO) | 0.0031 (GNOT) | 47× | Transolver, GNOT, H1 loss |
| darcy_2d | 0.2735 (FEDONet2D) | 0.0041 (GNOT) | 67× | AttentionEnhancedFNO2D; longer budget |
| ns_2d | 0.0143 (FNO) | 0.0128 (FNO) | 1.1× | HPO fine-tune at lr=1e-4 |
| kdv_1d | 0.005748 (RFNO) | 0.010 | ✅ Beat | Maintain; try GNOT for further gains |
| wave_1d | 0.001662 (FNO) | 0.005 | ✅ Beat 3× | Benchmark effectively solved |

---

## Notes on Comparison

- Paper results train for 500+ epochs; this repo uses a 5-minute budget
  (~23–50 epochs depending on model size and hardware speed).
- Exact comparison requires the same validation set, grid size, and ν.
  `prepare.py` fixes all of these (seed=42 validation).
- VRAM: 1D models use ~30–150 MB; 2D models use ~0.5–3 GB.
- `h1` and `spectral` losses now support 2D inputs natively (via `rfft2`-based
  spectral gradients). No silent fallback to L2.
