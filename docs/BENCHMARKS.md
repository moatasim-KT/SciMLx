# Per-Benchmark Tuning Guide

Reference for choosing hyperparameters, loss functions, and models for each PDE
benchmark. Read alongside `docs/SOTA.md` for paper references.

---

## 1D vs 2D: Key Differences

| Aspect | 1D Benchmarks | 2D Benchmarks |
|---|---|---|
| Input shape | `(B, 64, C)` | `(B, 64, 64, C)` |
| Budget floor | 1800 s (30 min) | 3600 s (60 min) |
| Memory pressure | Low — `hidden_dim ≤ 256` safe | High — `hidden_dim ≥ 64` causes OOM |
| `n_modes` | Up to 32 typical | Up to 16 per dim; auto-mapped |
| Recommended `hidden_dim` | 64–256 | **32** (hard limit: < 64) |
| Recommended `n_layers` | 4–12 | **2–4** (hard limit: < 8) |

### 2D Memory Safety Limits

Enforced at `ModelRegistry.build()` via `ValueError` — not a soft warning:

```python
# Applies to ALL 2D benchmarks (darcy_2d, ns_2d, swe_2d, allen_cahn_2d,
# ns_hre_2d, mhd_2d, elasticity_2d, wavebench_2d, pdebench_2d,
# multiphysics_2d, radiative_2d)
if hidden_dim >= 64 or n_layers >= 8:
    raise ValueError("Configuration too large for 2D benchmark...")
```

**Hard limit**: `hidden_dim < 64` and `n_layers < 8` — values in this range will not raise.  
**Recommended**: `hidden_dim = 32` and `n_layers ≤ 4` to avoid OOM on 16 GB devices.

If you hit this with `autorun.py`, the r2/r3 retry will automatically fall back
to safe values (`hidden_dim=32, n_layers=2`).

---

## Loss Function × Benchmark Matrix

| Benchmark | Recommended Loss | Reason |
|---|---|---|
| `burgers_1d` | `h1` | Sharp shock fronts benefit from gradient penalty |
| `burgers_nu_001` | `h1_strong` (α=1.0) | Very low viscosity → steep shocks |
| `kdv_1d` | `l2_rel` | Smooth solitons; gradient penalty unnecessary |
| `wave_1d` | `l2_rel` | Smooth wave propagation |
| `euler_1d` | `h1_adaptive` | Multi-channel, varying gradient scales |
| `darcy_2d` | `h1` | Variable-coefficient elliptic; sharp conductivity boundaries |
| `ns_2d` | `spectral` | Turbulent cascades; spectral weighting helps high-k modes |
| `ns_hre_2d` | `spectral` | High Reynolds → energy in all wavenumbers |
| `swe_2d` | `l2_rel` | Linearized — smooth solution |
| `allen_cahn_2d` | `h1` | Sharp interface between phases |
| `elasticity_2d` | `l2_rel` | Smooth displacement fields |
| `wavebench_2d` | `l2_rel` | Standard wave propagation |
| `mhd_2d` | `h1_adaptive` | Multi-field; unknown gradient balance |
| `multiphysics_2d` | `h1_adaptive` | Coupled fields with different scales |

---

## Budget Recommendations

Realistic wall-time estimates on M2 Pro with 16 GB:

| Category | Minimum | Full Run | Notes |
|---|---|---|---|
| 1D, small model (h=64, l=4) | 30 min | 1–2 hr | Fast iteration |
| 1D, large model (h=256, l=12) | 45 min | 2–4 hr | Deep networks |
| 2D, safe config (h=32, l=4) | 60 min | 2–4 hr | Memory-safe |
| 2D, attention models | 90 min | 4–6 hr | Transolver, GNOT are ~3× slower |

Budget floors are enforced by `autorun.py` (not configurable via CLI):
- 1D: `BUDGET_FLOOR_1D = 1800`
- 2D: `BUDGET_FLOOR_2D = 3600`

---

## Benchmark Tuning Cards

### `burgers_1d`

**PDE:** u_t + u·u_x = ν·u_xx, ν = 0.01/π  
**SOTA target:** 0.0031 (GNOT, Hao et al. 2023 — tightest published)  
**Current champion:** FNO, h=128, l=8, m=24 → val=0.1468 (gap: 47×)

**Recommended first experiment:**
```yaml
model: FNO
hidden_dim: 128
n_layers: 8
n_modes: 24
lr: 1e-3
loss_type: h1
h1_alpha: 0.1
budget_s: 3600
```

**Known failure modes:**
- AFNO on burgers often collapses (`wrong_inductive_bias` — periodic modes mismatch)
- `n_modes > 28` with `l > 8` leads to gradient instability; use `grad_clip: 0.5`

**What to try next when stuck:**
- Curriculum training (`curriculum: true`) helps stabilize early epochs
- EMA (`ema_decay: 0.999`) often gives 5–10% improvement on val at no training cost

---

### `burgers_nu_001`

**PDE:** Burgers with ν = 0.001 (near-inviscid, strong shocks)  
**SOTA target:** no published reference  
**Current champion:** WNO, h=128, l=8 → val=0.6338 (largely unsolved)

**Notes:** Low viscosity creates near-discontinuous solutions that spectral methods
struggle with. Wavelet-based models (WNO) outperform Fourier-based models here due
to better spatial localization.

**Recommended first experiment:**
```yaml
model: WNO
hidden_dim: 128
n_layers: 8
loss_type: h1_strong
budget_s: 3600
```

---

### `kdv_1d`

**PDE:** u_t + u·u_x + u_xxx = 0 (Korteweg-de Vries, soliton dynamics)  
**SOTA target:** 0.010 (estimated)  
**Current champion:** RFNO, h=128, m=24, l=8 → **val=0.005748** (beat SOTA)

**Notes:** RFNO with high modes and depth is the clear winner. Solitons are smooth
periodic structures — ideal for Fourier methods.

**Recommended first experiment:**
```yaml
model: RFNO
hidden_dim: 128
n_layers: 8
n_modes: 24
lr: 1e-3
loss_type: l2_rel
budget_s: 3600
```

---

### `wave_1d`

**PDE:** u_tt = c²·u_xx (second-order wave equation)  
**SOTA target:** 0.005 (estimated)  
**Current champion:** FNO → **val=0.001662** (beat SOTA 3×)

**Notes:** Smooth propagating waves. FNO already beats SOTA significantly — this
benchmark is effectively solved for the standard formulation.

---

### `euler_1d`

**PDE:** 1D Euler equations (multi-channel: density, momentum, energy)  
**SOTA target:** 0.003 (estimated)  
**Current champion:** FEDONet2D → val=0.0024 (beat SOTA)

**Notes:** Multi-channel benchmark; `FNO` auto-routes to `FNO_MC` in `train.py`.
Use `h1_adaptive` to handle varying gradient scales across channels.

---

### `darcy_2d`

**PDE:** -div(a(x)·∇u) = f, where a(x) is a random field  
**SOTA target:** 0.0041 (GNOT, Hao et al. 2023 — tightest published)  
**Current champion:** FEDONet2D, h=32, l=4 → val=0.2735 (gap: 67×)

**Notes:** The hardest remaining benchmark. The variable-coefficient operator a(x)
requires learning spatially-varying responses. FEDONet2D currently wins by combining
frequency and spatial branches.

**Recommended first experiment:**
```yaml
model: FEDONet2D
hidden_dim: 32
n_layers: 4
n_modes: 12
lr: 2e-3
loss_type: h1
h1_alpha: 0.1
budget_s: 7200
```

**What to try:**
- Transolver2D with physics slices may capture the heterogeneous a(x) better
- Augmentation (`augment: true`) helps with spatial generalization
- Longer budget (2–3 hr) consistently improves Darcy results

---

### `ns_2d`

**PDE:** 2D Navier-Stokes (vorticity formulation), semi-implicit spectral solver  
**SOTA target:** 0.0128 (FNO, Li et al. 2021)  
**Current champion:** FNO, h=32, m=8, l=4 → val=0.0143 (near SOTA)

**Notes:** 2D memory limits make this challenging. `hidden_dim=32` is the safe
maximum. Spectral loss helps recover turbulent high-k energy.

**Recommended first experiment:**
```yaml
model: FNO2D
hidden_dim: 32
n_layers: 4
n_modes: 8
lr: 1e-3
loss_type: spectral
budget_s: 7200
```

---

### `ns_hre_2d`

**PDE:** Navier-Stokes Re=1000 (high-Reynolds turbulence)  
**SOTA target:** 0.050 (estimated)  
**Current champion:** unsolved (val=1.000 — no architecture has converged)

**Notes:** High Reynolds number produces chaotic, multi-scale vortical structures.
No architecture has converged on this benchmark yet. Attention-based models
(Transolver2D, GNOT) are the recommended starting point given their multi-scale
sensitivity.

---

### `swe_2d`

**PDE:** Shallow Water Equations (linearized gravity waves)  
**SOTA target:** 0.015 (estimated FNO baseline)  
**Current champion:** FNO → val=0.0107 (beat SOTA)

---

### `allen_cahn_2d`

**PDE:** Allen-Cahn phase field: u_t = ε²·Δu + u − u³  
**SOTA target:** 0.080 (estimated)  
**Current champion:** FNO → val=0.0628 (beat SOTA)

**Notes:** Sharp but smooth phase interfaces. H1 loss helps track the interface
location accurately.

---

### `elasticity_2d`

**PDE:** Linear elasticity (displacement under load)  
**SOTA target:** 0.010  
**Current champion:** FEDONet2D, h=32, l=2, m=8 → val=0.0077 (beat SOTA)

---

### `wavebench_2d`

**PDE:** 2D wave propagation  
**SOTA target:** 0.015  
**Current champion:** SNO2D, h=32, l=2, m=8 → val=0.0099 (beat SOTA)

---

### `pdebench_2d`

**PDE:** PDEBench comprehensive 2D suite  
**SOTA target:** 0.005  
**Current champion:** FEDONet2D, h=32, l=2, m=8 → val=0.0026 (beat SOTA)

---

### `mhd_2d`

**PDE:** 2D Magnetohydrodynamics (2-channel: vorticity + magnetic potential)  
**SOTA target:** 0.050  
**Current champion:** unsolved (val=1.000)

**Notes:** Multi-channel 2D benchmark with coupled physical fields. No successful
run yet. Recommended approach: start with a 2-channel FNO2D variant.

---

### `multiphysics_2d`

**PDE:** Coupled multi-physics simulation  
**SOTA baseline:** ~0.200  
**Current champion:** HybridDecoderDeepONet2D → val=0.6923 (gap)

**Notes:** The coupled nature makes single-physics architectures struggle. Hybrid
branch-trunk architectures (DeepONet family) are the recommended starting point.

---

## Cross-Benchmark Strategy

When starting on a new benchmark with no prior runs:

1. **First run:** FNO with `hidden_dim=64` (1D) or `32` (2D), `n_layers=4`,
   `n_modes=16`, `loss_type=l2_rel`, default budget
2. **If val > 0.3:** Switch loss to `h1`; try `WNO` if the PDE has shocks
3. **If val stagnating between 0.1–0.3:** Increase depth (`n_layers+2`) or modes
   (`n_modes+4`); try `RFNO` which handles depth better than `FNO`
4. **If val < 0.05:** Fine-tune: `lr=1e-4`, `resume_from: champion:<benchmark>`,
   `snapshot_ensemble: true`
5. **If 2D and OOM:** Reduce to `hidden_dim=24`, `n_layers=2` — the r2/r3 retry
   will do this automatically via `autorun.py`
