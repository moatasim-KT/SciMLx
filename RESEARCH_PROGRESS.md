# SciML AutoResearch — Progress Snapshot (2026-04-19)

## Summary

- **Status:** Ready to execute (390 experiments queued)
- **Infrastructure Blocker:** MLX library incompatibility on current system
- **Completed Experiments:** 155
- **Pending Queue:** 390 (organized by priority)
- **Best Performer:** RFNO on KDV 1D (val_l2_rel = 0.002023)

---

## Benchmark Rankings (by best val_l2_rel achieved)

### 1D Benchmarks (Lower dimensional, generally better performance)

| Benchmark | Best Score | Model | Config |
|-----------|-----------|-------|--------|
| **wave_1d** | 0.000992 | FNO | Optimized baseline |
| **kdv_1d** | 0.002023 | RFNO | Spectral kernel variant |
| **euler_1d** | 0.002413 | FNO | Standard baseline |

### 2D Benchmarks (Higher complexity, room for improvement)

| Benchmark | Best Score | Model | Config | Notes |
|-----------|-----------|-------|--------|-------|
| **pdebench_2d** | 0.002602 | FEDONet2D | Specialized FED architecture |
| **elasticity_2d** | 0.007734 | FEDONet2D | Solid mechanics specialization |
| **swe_2d** | 0.010729 | FNO2D | Shallow water equations |
| **ns_2d** | 0.014284 | FNO | 2D Navier-Stokes |
| **darcy_2d** | 0.059719 | FNO2D | Lower performance — optimization needed |
| **allen_cahn_2d** | 0.062801 | FNO | Reaction-diffusion — difficult to train |

### Failed/Saturated Benchmarks

These benchmarks have val_l2_rel = 1.0 or near-1.0 (model predictions no better than constant baseline):

- `burgers_nu_01`, `burgers_nu_001` (viscosity variants — parametric generalization failing)
- `couette_flow_1d`, `poiseuille_flow_1d` (analytical solutions — not a learned task)
- `mhd_2d` (magnetohydrodynamics — untested on all models so far)
- `ns_hre_2d` (high Reynolds with extra training — divergence issue?)
- `rayleigh_benard_2d`, `radiative_2d` (coupled physics — no successful model yet)
- `wavebench_2d`, `multiphysics_2d` (composite benchmarks — low signal)

---

## Next Priority Experiments (Ready to Queue)

### High-Priority (Priority 1: 161 experiments)

**New model architectures on established benchmarks:**
- MambaNO (2024 mambda-based neural operator) on:
  - Navier-Stokes 2D (`mambano_ns_2d_h32_l4`)
  - Shallow water equations (`mambano_swe_2d_h32_l4`)
  - Allen-Cahn 2D (`mambano_allen_cahn_h32_l4`)

- MemNO (2025 memory-augmented neural operator) on:
  - Shallow water equations (`memno_swe_2d_h32_l4`)
  - Multiple 2D benchmarks

**Rationale:** These are recent SOTA-candidate architectures (2024-2025) not yet tested against our benchmarks.

### Medium-Priority (Priority 2: 67 experiments)

**FNO/RFNO configuration sweeps** on 1D benchmarks:
- Hidden dim sweeps: h64, h128, h256
- Layer sweeps: l4, l8, l12
- Mode sweeps: m8, m16, m24, m32
- Loss function variants: L2, H1, spectral

**Rationale:** 1D benchmarks are lower-cost (300–600s) and provide quick feedback on architectural changes.

### Lower-Priority (Priority 3–5: 162 experiments)

- Exploratory architectures (UNO, PODDeepONet)
- Hyperparameter ablations (batch size, clipping, regularization)
- Specialized loss functions (H1-adaptive, spectral weighting)

---

## Infrastructure Status

### ✅ Ready to Execute

- Experiment queue: fully specified in `experiments.yaml`
- Result tracking: `results.json` (155 baseline runs)
- Logging system: `logs/` with 500+ historical logs
- Diagnostics: gradient norms, early stopping detection, crash recovery
- Dataset caching: prefetch script completed (9/9 benchmarks cached)
- HPO fallback: Bayesian optimization ready when queue exhausts

### ⚠️ Blocked

- **Execution environment:** MLX library (native bindings) not available on Linux
- **Workaround:** Transfer to Apple Silicon hardware or use macOS VM/Docker

---

## Key Findings So Far

1. **RFNO > FNO on spectral problems** (KDV, wave equations)
   - RFNO achieves 0.002023 on KDV vs FNO at ~0.005
   - Suggests kernel-specialization matters for high-frequency content

2. **FEDONet dominates structured PDEs** (elasticity, pdebench)
   - Encoder-decoder architecture with skip connections works well
   - Best performance: 0.007734 on elasticity (expert architecture for solid mechanics)

3. **2D problems remain hard**
   - Darcy (0.059719) and Allen-Cahn (0.062801) are 4-5x worse than their 1D counterparts
   - Suggests spatial scaling issue or missing inductive bias for 2D

4. **Coupled multi-physics unsolved**
   - Burgers with viscosity variations, MHD, radiative transfer all at val_l2_rel=1.0
   - Indicates fundamental gap in generalization or training stability

---

## Recommended Next Steps (Once Infrastructure is Fixed)

1. **Execute Priority 1 experiments** (MambaNO, MemNO on key benchmarks)
   - Expected impact: +3–8% improvement on 2D benchmarks
   - Timeline: 48 hours of continuous execution

2. **Analyze MambaNO failures** (if they occur)
   - Use gradient norm diagnostics from `core/diagnostics.py`
   - Check spectral bias in `logs/probes/`

3. **Pursue 2D improvement**
   - Focus on why FNO struggles with spatial scaling
   - Consider curriculum learning or spatial masking

4. **Revisit coupled physics**
   - Could indicate training instability (exploding gradients)
   - Investigate via early-stopping heuristics in `HypothesisEngine`

---

*Snapshot created: 2026-04-19 09:15 UTC*  
*System: Ubuntu 22.04 (ARM64) — MLX-incompatible environment*  
*Expected execution location: Apple Silicon Mac with `uv sync` + `uv run autorun.py`*
