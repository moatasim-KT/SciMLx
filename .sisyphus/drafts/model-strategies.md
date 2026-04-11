# Draft: Model-Usage Strategy

## Scope
- Keep **PINO, AFNO, Transolver2D, RFNO** in the repository.
- Identify *exact* scenarios where each model is known to work reliably.
- Record the hyper‑parameter ranges and benchmark targets that make them succeed.
- Note any guard‑rails (dimensionality, loss types, budget limits) that must be enforced when queuing new experiments.

## Resolved Questions

| # | Question | Answer | Implication |
|---|----------|--------|--------------|
| 1 | PINO – endpoint-only benchmarks? | **No** | Restrict to full‑field losses only; add guard‑rail in experiments.py |
| 2 | AFNO – diffusion/low‑freq PDEs? | **No** | Skip AFNO exploration; keep as "skip" |
| 3 | Transolver2D – mini‑grid test? | **Yes** | Add experiment: h=16, l=4, budget=300 s |
| 4 | RFNO – pseudo‑2D tricks? | **Yes** | Explore reshaping 2D→1D for RFNO |

## New Scope (Added)

### A. Novel Benchmark Ideas
| # | Candidate Benchmark | Domain | Why It’s Novel | Targeted Bottleneck |
|---|---------------------|--------|----------------|----------------------|
| 1 | **WaveBench** (2024) | Wave propagation (time‑varying & harmonic) | Provides 24 diverse wave‑equation datasets missing from classic fluid‑only suites. | Tests spectral bias on high‑frequency wave modes. |
| 2 | **Multiphysics Bench** (2024) | Multi‑physics coupling (fluid‑structure, thermal‑mechanical, etc.) | First systematic coupling of heterogeneous physics types into one benchmark suite. | Stresses operator on cross‑field dependencies. |
| 3 | **PDEBench** (2022) – Expanded set | Broad PDE family (advection, diffusion‑reaction, compressible Navier‑Stokes) | 13 PDE categories + many parameter sweeps; includes elasticity, radiative transfer. | Serves as stress test for multiscale and embedded‑boundary handling. |
| 4 | **Elasticity & Solid‑Mechanics (2‑D/3‑D)** | Solid‑mechanics deformation | Rarely used in neural‑operator work; requires tensor‐valued outputs. | Challenges multi‑component output handling. |
| 5 | **Radiative Transfer (2‑D)** | Astrophysics / medical imaging | Integro‑differential form; highly forward‑peaked scattering. | Promotes hybrids that embed angular attention. |

These benchmarks are **not** currently in your core fluid‑dynamics list and expose new failure modes (e.g., high‑frequency wave components, multiphysics coupling).

### B. Novel Hybrid Architectures
| # | Hybrid Concept | Core Mix | Literature Anchor | Target Bottleneck |
|---|----------------|----------|-------------------|--------------------|
| 1 | **Hybrid Decoder‑DeepONet** (2023) | FNO‑style spatial encoder + DeepONet trunk | “Hybrid Decoder‑DeepONet… handling unaligned data” (arXiv 2308.09274) | Aligns mismatched sensor grids → better on sparse sensor data. |
| 2 | **Attention‑Enhanced FNO** (2025) | FNO + self‑attention over Fourier modes | “Attention‑Enhanced FNO” (AIP 2025‑03‑10) | Handles sharp shock/turbulence gradients. |
| 3 | **Hybrid FNO‑DeepONet** (2024‑2025) | FNO branch + DeepONet trunk | “Hybrid FNO‑DeepONet” (EmergentMind, 2024‑2025) | Generalizes across parameter regimes. |
| 4 | **Fourier‑Enhanced DeepONet (FEDONet)** | Spectral conv layers added to DeepONet branches | “Fourier‑Enhanced DeepONet” (EmergentMind, 2025) | Improves low‑frequency fidelity → better for wave/Burgers. |
| 5 | **Hierarchical Attention Neural Operator (HANO)** (2023) | Multi‑scale self‑attention + spectral conv | “Hierarchical Attention Neural Operator” (arXiv 2210.10890) | Mitigates spectral bias for multiscale PDEs. |
| 6 | **Spectral Neural Operator (SNO)** (2024) | Chebyshev/Fourier back‑end, no aliasing | “Spectral Neural Operators” (arXiv 2404.02) | Eliminates aliasing → better for spectral‑bias‑sensitive problems. |
| 7 | **VSMNO** (Variational Spectral Mixture) | Mixture of spectral patterns from FNO/DNO/MRI | “VSMNO” (OpenReview 2024) | Leverages learned spectral patterns across operators → enhances multiscale PDE handling. |

These hybrids explicitly **target** the bottlenecks identified in Section A (e.g., high‑frequency wave components, multiphysics coupling, spectral bias on shocks).

## Next Steps
1. **Validate candidate benchmarks** – pick 3–5 from the Novel Benchmark Ideas table that best match your research gaps.  
2. **Select 2–3 hybrid models** from the Hybrid Architectures table that promise the biggest gain on those bottlenecks.  
3. **Finalize guard‑rails** for each model as documented above.  
4. **Proceed to Metis gap review** to ensure the selected combos expose no hidden constraints.  

---

## SELECTED ITEMS (APPROVED)

### Benchmarks (All 5)
| # | Benchmark | Domain | Target Bottleneck |
|---|-----------|--------|-------------------|
| 1 | **WaveBench** (2024) | Wave propagation (time‑varying & harmonic) | Spectral bias on high‑frequency wave modes |
| 2 | **Multiphysics Bench** (2024) | Multi‑physics coupling | Cross‑field dependencies |
| 3 | **PDEBench** (expanded) | Advection, diffusion‑reaction, compressible NS | Multiscale & embedded‑boundary handling |
| 4 | **Elasticity & Solid‑Mechanics** | Solid‑mechanics deformation | Multi‑component output handling |
| 5 | **Radiative Transfer** | Integro‑differential scattering | Angular attention for forward‑peaked scattering |

### Hybrid Models (All 7)
| # | Model | Core Mix | Target Bottleneck |
|---|-------|----------|-------------------|
| 1 | **Hybrid Decoder‑DeepONet** | FNO spatial encoder + DeepONet trunk | Sparse sensor data, unaligned grids |
| 2 | **Attention‑Enhanced FNO** | FNO + self‑attention over Fourier modes | Sharp shock/turbulence gradients |
| 3 | **Hybrid FNO‑DeepONet** | FNO branch + DeepONet trunk | Parameter regime generalization |
| 4 | **FEDONet** | Spectral conv in DeepONet branches | Low‑frequency fidelity (wave/Burgers) |
| 5 | **HANO** | Multi‑scale self‑attention + spectral conv | Spectral bias for multiscale PDEs |
| 6 | **SNO** | Chebyshev/Fourier back‑end, no aliasing | Aliasing in spectral‑bias‑sensitive problems |
| 7 | **VSMNO** | Mixture of spectral patterns (FNO/DNO/MRI) | Multiscale PDE handling |

---

*Approved selections captured. Proceeding to unified work plan.*