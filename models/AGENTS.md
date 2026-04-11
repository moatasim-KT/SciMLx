# MODELS DIRECTORY

**Purpose:** All neural operator implementations for PDE solving

## STRUCTURE
```
models/
├── __init__.py       # MODEL_REGISTRY (28+ exports)
├── fno.py            # FNO, RFNO, FNO2d, FNO_MC (14KB, core)
├── tfno.py           # TFNO, RTFNO, CPFNO, TFNO2d (16KB)
├── transolver.py     # Transolver, Transolver2D (11KB)
├── gnot.py           # GNOT, GNOT2d (4KB)
├── deeponet.py       # DeepONet, PODDeepONet (4KB)
├── time_deeponet.py  # TimeDeepONet, DualDeepONet (8KB)
├── s4d.py            # S4NO (4KB, state-space)
├── ssno.py           # SSNO (8KB, S4D + spectral conv)
├── afno.py           # AFNO (11KB, broken - skip)
├── wno.py            # WNO (Haar wavelet, non-periodic BCs)
├── hnn.py            # HNN, EnergyFNO (10KB, Hamiltonian)
├── neural_ode.py     # NeuralODE, UDE, LatentODE (12KB)
└── pinn.py           # PINN, PINO (3KB, PINO broken)
```

## STATUS (184+ experiments)
| Model | Status | Notes |
|-------|--------|-------|
| FNO | ✓ | Reliable baseline. Best on burgers+aug. |
| RFNO | ✓ 1D only | Best on kdv_1d (SOTA). Crashes on 2D. |
| UNO | ✓ | Multiscale encoder-decoder. Good potential for shocks. |
| DeepONet | ✓ | Trunk/Branch architecture. |
| SSNO | ⚠ Unstable | val=80.5 at h=128. Needs small config/low LR. |
| AFNO | ✗ Skip | Wrong spectral bias (0.5-0.7 on Burgers). |
| PINO | ✗ Broken | Never use. |

## CONVENTIONS
- Export from `models/__init__.py`.
- Register in `research_plugins.py` ModelRegistry.
- Use `model_scaffold.py --stub` for gated addition.
- 28+ models registered.

## ANTI-PATTERNS
- **AFNO** — skip.
- **PINO** — never use.
- **RFNO on 2D** — crashes.
- **h≥64 or l≥8 on 2D** — OOM.
