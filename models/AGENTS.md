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

## WHERE TO LOOK
| Model | File | Status | Best Result |
|-------|------|--------|-------------|
| FNO (baseline) | fno.py | ✓ | 0.1468 burgers+aug |
| RFNO (residual) | fno.py | ✓ 1D only | 0.0020 kdv_1d (SOTA) |
| FFNO | tfno.py | ~ | 0.24 burgers |
| DeepONet | deeponet.py | ✓ | 0.808 |
| S4NO | s4d.py | ✓ | not benchmarked |
| SSNO | ssno.py | ⚠ unstable | val=80.5 at h=128 |
| PINO | pinn.py | ✗ | never use |

## CONVENTIONS
- Export from `models/__init__.py`
- Register in `research_plugins.py` ModelRegistry
- Follow existing patterns (afno.py for reference)
- Use `model_scaffold.py --stub` for gated addition
- 28+ models registered via `ModelRegistry`

## ANTI-PATTERNS
- **AFNO** — wrong bias, 0.50-0.72 on Burgers, skip
- **PINO** — broken for endpoint-only, never queue
- **Transolver2D** — unreliable on 2D benchmarks
- **RFNO on 2D** — ValueError: too many values to unpack
- **SSNO at h≥128** — unstable (val=80.5), use h≤64 l≤4 + lower lr