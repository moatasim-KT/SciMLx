# SIMULATIONS DIRECTORY

**Purpose:** High-fidelity PDE solvers for benchmark generation

## STRUCTURE
```
simulations/
├── __init__.py
├── euler1d.py         # Compressible Euler 1D (euler_1d)
├── shallow_water.py   # 2D Shallow Water Equations (swe_2d)
├── allen_cahn.py      # Allen-Cahn phase field 2D (allen_cahn_2d)
└── ns_etdrk4.py       # Navier-Stokes 2D Re=1000 (ns_hre_2d)
```

## STATUS
| Benchmark | Best Result | Gap to SOTA | Note |
|-----------|-------------|-------------|------|
| `euler_1d` | 0.0024 | **BEATS SOTA** | FNO h=64 l=4 m=16 |
| `swe_2d` | 0.0107 | 5.36× | FNO2D h=32 l=4 |
| `allen_cahn_2d` | 0.0628 | 3.14× | FNO h=64 l=4 |
| `ns_hre_2d` | — | — | ~70 min first run |

## KEY INSIGHTS
- **ns_hre_2d** — first run ~70 min, then cached.
- All data cached to: `~/.cache/sciml_autoresearch/`.
- Run `uv run prefetch_data.py` to ensure all caches exist.
- Used by `prepare.py` for ground-truth data.

## CONVENTIONS
- Solvers return (u, t) tuples for training data.
- Registered in `benchmarks_ext.py` + `research_plugins.py`.
- Safe 2D config: h≤32, l≤4, m≤8, budget_s=480.
