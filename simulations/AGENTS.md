# SIMULATIONS DIRECTORY

**Purpose:** High-fidelity PDE solvers for benchmark generation

## STRUCTURE
```
simulations/
├── __init__.py
├── euler1d.py         # Compressible Euler 1D
├── shallow_water.py   # 2D Shallow Water Equations (swe_2d)
├── allen_cahn.py      # Allen-Cahn phase field 2D
└── ns_etdrk4.py       # Navier-Stokes 2D Re=1000 (ns_hre_2d)
```

## BENCHMARKS
| Benchmark | File | First-run time | Cache |
|-----------|------|----------------|-------|
| euler_1d | euler1d.py | ~5 min | ~6MB |
| swe_2d | shallow_water.py | ~5 min | ~136MB |
| allen_cahn_2d | allen_cahn.py | ~5 min | ~136MB |
| ns_hre_2d | ns_etdrk4.py | ~70 min | ~136MB |

## KEY INSIGHTS
- **ns_hre_2d** — first run ~70 min, then cached
- All data cached to: ~/.cache/sciml_autoresearch/
- Run `uv run prefetch_data.py --skip-slow` to skip ns_hre_2d
- Used by prepare.py to generate training/validation datasets

## CONVENTIONS
- solvers return (u, t) tuples for training data
- registered in benchmarks_ext.py + research_plugins.py
- Safe 2D config: h≤32, l≤4, m≤8, budget_s=480