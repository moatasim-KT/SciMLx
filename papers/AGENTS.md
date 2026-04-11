# PAPERS DIRECTORY

**Purpose:** Paper registry with SOTA results and suggested experiments

## STRUCTURE
```
papers/
├── fno_2020.yaml          # Fourier Neural Operator (0.0128 on NS2D)
├── rfno_2024.yaml        # Residual FNO (0.010 on KdV)
├── gnot_2023.yaml       # GNOT (0.0031 on Burgers - our primary target)
├── augmentation_2023.yaml # Key for Burgers win (+aug)
└── ... (16 more)
```

## STATUS
| Paper | SOTA Target | Our Best | Gap |
|-------|-------------|----------|-----|
| GNOT (Burgers) | 0.0031 | 0.1468 | 47.3× |
| GNOT (Darcy) | 0.0041 | 0.1041 | 25.4× |
| FNO (NS2D) | 0.0128 | 0.0142 | 1.12× |

## KEY INSIGHTS
- **gnot_2023.yaml** — defines the 0.0031 target for Burgers, 0.0041 for Darcy.
- **rfno_2024.yaml** — baseline for KdV win.
- 20 papers registered; many pending ideas in `suggested_experiments`.

## COMMANDS
| Task | Command |
|------|---------|
| SOTA gap report | `uv run paper_registry.py --gaps` |
| Pending ideas | `uv run paper_registry.py --pending` |
| Auto-suggestions | `uv run auto_suggest.py` |

## CONVENTIONS
- Status: pending/implemented.
- `auto_suggest.py` reads `results.json` to compute dynamic wins.
