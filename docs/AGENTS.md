# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-11
**Commit:** Latest
**Branch:** SciML

## OVERVIEW
Autonomous AI-driven research loop for Scientific Machine Learning (SciML) on Apple Silicon, built on MLX. Explores PDE solver architectures (Neural Operators, PINNs) within fixed 5-min training budget.

## CURRENT STATUS (184+ experiments)
- **SOTA BEATEN**: `kdv_1d` (5×), `wave_1d` (5×), `euler_1d` (6×).
- **CRITICAL GAPS**: `burgers_1d` (47× gap to GNOT), `darcy_2d` (25× gap).
- **NEAR SOTA**: `ns_2d` (1.12× gap, budget=600 key).

## STRUCTURE
```
./
├── models/           # 14 model implementations (FNO, RFNO, DeepONet, S4NO, etc.)
├── papers/           # 20 paper YAMLs with SOTA + suggested experiments
├── simulations/     # 4 high-fidelity PDE solvers
├── docs/             # SOTA.md, LITERATURE.md, TERMINOLOGY.md
├── logs/             # Per-experiment training logs
├── figs/             # Auto-generated plots
├── ui/               # dashboard.html + React app
├── train.py          # Training harness
├── experiments.py    # ExperimentConfig queue (205+ entries)
├── results.json      # DAG of 184+ completed experiments
├── autorun.py        # Subprocess runner with crash recovery
└── CLAUDE.md        # Existing comprehensive guide
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Run experiment | `uv run train.py --model FNO --hidden 128 --layers 8 --modes 24` | 5-min budget |
| Add experiment | `experiments.py` | Append ExperimentConfig |
| View results | `uv run analyze.py --papers` | SOTA gap analysis |
| Dashboard | `uv run uvicorn app:app --port 8000` | Open ui/dashboard.html |
| Add model | `models/` + `research_plugins.py` | Update ModelRegistry |

## CONVENTIONS
- **Never modify `prepare.py`** — defines ground-truth metric.
- **results.json is SSoT** — never hand-edit; use `tracker.py`.
- **ExperimentConfig.name must be unique** — dedup key.
- **darcy_2d is broken** — use `darcy_2d` ( Richardson solver).
- **RFNO is 1D-only** — crashes on 2D benchmarks.
- **2D safe config**: h≤32, l≤4, m≤8, budget_s=480.
- **SSNO** — unstable at h≥128, use h≤64 l≤4 + lower lr.
- **WARMDOWN_RATIO=0.2** — cosine decay starts at 80% of budget.

## ANTI-PATTERNS
- **Never use PINO** — broken for endpoint-only formulation.
- **Never use AFNO** — wrong spectral bias (0.50-0.72 on Burgers).
- **Never use h≥64 or l≥8 on 2D** — OOM/broadcast errors.
- **Never git add -A** — stage only code and results.

## UNIQUE STYLES
- Dual orchestration: Mode A (external agent) + Mode B (agent_loop.py).
- Bayesian HPO auto-loop when queue exhausts.
- Multi-fix crash recovery (batch_size//2 + lr//10 applied together).
- Sentinel kill via `.kill_{name}` files.

## COMMANDS
```bash
uv sync                          # Install dependencies
uv run prefetch_data.py          # Cache PDE datasets (~20 min)
uv run train.py --model FNO --hidden 128 --layers 8 --modes 24
uv run analyze.py --papers       # Results vs SOTA
uv run auto_suggest.py --generate  # Output ready-to-paste ExperimentConfig snippets
uv run autorun.py --priority 1 --commit
uv run uvicorn app:app --reload --port 8000  # Dashboard
```

## RESEARCH PRIORITIES
1. **Close Burgers gap** — best 0.1468, SOTA 0.0031 (47×). Try SSNO, curriculum.
2. **Scale darcy_2d** — try h=32 l=4 m=12 budget=480 (small models only).
3. **Push ns_2d below SOTA** — 0.01428 vs 0.0128; try RFNO h=32 m=8 (if it doesn't crash).
4. **Benchmark SSNO** on burgers_1d (paper target: 0.007).
5. **Run simulation benchmarks** — swe_2d, allen_cahn_2d need more experiments.
