# PROJECT KNOWLEDGE BASE

**Generated:** 2025-01-13
**Commit:** f582012
**Branch:** SciML

## OVERVIEW
Autonomous AI-driven research loop for Scientific Machine Learning (SciML) on Apple Silicon, built on MLX. Explores PDE solver architectures (Neural Operators, PINNs) within fixed 5-min training budget.

## STRUCTURE
```
./
├── models/           # 14 model implementations (FNO, RFNO, DeepONet, S4NO, etc.)
├── papers/           # 15 paper YAMLs with SOTA + suggested experiments
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
- **Never modify `prepare.py`** — defines ground-truth metric
- **results.json is SSoT** — never hand-edit; use `tracker.py`
- **ExperimentConfig.name must be unique** — dedup key
- **darcy_2d is broken** — use darcy_2d only
- **RFNO is 1D-only** — crashes on 2D with ValueError
- **2D safe config**: h≤32, l≤4, m≤8, budget_s=480
- **SSNO** — unstable at h≥128, use h≤64 l≤4 + lower lr
- **WARMDOWN_RATIO=0.2** — cosine decay starts at 80% of budget

## ANTI-PATTERNS (THIS PROJECT)
- **Never use PINO** — broken for endpoint-only formulation
- **Never use AFNO** — wrong spectral bias (0.50-0.72 on Burgers)
- **Never use h≥64 or l≥8 on 2D** — OOM/broadcast errors
- **Never git add -A** — stage only train.py, models/, experiments.py, results.tsv

## UNIQUE STYLES
- Dual orchestration: Mode A (external agent) + Mode B (agent_loop.py)
- Bayesian HPO auto-loop when queue exhausts
- Multi-fix crash recovery (batch_size//2 + lr//10 applied together)
- Sentinel kill via .kill_{name} files

## COMMANDS
```bash
uv sync                          # Install dependencies
uv run prefetch_data.py          # Cache PDE datasets (~20 min)
uv run prefetch_data.py --skip-slow  # Skip ns_hre_2d (~2 min)
uv run train.py --model FNO --hidden 128 --layers 8 --modes 24
uv run analyze.py --papers       # Results vs SOTA
uv run auto_suggest.py --generate  # Output ready-to-paste ExperimentConfig snippets
uv run autorun.py --priority 1 --commit
uv run uvicorn app:app --reload --port 8000  # Dashboard
```

## NOTES
- Data cached at: ~/.cache/sciml_autoresearch/
- 184+ experiments completed, 3 benchmarks beat SOTA (kdv_1d, wave_1d, euler_1d)
- burgers_1d has 9.8× gap to SOTA (priority target)
- SSNO unstable at h≥128 — start with h≤64 l≤4 + lower lr if exploring
- Trainer writes rolling loss to `.vram_telemetry`

## RESEARCH PRIORITIES
1. **Close Burgers gap** — best 0.1468, SOTA 0.0149 (9.8×). SSNO paper claims 0.007.
2. **Scale darcy_2d** — try h=32 l=4 m=12 budget=480 (small models only)
3. **Push ns_2d below SOTA** — 0.01428 vs 0.0128; try RFNO h=32 m=8
4. **Benchmark SSNO** on burgers_1d (paper target: 0.007)
5. **Run simulation benchmarks** — swe_2d, allen_cahn_2d need more experiments