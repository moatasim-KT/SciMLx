# SciML AutoResearch — Agent Field Guide

This file is the primary reference for an **external AI agent** (Claude Code,
Gemini, GPT-4, human researcher) driving the research loop in **Mode A**.

For a quick overview see `CLAUDE.md` or `GEMINI.md`. This file goes deeper:
the full tool chain, interpretation guide, and what to do next.

---

## Orchestration Modes

### Mode A · External Agent (this file)

The agent reads state, reasons about it, edits files, and runs commands.
Full control. Best for: novel architecture ideas, hypothesis-driven branching,
human intuition, careful steering.

```bash
# Canonical session:
uv run analyze.py --papers          # 1. read current state
uv run auto_suggest.py              # 2. get ranked suggestions
# 3. edit experiments.py or models/*.py
uv run autorun.py --priority 1 --commit   # 4. run
# 5. go to 1
```

### Mode B · agent_loop.py (optional, automated)

An in-process loop that proposes + appends + runs new experiments with no
human input. Use after a Mode A session to saturate the queue overnight.

```bash
uv run agent_loop.py --dry-run       # see what it would do
uv run agent_loop.py --top 5         # propose 5 new configs
uv run agent_loop.py --run           # propose + run top-3
uv run agent_loop.py --no-hpo        # skip Bayesian HPO, use heuristics only
```

The `--auto` flag wires both modes together: when the queue empties, `autorun.py`
invokes `agent_loop.py --top 5` and recurses if new experiments are generated.

```bash
# Guarded overnight run
uv run autorun.py --auto --commit --max-auto-experiments 20 --max-auto-time 10800
```

Both modes share the same `experiments.py` queue and `results.json` lineage.
You can pause/resume either from the dashboard or terminal:

```bash
# Pause (autorun/agent_loop checks this before each experiment)
curl -X POST http://localhost:8000/api/pause
# or:
touch .autorun_pause

# Resume
curl -X POST http://localhost:8000/api/resume
# or:
rm .autorun_pause

# Kill a specific running experiment (terminates within 2s)
curl -X POST http://localhost:8000/api/kill/<name>
```

---

## Tool-by-tool guide

### 1. Read state: `analyze.py`

```bash
uv run analyze.py           # results summary, model ranking
uv run analyze.py --papers  # adds SOTA gap from papers/*.yaml
```

Output tells you: best val_l2_rel per benchmark, model comparison, whether
any run beat SOTA, hyperparameter sensitivity.

### 2. Get suggestions: `auto_suggest.py`

```bash
uv run auto_suggest.py              # full ranked list
uv run auto_suggest.py --gaps       # SOTA gap table only
uv run auto_suggest.py --generate   # print ready-to-paste ExperimentConfig snippets
```

Suggestion sources (in priority order):
1. **Empirical wins** — configs that worked well on related benchmarks (loaded dynamically from results.json, always up-to-date)
2. **Paper ideas** — `papers/*.yaml` entries with `status: pending`
3. **Spectral diagnostic feedback** — if `diag_high_freq_error` is high,
   suggests increasing n_modes or switching to spectral loss
4. **Cross-benchmark transfer** — winning config on kdv_1d suggested on wave_1d,
   winning config on burgers_1d suggested on ns_2d, etc.

### 3. Run the queue: `autorun.py`

```bash
uv run autorun.py --priority 1 --commit      # run all priority-1 pending
uv run autorun.py --priority 2 --commit      # run priority 1 + 2
uv run autorun.py --auto --commit            # run + re-suggest loop until queue empty
uv run autorun.py --auto --commit \
  --max-auto-experiments 20 \
  --max-auto-time 10800                      # guarded overnight run
uv run autorun.py --dry-run                  # preview without running
uv run autorun.py --model RFNO               # run only RFNO experiments
uv run autorun.py --benchmark kdv_1d         # run only kdv_1d experiments
```

What autorun does:
- Skips experiments whose `name` is already in `results.json`
- Writes full stdout to `logs/<name>.log`
- Parses `val_l2_rel`, `peak_vram_mb`, `diag_*`, `inspect_id` from log
- Classifies crashes: OOM / NaN-Inf / ImportError / ValueError / Timeout / UnknownError
- **Multi-fix retry**: collects ALL applicable fixes and merges them — OOM+NaN → both `batch_size//2` AND `lr//10` in one retry
- Polls every 2s for `.kill_{name}` sentinel; terminates process if found
- With `--commit`: stages code and `results.tsv` and commits each kept result

### 4. Add experiments: `experiments.py`

Every experiment is a `ExperimentConfig` in the `EXPERIMENTS` list:

```python
ExperimentConfig(
    name="my_exp",             # unique string — dedup key
    benchmark="burgers_1d",   # see Benchmark Catalog
    model="FNO",               # see Model Zoo
    hidden_dim=128,
    n_layers=8,
    n_modes=24,
    budget_s=300,              # seconds (300 for 1D, 480 for 2D)
    priority=1,                # 1=highest; autorun runs lowest first
    parent_name="",            # name of experiment this branches from (for DAG)
    rationale="Why this experiment?",
)
```

Priority conventions:
- `1` — high value, run immediately
- `2` — medium value, run after priority-1 queue is clear
- `3` — speculative, auto-generated, run when queue is otherwise empty

**2D benchmark constraint**: always use `h≤32, l≤4, m≤12, budget_s=480`.
Models larger than h=64 or l=8 crash on all 2D benchmarks (OOM/broadcast errors
on Apple Silicon unified memory). **RFNO is 1D-only.**

### 5. Inspect failures: `hypothesis.py` + `diagnostics.py`

When a run underperforms:

```python
from hypothesis import HypothesisEngine
engine = HypothesisEngine()

# Analyze a benchmark's failure pattern
report = engine.analyze_benchmark("burgers_1d")
# Returns: best_val, trend, hp_importance, model_ranking

# Get a concrete config suggestion
intervention = engine.suggest_intervention("burgers_1d", best_val=0.1468)
# Returns: {model, hidden_dim, n_layers, n_modes, rationale, paper_ref}
```

HypothesisEngine detects:
- **Spectral bias** (`diag_high_freq_error` high) → increase n_modes or use spectral loss
- **Shock/gradient errors** → switch to UNO or WNO
- **Gradient collapse** → switch to RFNO (pre-LN residual)
- **Step-time saturation** → reduce hidden_dim

Spectral diagnostics (FFT-based, logged during training):
```bash
grep "^diag_" logs/<name>.log
# diag_high_freq_error: 0.312
# diag_low_freq_error: 0.021
# diag_spectral_gap: 0.291
```

### 6. Adaptive HPO: `bayesian_hpo.py`

```bash
# See top-5 configs predicted by GP surrogate
uv run bayesian_hpo.py --benchmark burgers_1d --top 5

# Next EI-optimal config
uv run bayesian_hpo.py --benchmark burgers_1d

# Multi-objective: accuracy + memory
uv run bayesian_hpo.py --benchmark burgers_1d --multi --pareto --mem-weight 0.3
```

In Python (used by agent_loop.py):
```python
from bayesian_hpo import BayesianHPO
hpo = BayesianHPO("burgers_1d", objectives=[
    ("val_l2_rel", 1.0, "minimize"),
    ("memory_gb",  0.3, "minimize"),   # optional secondary objective
])
hpo.load_history()      # seeds from results.json automatically
cfg = hpo.ask()         # returns {hidden_dim, n_layers, n_modes, lr}
hpo.tell_multi(cfg, {"val_l2_rel": 0.15, "memory_gb": 0.12})
front = hpo.pareto_front()   # Pareto-optimal configs
```

The HPO is now wired into the `--auto` loop automatically. When the queue exhausts,
`autorun.py` invokes `agent_loop.py --top 5` (which calls HPO) and recurses.

### 7. New model code generation: `model_scaffold.py`

```bash
# Generate a stub
uv run model_scaffold.py --stub MyFNO --base FNO --notes "Add self-attention after each block"
# → writes models/myfno.py

# Edit the stub, then validate + register in one step
uv run model_scaffold.py --register MyFNO models/myfno.py \
    --benchmarks burgers_1d kdv_1d

# The register command:
# 1. Validates syntax → import → smoke test (shape check on 1D and 2D inputs)
# 2. Copies to models/
# 3. Adds export to models/__init__.py
# 4. Registers in research_plugins.py MODEL_REGISTRY
# 5. Appends ExperimentConfig entries to experiments.py
```

### 8. Dashboard: `app.py` + `ui/dashboard.html`

```bash
uv run uvicorn app:app --reload --port 8000
# Then open ui/dashboard.html in a browser
```

Dashboard panels:
- **Left sidebar** — per-benchmark SOTA comparison bars, model comparison chart, live training quick stats
- **Results tab** — all 184 completed experiments, sortable/searchable/filterable
- **Queue tab** — all pending experiments with priority, config, rationale
- **Lineage DAG tab** — SVG graph of experiment parent→child relationships
- **Right inspector** — click any experiment for details, log, spectral diagnostics, parent comparison
- **Active strip** — running experiment with progress bar, rolling loss sparkline, Kill button

API endpoints (all re-load results.json fresh on each call):
```
GET  /api/experiments          all completed experiments
GET  /api/experiment/{id}      detail + inspect_url
GET  /api/sota                 SOTA targets + our best + ratio per benchmark
GET  /api/queue                all pending experiments with full config
GET  /api/logs/{name}?tail=N   last N lines of a log file
GET  /api/status               VRAM, progress, remaining_s, step, loss, loss_history, paused flag
POST /api/pause                create .autorun_pause sentinel
POST /api/resume               remove .autorun_pause sentinel
POST /api/inject               add experiment to .injected_experiments.json
POST /api/priority             override priority via .priority_overrides.json
GET  /api/lineage              nodes + links for DAG visualization
POST /api/kill/{name}          write .kill_{name} sentinel → experiment terminates within 2s
```

### 9. Lineage tracking: `tracker.py`

All runs are logged with DAG structure. IDs include a UUID suffix to avoid collisions.
To log programmatically:
```python
from tracker import Tracker
t = Tracker()
t.log_experiment(
    benchmark="burgers_1d",
    model="FNO",
    val_l2_rel=0.155,
    status="keep",
    config={"hidden_dim": 128, "n_layers": 8, "n_modes": 24},
    parent_name="fno_h128_m24_l6",   # links to parent in DAG
)

# Analyze lineage
analysis = t.analyze_lineage("burgers_1d")
# Returns: best_val, hp_importance (Pearson correlations), model ranking, trend
```

---

## Current state at-a-glance

| Benchmark      | SOTA   | Our best          | Gap      | Priority |
|----------------|--------|-------------------|----------|----------|
| burgers_1d     | 0.0031 | 0.1468            | 47.3×    | **CRITICAL** |
| darcy_2d       | 0.0041 | 0.1041            | 25.4×    | **HIGH** |
| kdv_1d         | 0.010  | **0.0020** ✓      | 5× better SOTA | hold |
| wave_1d        | 0.005  | **0.000992** ✓    | 5× better SOTA | hold |
| euler_1d       | ~0.015 | **0.002413** ✓    | 6.2× better SOTA | hold |
| ns_2d          | 0.0128 | 0.01428           | 1.1×     | MEDIUM   |
| swe_2d         | ~0.002 | 0.0107            | 5.4×     | MEDIUM   |
| allen_cahn_2d  | ~0.020 | 0.0628            | 3.1×     | MEDIUM   |
| ns_hre_2d      | ~0.070 | not run           | —        | LOW (70 min first gen) |

184 completed experiments. Check `uv run autorun.py --dry-run` for pending queue.

---

## Suggested next actions (ranked)

1. **Close Burgers gap** — try SSNO (paper claims 0.007), curriculum training, ensemble FNO+RFNO
2. **darcy_2d** — FNO h=32 l=4 m=12 budget_s=480 (small model constraint — h≥64 crashes)
3. **ns_2d** — try H1 loss or longer budget (720s) to push below SOTA 0.0128
4. **SSNO on all 1D benchmarks** — newly registered model with adaptive S4D damping
5. **allen_cahn_2d / swe_2d** — only a few experiments each; explore model families

```bash
# Quick wins to run right now:
uv run train.py --benchmark burgers_1d   --model SSNO --hidden 64 --layers 4 --modes 16
uv run train.py --benchmark darcy_2d --model FNO  --hidden 32 --layers 4 --modes 12 --budget 480
```

---

## Data Cache

All PDE training and validation datasets are pre-generated and disk-cached at:

```
~/.cache/sciml_autoresearch/
```

**Always run this before starting any experiment session:**

```bash
uv run prefetch_data.py              # pre-cache all benchmarks (~20 min total)
uv run prefetch_data.py --skip-slow  # skip ns_hre_2d (~2 min, everything else)
```

Without this, train.py regenerates training data per subprocess. For 2D benchmarks
(especially `ns_2d` with 4096 × 2D NS samples), this exceeds the 1500s hard timeout
in `autorun.py` and causes every experiment to crash before training begins.

---

## Invariants the agent must respect

| Rule | Why |
|------|-----|
| Never modify `prepare.py` | Defines ground-truth metric for all experiments |
| `ExperimentConfig.name` must be globally unique | Dedup key in results.json |
| Never hand-edit `results.json` | Use tracker.py; hand edits break DAG |
| `darcy_2d` is broken | Wrong solver — only use `darcy_2d` |
| Never add PINO experiments | Endpoint-only formulation always fails |
| 2D benchmarks: h≤32, l≤4, budget_s=480 | h≥64 or l≥8 crashes (OOM/broadcast on Apple Silicon) |
| Stage only specific files with git | Never `git add -A` (avoids committing secrets/data) |
| No new packages | Only mlx, numpy, scipy, matplotlib, pyyaml, fastapi, uvicorn |
