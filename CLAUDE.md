# CLAUDE.md

Guidance for Claude Code and any external AI agent working in this repository.

---

## Quick Start

```bash
uv sync                       # install dependencies
uv run prefetch_data.py       # one-time: pre-generate + cache ALL PDE datasets
                              # (~20 min total; use --skip-slow to skip ns_hre_2d)
                              # cached to: ~/.cache/sciml_autoresearch/

# Run a single experiment (~6-8 min on Apple Silicon)
uv run train.py --model FNO --hidden 128 --layers 8 --modes 24

# See current state vs SOTA
uv run analyze.py --papers
uv run auto_suggest.py --gaps

# Run all pending priority-1 experiments
uv run autorun.py --priority 1 --commit

# Dashboard (open ui/dashboard.html in browser after starting)
uv run uvicorn app:app --reload --port 8000
```

---

## Orchestration Modes

There are two ways to drive the research loop. Both write to the same
`results.json` / `results.tsv` and share the `experiments.py` queue.

### Mode A — External Agent (this file, original, always supported)

An external AI (Claude Code, Gemini, human) reads this file and drives
the loop manually.

```bash
1. uv run analyze.py --papers       # understand current state vs SOTA
2. uv run auto_suggest.py           # get ranked suggestions
3. # Edit experiments.py / models/*.py as needed
4. uv run autorun.py --priority 1 --commit
5. Repeat from step 1
```

### Mode B — agent_loop.py (optional, automated)

A fully automated in-process loop that generates new `ExperimentConfig`
entries using the HypothesisEngine + Bayesian HPO, without any human or
external-agent input.

```bash
uv run agent_loop.py --dry-run           # plan without writing
uv run agent_loop.py --top 5             # generate 5 new configs + append
uv run agent_loop.py --run                     # generate + run top-3
```

Mix freely: use Mode A for novel ideas, Mode B for saturation.

---

## All Commands

```bash
# ── Training ──────────────────────────────────────────────────────────────────
uv run train.py --model FNO  --hidden 128 --layers 8  --modes 24
uv run train.py --model RFNO --hidden 128 --layers 10 --modes 24
uv run train.py --benchmark kdv_1d       --model RFNO --hidden 128 --layers 8  --modes 24
uv run train.py --benchmark darcy_2d --model FNO  --hidden 32 --layers 4  --modes 8 --budget 480

# ── Autonomous runner ─────────────────────────────────────────────────────────
uv run autorun.py --priority 1 --commit        # run priority-1 pending experiments
uv run autorun.py --auto --commit              # run + re-suggest until queue empty

# ── Analysis ──────────────────────────────────────────────────────────────────
uv run analyze.py                              # full results report
uv run analyze.py --papers                     # results + SOTA gap
uv run auto_suggest.py                         # ranked next-step suggestions

# ── Dashboard ─────────────────────────────────────────────────────────────────
uv run uvicorn app:app --reload --port 8000
# Then open ui/dashboard.html in a browser
```

---

## Empirical Findings (184 experiments)

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

- **RFNO is 1D-only** — crashes on 2D input.
- **2D constraint**: use `h≤32, l≤4, budget_s=480`. Larger models OOM on Apple Silicon.
- **Burgers single biggest win**: Augmentation (+aug → 0.1468).

---

## Hard Constraints

- **Never modify `prepare.py`** — defines ground-truth metric.
- **No new packages** beyond `pyproject.toml`.
- **`ExperimentConfig.name` must be unique**.
- **`results.json` is SSoT** — never hand-edit; use `tracker.py`.
- **`darcy_2d` is broken** — only use `darcy_2d` and `ns_2d`.
- **PINO is broken** — never queue PINO experiments.
- **Git hygiene** — stage only code and results. Never `git add -A`.
