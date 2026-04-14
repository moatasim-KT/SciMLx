# SciML AutoResearch — Agent Operational Guide

The single canonical reference for any external agent (Claude Code, Gemini,
human) driving this research pipeline. Read this in full before taking any
action.

---

## Your Role

You are an active AI Researcher. Your job is to close the gap between our
current results and published SOTA on PDE benchmarks — by queuing experiments,
analyzing failures, and when necessary inventing new model architectures.

You are not a hyperparameter grid-searcher. Reason about *why* a config might
work before queuing it.

---

## First-Time Setup

```bash
uv sync                                    # install dependencies
uv run data/prefetch_data.py               # pre-cache all PDE datasets (~20 min)
uv run data/prefetch_data.py --skip-slow   # or skip ns_hre_2d (~2 min)
# datasets cached at: ~/.cache/sciml_autoresearch/
```

Run `prefetch_data.py` once per machine. Without it, 2D benchmarks regenerate
data per subprocess and exceed the hard timeout before training starts.

---

## The Research Loop

```bash
# 1. Read current state — always start here
uv run analyze.py --papers

# 2. Get ranked next actions
uv run auto_suggest.py

# 3. Queue new experiments (edit experiments.yaml)
#    OR scaffold a new model architecture (see below)

# 4. Run
uv run autorun.py --priority 1 --commit

# 5. Go to 1
```

This loop is the only sanctioned way to drive the pipeline. Do not skip steps.

**Overnight / unattended run:**
```bash
uv run autorun.py --auto --commit --max-auto-experiments 20 --max-auto-time 10800
```
When the queue empties, `--auto` invokes `agent_loop.py --top 5` (Bayesian HPO)
and recurses until the time or experiment cap is hit.

---

## Finding Current State

| What you need | Where to look |
|---------------|---------------|
| Best results per benchmark + SOTA gaps | `uv run analyze.py --papers` |
| Ranked next-step suggestions | `uv run auto_suggest.py` |
| Raw experiment history (DAG) | `results.json` |
| Pending queue | `experiments.yaml` |
| All model implementations | `models/*.py` |
| SOTA targets per benchmark | `docs/papers/*.yaml` and `docs/SOTA.md` |
| Unimplemented paper ideas | `uv run -m core.paper_registry --pending` |
| Spectral diagnostics for a run | `grep "^diag_" logs/<name>.log` |
| Full log for a run | `logs/<name>.log` |

Never embed results tables or model lists in these docs. Read the live sources.

---

## Queuing Experiments

Add entries to `experiments.yaml`. Each entry must be a valid YAML object:

```yaml
- name: rfno_burgers_spectral_h128        # globally unique — the dedup key
  benchmark: burgers_1d                   # see docs/SOTA.md for benchmark names
  model: RFNO                             # see models/__init__.py for valid names
  hidden_dim: 128
  n_layers: 8
  n_modes: 24
  lr: 0.001
  batch_size: 64
  budget_s: 300                           # 300 for 1D, 480 for 2D, 600 for ns_2d
  priority: 1                             # 1=now, 2=soon, 3=speculative
  parent_name: "fno_burgers_aug_h128"     # parent run in lineage DAG
  rationale: "One-line reason for this experiment"
```

Then: `uv run autorun.py --priority 1 --commit`

**Before queuing anything**, check `results.json` and `experiments.yaml` to
confirm the name is not already taken and the config has not already been run.

**2D benchmark hard limit** (Apple Silicon unified memory):
`hidden_dim ≤ 32`, `n_layers ≤ 4`, `n_modes ≤ 12`, `budget_s ≥ 480`.
Anything larger crashes before training starts.

---

## Running Experiments

```bash
uv run autorun.py --priority 1 --commit       # run all priority-1 pending
uv run autorun.py --dry-run                   # preview without running
uv run autorun.py --benchmark burgers_1d      # filter by benchmark
uv run autorun.py --model RFNO                # filter by model

# Single experiment (for quick tests)
uv run train.py --benchmark burgers_1d --model FNO --hidden 128 --layers 8 --modes 24

# Pause / resume
touch .autorun_pause
rm .autorun_pause

# Kill a specific running experiment
curl -X POST http://localhost:8000/api/kill/<name>
```

`autorun.py` skips names already in `results.json`, writes logs to `logs/`,
applies multi-fix crash recovery (OOM → `batch_size//2`; NaN → `lr//10`),
and commits results to git after each kept run.

---

## Diagnosing Poor Results

When a run underperforms, use these tools before queuing a follow-up:

```bash
# What does the spectral error profile look like?
grep "^diag_" logs/<name>.log
# diag_high_freq_error high → increase n_modes or use spectral loss
# diag_low_freq_error high  → check data normalization

# What does the hypothesis engine say?
python3 -c "
from core.hypothesis import HypothesisEngine
e = HypothesisEngine()
print(e.suggest_intervention('burgers_1d', best_val=0.1468))
"

# What hyperparameters matter most on this benchmark?
uv run -m core.hpo --benchmark burgers_1d --top 5
```

---

## Loss Functions

```bash
uv run train.py --loss l2_rel      # default — always start here
uv run train.py --loss h1          # Sobolev H1 (adds gradient term)
uv run train.py --loss spectral    # frequency-weighted — try when high-freq error is high
uv run train.py --loss l1_rel
```

---

## Scaffolding New Model Architectures

When existing models cannot close a gap, invent new blocks. Workflow:

```bash
# 1. Generate stub
uv run -m core.scaffold --stub MyModel --base FNO --notes "Describe innovation"
# → writes models/mymodel.py

# 2. Edit models/mymodel.py
#    Allowed: linear projections, spectral gating (FFT/IFFT + complex multiply),
#             windowed 1D attention, 1D/2D convolutions
#    Forbidden: 3D convolutions, full 2D attention, any op requiring >4 GB

# 3. Validate + register (runs syntax → import → shape smoke tests)
uv run -m core.scaffold --register MyModel models/mymodel.py --benchmarks burgers_1d
```

Registration adds the model to `models/__init__.py`, `core/research_plugins.py`,
and appends starter experiment entries to `experiments.yaml`.

---

## Mandatory: RL Trajectory Logging

Every time you change the experiment queue or scaffold a model, append to
`logs/trajectories.jsonl`. This is the Replay Buffer for future agents.

```json
{
  "state": "<benchmark + current best val_l2_rel context>",
  "hypothesis": "<mathematical or empirical reasoning>",
  "action": "<what you queued or scaffolded>"
}
```

There are no exceptions to this rule.

---

## Dashboard

```bash
python3 dashboard/app.py
# open dashboard/ui/dashboard.html in a browser
```

Useful during a run: live loss, VRAM, kill button, lineage DAG, log tail.

---

## Hard Constraints

| Rule | Consequence if broken |
|------|-----------------------|
| Never modify `data/prepare.py` | Corrupts ground-truth metric for all experiments |
| `name` in `experiments.yaml` must be globally unique | Causes silent dedup skip in results |
| Never hand-edit `results.json` | Breaks the lineage DAG |
| Never add PINO experiments | Always diverges (endpoint-only formulation) |
| RFNO is 1D-only | Crashes on all 2D benchmarks (shape mismatch) |
| 2D: `h≤32`, `l≤4`, `budget_s≥480` | OOM crash before training starts |
| SSNO: `h≤64`, `l≤4` only | Diverges at h≥128 (val explodes to 80+) |
| AFNO: do not queue | Wrong spectral bias — consistently 0.50–0.72 |
| No new packages | Only what is in `pyproject.toml` |
| Never `git add -A` | Commits data blobs and secrets |
| Log to `logs/trajectories.jsonl` on every change | Breaks the RL replay buffer |
