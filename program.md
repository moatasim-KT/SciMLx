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

## The Research Loop (v2.0)

```bash
# 1. Read current state — always start here
uv run analyze.py --papers

# 2. Get ranked next actions with Adversarial Review
uv run auto_suggest.py --generate

# 3. Handle any [AGENT] review requests in the output

# 4. Queue new experiments (edit experiments.yaml)

# 5. Run with Scientific Debugging (auto-rescue on crash)
uv run autorun.py --priority 1 --commit

# 6. Check logs/probes/ if NaNs occurred; fulfill [AGENT] diagnoses
```

This loop is the only sanctioned way to drive the pipeline. Do not skip steps.

**Overnight / unattended run:**
```bash
uv run autorun.py --auto --commit \
    --max-auto-experiments 50 \
    --max-auto-time 86400      # 24-hour wall clock cap
```
`--auto` is a non-recursive iterative loop (stack-safe for indefinite runs):
1. Runs all pending experiments from `experiments.yaml`
2. On queue exhaustion → calls `auto_suggest --generate --write-yaml` (replenishes queue)
3. On failure to generate → falls back to `BayesianHPO.suggest_top(3)` which injects HPO-generated configs
4. Stops after `--max-auto-time` seconds or `--max-auto-experiments` total runs

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
  model: RFNO                             # MODEL_REGISTRY key — case-sensitive! (FNO2D not FNO2d)
  hidden_dim: 128
  n_layers: 8
  n_modes: 24
  lr: 0.001
  batch_size: 64
  budget_s: 1800                          # auto-upgraded: 1D→1800s (30min), 2D→3600s (60min)
  priority: 1                             # 1=now, 2=soon, 3=speculative
  loss_type: h1                           # l2_rel | h1 | h1_adaptive | h1_strong | spectral | l1_rel
  h1_alpha: 0.3                           # only when loss_type: h1
  ema_decay: 0.999                        # EMA weights (0=disabled, 0.999 recommended)
  curriculum: true                        # spectral curriculum (1D+2D)
  snapshot_ensemble: 3                    # inverse-val-error weighted ensemble
  patience: 5                             # early-stop patience (0=off)
  rationale: "One-line reason for this experiment"
```

Then: `uv run autorun.py --priority 1 --commit`

**MODEL_REGISTRY key names are case-sensitive**: use `FNO2D` not `FNO2d`,
`Transolver2D` not `Transolver2d`, `TFNO2D` not `TFNO2d`, `SSNO` not `SSNO1d`,
`EnergyFNO` not `EnergyConservingFNO1d`. Registry lookup fails silently otherwise.

**Budget floors are auto-applied** — any experiment below 1800 s (1D) or 3600 s (2D)
is silently upgraded when autorun loads the queue. No manual floor needed.

**2D benchmark hard limit** (Apple Silicon unified memory):
`hidden_dim ≤ 32`, `n_layers ≤ 4`, `n_modes ≤ 12`.
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

# Training feature flags
uv run train.py --lr_schedule cosine          # warmup_cosine (default), cosine, onecycle, none
uv run train.py --seed 123                    # reproducibility seed (default: 42)
uv run train.py --ema_decay 0.999             # EMA weights (0=disabled, 0.999 recommended)
uv run train.py --patience 5                  # early-stop patience in 10%-budget evals (0=off)

# Pause / resume
touch .autorun_pause
rm .autorun_pause

# Kill a specific running experiment
curl -X POST http://localhost:8000/api/kill/<name>
```

`autorun.py` skips names already in `results.json`, writes logs to `logs/`,
applies multi-fix crash recovery (OOM → `batch_size//2`; NaN → `lr//10`),
and commits results to git after each kept run.

These fields can also be set declaratively in `experiments.yaml`:
```yaml
lr_schedule: onecycle    # warmup_cosine | cosine | onecycle | none
seed: 42                 # integer; affects data ordering and model init
```

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
uv run train.py --loss l2_rel         # default — always start here
uv run train.py --loss h1             # Sobolev H1 (adds gradient term) — 1D and 2D native
uv run train.py --loss h1_adaptive    # H1 with auto-scaled alpha (targets 30% gradient contribution)
uv run train.py --loss h1_strong      # H1 with alpha=1.0 — very sharp fronts
uv run train.py --loss spectral       # frequency-weighted — try when high-freq error is high
uv run train.py --loss l1_rel
```

All losses support 1D and 2D inputs natively. `h1` and `spectral` use
`rfft2`-based spectral gradients for 2D inputs — there is no silent fallback
to plain L2. Use `h1_adaptive` when you are unsure about alpha scaling;
it auto-adjusts per-batch so gradient term ≈ 30% of total loss.

Per-run diagnostics (gradient norms, spectral error breakdown) are available at:
```bash
grep "^diag_" logs/<name>.log    # spectral + timing diagnostics
# diag_grad_norm_max extracted automatically by core/diagnostics.py

# Or via dashboard API:
curl http://localhost:8000/api/diagnostics/<name>
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
| 2D: `h≤32`, `l≤4`, `n_modes≤12` | OOM crash before training starts |
| 2D: `budget_s ≥ 3600` (1D: `≥ 1800`) | Auto-enforced by autorun — never set lower |
| SSNO: `h≤64`, `l≤4` only | Diverges at h≥128 (val explodes to 80+) |
| AFNO: do not queue | Wrong spectral bias — consistently 0.50–0.72 |
| Model keys must be exact: `FNO2D` not `FNO2d` | Registry lookup fails silently |
| No new packages | Only what is in `pyproject.toml` |
| Never `git add -A` | Commits data blobs and secrets |
| Log to `logs/trajectories.jsonl` on every change | Breaks the RL replay buffer |

## Architecture Bug History (Session 1–2 Fixes)

| Bug | File | Fix Applied |
|-----|------|-------------|
| RFNO2d defined twice — second (broken) definition silently took precedence | `models/fno.py` | Deleted duplicate; Pre-LN version restored |
| GNOT_FFNO gate was `mx.zeros` (not tracked by nn.Module) — frozen at 0.5 blend | `models/gnot.py` | Replaced with `nn.Linear(dims, 1)` trainable gate |
| H1/spectral losses silently fell back to L2 on 2D inputs | `core/losses.py` | Implemented proper 2D paths via `rfft2` |
| RFNO1d crashed on multi-channel `[B, N, C]` input | `models/fno.py` | Fixed shape contract to mirror `FNO1dMC` |
| `train.py --help` crashed on `%` in argparse format string | `train.py` | Escaped to `%%` |
