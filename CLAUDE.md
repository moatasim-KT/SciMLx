# CLAUDE.md — SciML AutoResearch

Project instructions for Claude Code. Read `program.md` for the full
operational guide before taking any action in this repository.

---

## What This Is

An autonomous SciML research loop on Apple Silicon (MLX). You train neural
PDE solvers within a fixed budget and push `val_l2_rel` (lower is better)
toward published SOTA. You are an active researcher — reason about *why*
before queuing anything.

---

## Start Here

```bash
uv sync
uv run data/prefetch_data.py --skip-slow   # cache datasets first (always)

uv run analyze.py --papers                  # read current state vs SOTA
uv run auto_suggest.py                      # get ranked next actions
# edit experiments.yaml, then:
uv run autorun.py --priority 1 --commit
```

**Fully autonomous (unattended):**
```bash
uv run autorun.py --auto --commit \
    --max-auto-experiments 50 \
    --max-auto-time 86400      # 24-hour wall clock cap
```
When the queue empties, `--auto` calls `auto_suggest --generate --write-yaml`
to replenish it, then falls back to `BayesianHPO.suggest_top(3)`. The loop
is iterative (no recursion) and honours `--max-auto-experiments` / `--max-auto-time`.

Full tool reference is in `program.md`.

---

## Key File Locations

| What | Where |
|------|-------|
| Current results + SOTA gaps | `uv run analyze.py --papers` |
| What to do next | `uv run auto_suggest.py` |
| Experiment queue | `experiments.yaml` |
| All completed runs | `results.json` (SSoT — never hand-edit) |
| Models | `models/*.py` and `models/__init__.py` (48+ models) |
| Model registry | `core/research_plugins.py` — `MODEL_REGISTRY` |
| SOTA targets | `docs/papers/*.yaml` and `docs/SOTA.md` |
| Run logs | `logs/<name>.log` |
| RL replay buffer | `logs/trajectories.jsonl` (you must write here) |
| Ground-truth evaluator | `data/prepare.py` (read-only — never modify) |
| Per-run diagnostics API | `GET /api/diagnostics/<name>` (dashboard) |
| Spectral/gradient diagnostics | `core/diagnostics.py` — `parse_log_file()` |
| Plateau detection | `core/hypothesis.py` — `HypothesisEngine.detect_plateau()` |

---

## Non-Negotiable Rules

- **Never modify `data/prepare.py`**
- **Never hand-edit `results.json`** — use `core/tracker.py`
- **Never queue PINO** — always diverges (endpoint-only formulation)
- **RFNO is 1D-only** — crashes on all 2D benchmarks
- **AFNO: do not queue** — consistently 0.50–0.72 on all benchmarks
- **SSNO**: registry key is `SSNO`, `hidden_dim ≤ 64`, `n_layers ≤ 4`
- **2D benchmarks**: `hidden_dim ≤ 32`, `n_layers ≤ 4`, `n_modes ≤ 12`
- **Budget floors auto-applied**: 1D → 1800 s (30 min), 2D → 3600 s (60 min)
- **`name` in `experiments.yaml` must be globally unique**
- **Model names in `experiments.yaml` must match MODEL_REGISTRY keys exactly** (e.g. `FNO2D` not `FNO2d`)
- **Log to `logs/trajectories.jsonl`** on every queue change or model scaffold
- **No new packages** beyond `pyproject.toml`
- **Never `git add -A`** — stage specific files only

---

## Model Registry Key Naming

Use these exact keys in `experiments.yaml` — case matters:

| Dimension | Correct key | Wrong |
|-----------|-------------|-------|
| 2D FNO | `FNO2D` | `FNO2d` |
| 2D RFNO | `RFNO2D` | `RFNO2d` |
| 2D TFNO | `TFNO2D` | `TFNO2d` |
| 2D Transolver | `Transolver2D` | `Transolver2d` |
| 2D FEDONet | `FEDONet2D` | `FEDONet2d` |
| 2D AttentionFNO | `AttentionEnhancedFNO2D` | `AttentionEnhancedFNO2d` |
| SSNO (any dim) | `SSNO` | `SSNO1d` |
| Energy FNO | `EnergyFNO` | `EnergyConservingFNO1d` |

---

## Loss Functions

| Key | When to use |
|-----|-------------|
| `l2_rel` | Default — always start here |
| `h1` | Shock/gradient errors; Burgers, Darcy (`h1_alpha: 0.1–0.5`) |
| `h1_adaptive` | Same as `h1` but auto-scales alpha per-batch; safer default |
| `h1_strong` | `alpha=1.0`; heavy gradient penalty for very sharp fronts |
| `h2` | Second-derivative penalty; 1D only |
| `spectral` | High-frequency under-prediction; KdV solitons |
| `l1_rel` | Outlier robustness |
| `mse` | Debug only — no normalisation |

---

## New Capabilities (Sessions 1–3)

- **LR schedules**: `--lr_schedule {warmup_cosine,cosine,onecycle,none}` — default `warmup_cosine`
- **Seed control**: `--seed <int>` — default 42 (also field in `ExperimentConfig`)
- **2D H1/spectral loss**: works natively via `rfft2` — no silent L2 fallback
- **Gradient norm tracking**: `diag_grad_norm_max` extracted from all run logs
- **HPO auto-trigger**: `--auto` falls back to `BayesianHPO.suggest_top(3)` when queue empties
- **Dashboard diagnostics**: `/api/diagnostics/<name>` returns grad norms + spectral bias
- **GNOT_FFNO gate fix**: gate is now a trainable `nn.Linear` (was frozen `mx.zeros`)
- **RFNO2d dedup fix**: duplicate class definition deleted — Pre-LN version now active
- **EMA weights**: `--ema_decay 0.999` — shadow copy used for eval; 3–8% free improvement
- **Adaptive H1 loss**: `--loss h1_adaptive` — auto-scales alpha so grad term ≈ 30% of total
- **2D curriculum**: `--curriculum` now works on 2D inputs via `rfft2` masking
- **Weighted snapshot ensemble**: `--snapshot_ensemble N` — inverse-val-error weights (better snapshots get more weight)
- **Early stopping**: `--patience N` (default 5) — halts after N consecutive non-improving 10%-budget evals
- **Plateau detection**: `HypothesisEngine.detect_plateau(benchmark)` — reads `trajectories.jsonl`, returns `PlateauState`
- **Budget floors**: autorun enforces 1800 s (1D) / 3600 s (2D) minimums automatically
- **auto_suggest --generate --write-yaml**: appends new experiment YAML blocks directly to `experiments.yaml`
- **Iterative autonomous loop**: `--auto` is now a non-recursive `while True` loop; stack-safe for indefinite runs

Full constraint table with reasons: `program.md`.
