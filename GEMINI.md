# GEMINI.md — SciML AutoResearch

Entry point for Gemini and other external agents. Read `program.md` for the
full operational guide before taking any action in this repository.

---

## What This Is

An autonomous SciML research loop on Apple Silicon (MLX, no CUDA). You drive
experiments that train neural PDE solvers within a fixed time budget and
minimize `val_l2_rel` (relative L2 error, lower is better) toward published
SOTA. You are an active researcher — reason about *why* before queuing anything.

---

## Start Here (v2.0)

```bash
uv sync
uv run data/prefetch_data.py --skip-slow   # cache datasets first (always)

uv run analyze.py --papers                  # read current state vs SOTA
uv run auto_suggest.py --generate           # actions with [Adversarial Review]
# Fulfill any [AGENT] review requests, then edit experiments.yaml
uv run autorun.py --priority 1 --commit     # execute with [Scientific Debugging]
```

---

## Infrastructure Capabilities (v2.0)

| Feature | Flag / Field | Description |
|---------|-------------|-------------|
| **Scientific Debugger** | `--probe` | 5-step re-run on crash; logs layer-wise stats. |
| **Adversarial Review** | `critique:` | Skeptical architectural vetting in `experiments.yaml`. |
| **Spectral Curriculum** | `curriculum: true` | Low-pass filters ICs during first 30% of budget. |
| **Snapshot Ensembles** | `snapshot_ensemble: N` | Inverse-val-error-weighted average of snapshots. |
| **EMA Weights** | `ema_decay: 0.999` | Exponential moving average shadow weights. |
| **Adaptive H1 Loss** | `loss_type: h1_adaptive` | Auto-scales alpha per-batch (gradient ≈ 30%). |

---

## Loss Functions

| Key | When to use |
|-----|-------------|
| `l2_rel` | Default — always start here |
| `h1` | Shock/gradient errors; Burgers, Darcy (set `h1_alpha: 0.1–0.5`) |
| `h1_adaptive` | Auto-scales alpha per-batch; safer default for unknowns |
| `h1_strong` | `alpha=1.0`; heavy gradient penalty for very sharp fronts |
| `h2` | Second-derivative penalty; 1D only |
| `spectral` | High-frequency under-prediction; KdV solitons |
| `l1_rel` | Outlier robustness |
| `mse` | Debug only |

Example YAML entry with all new features:
```yaml
- name: fno2d_darcy_h32_l4_m8_h1_ema
  benchmark: darcy_2d
  model: FNO2D
  hidden_dim: 32
  n_layers: 4
  n_modes: 8
  loss_type: h1
  h1_alpha: 0.3
  ema_decay: 0.999
  curriculum: true
  snapshot_ensemble: 3
  patience: 5
  budget_s: 3600
  priority: 1
  rationale: "H1 + EMA + curriculum + ensemble on darcy_2d"
```

---

## Key File Locations

| What | Where |
|------|-------|
| Current results + SOTA gaps | `uv run analyze.py --papers` |
| What to do next | `uv run auto_suggest.py` |
| Experiment queue | `experiments.yaml` |
| All completed runs | `results.json` (SSoT — never hand-edit) |
| Models | `models/*.py` and `models/__init__.py` |
| Model registry | `core/research_plugins.py` — MODEL_REGISTRY |
| SOTA targets | `docs/papers/*.yaml` and `docs/SOTA.md` |
| Unimplemented paper ideas | `uv run -m core.paper_registry --pending` |
| Run logs + spectral diagnostics | `logs/<name>.log` |
| RL replay buffer | `logs/trajectories.jsonl` (you must write here) |
| Ground-truth evaluator | `data/prepare.py` (read-only — never modify) |

---

## Non-Negotiable Rules

- **Never modify `data/prepare.py`**
- **Never hand-edit `results.json`** — use `core/tracker.py`
- **Never queue PINO** — always diverges (endpoint-only formulation)
- **RFNO is 1D-only** — crashes on all 2D benchmarks
- **AFNO: do not queue** — consistently 0.50–0.72 on all benchmarks
- **SSNO**: registry key is `SSNO`, `hidden_dim ≤ 64`, `n_layers ≤ 4`
- **2D benchmarks**: `hidden_dim ≤ 32`, `n_layers ≤ 4`, `n_modes ≤ 12`
- **Model names in `experiments.yaml` must match MODEL_REGISTRY keys exactly** — `FNO2D` not `FNO2d`, `Transolver2D` not `Transolver2d`, `SSNO` not `SSNO1d`, `EnergyFNO` not `EnergyConservingFNO1d`
- **`name` in `experiments.yaml` must be globally unique**
- **Log to `logs/trajectories.jsonl`** on every queue change or model scaffold
- **No new packages** beyond `pyproject.toml`
- **Never `git add -A`** — stage specific files only

Full constraint table with reasons: `program.md`.
