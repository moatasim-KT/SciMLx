# AGENTS.md — SciML AutoResearch Agent Guide

The single canonical reference for any external agent (Claude Code, Gemini,
human operator) driving this research pipeline. Read this in full before
taking any action. For deeper detail see `program.md`.

---

## Role

You are an active AI Researcher. Your job is to close the gap between our
current results and published SOTA on PDE benchmarks — by queuing experiments,
analyzing failures, and when necessary scaffolding new model architectures.

You are not a hyperparameter grid-searcher. Reason about *why* a config might
work before queuing it.

---

## Quick Start

```bash
uv sync
uv run data/prefetch_data.py --skip-slow   # cache datasets once (~2 min)

uv run analyze.py --papers                  # read state vs SOTA
uv run auto_suggest.py                      # ranked next actions
# edit experiments.yaml, then:
uv run autorun.py --priority 1 --commit
```

**Fully autonomous (unattended):**
```bash
uv run autorun.py --auto --commit \
    --max-auto-experiments 50 \
    --max-auto-time 86400
```
`--auto` is a non-recursive iterative loop:
1. Runs all pending experiments from `experiments.yaml`
2. On queue exhaustion → calls `auto_suggest --generate --write-yaml` (replenishes queue)
3. On failure to generate → falls back to `BayesianHPO.suggest_top(3)`
4. Stops after `--max-auto-time` seconds or `--max-auto-experiments` total runs

---

## Finding Current State

| What | Where |
|------|-------|
| Best results per benchmark + SOTA gaps | `uv run analyze.py --papers` |
| Ranked next-step suggestions | `uv run auto_suggest.py` |
| Raw experiment history | `results.json` |
| Pending queue | `experiments.yaml` |
| Model implementations | `models/*.py` |
| Model registry (authoritative key names) | `core/research_plugins.py` |
| SOTA targets per benchmark | `docs/papers/*.yaml` and `docs/SOTA.md` |
| Unimplemented paper ideas | `uv run -m core.paper_registry --pending` |
| Spectral diagnostics for a run | `grep "^diag_" logs/<name>.log` |
| Plateau analysis | `python3 -m core.hypothesis --benchmark <bm>` |

---

## Queuing Experiments

Append YAML blocks to `experiments.yaml`. Required fields:

```yaml
- name: unique_experiment_name          # MUST be globally unique
  benchmark: burgers_1d                 # see benchmark list below
  model: FNO                            # see MODEL_REGISTRY key list below
  hidden_dim: 128
  n_layers: 8
  n_modes: 24
  budget_s: 1800                        # min 1800s (1D) or 3600s (2D) — enforced automatically
  priority: 1                           # 1 = highest
  rationale: "why this config?"
```

Optional fields: `loss_type`, `h1_alpha`, `ema_decay`, `curriculum`, `snapshot_ensemble`,
`patience`, `augment`, `lr`, `grad_clip`, `lr_schedule`, `seed`, `slice_num`, `n_head`.

**Budget floors are auto-applied** — any experiment below 1800 s (1D) or 3600 s (2D)
is silently upgraded when autorun loads the queue.

### Benchmark Registry

**1D**: `burgers_1d`, `kdv_1d`, `wave_1d`, `euler_1d`
**2D**: `darcy_2d`, `ns_2d`, `swe_2d`, `allen_cahn_2d`, `elasticity_2d`,
`wavebench_2d`, `pdebench_2d`, `multiphysics_2d`, `ns_hre_2d`

### MODEL_REGISTRY Key Names (case-sensitive)

| Key | Description |
|-----|-------------|
| `FNO` | Fourier Neural Operator 1D |
| `FNO2D` | FNO 2D (NOT `FNO2d`) |
| `RFNO` | Residual FNO 1D |
| `RFNO2D` | Residual FNO 2D |
| `TFNO` / `TFNO2D` | Tucker-factorized FNO |
| `GNOT` / `GNOT_FFNO` | Graph Neural Operator + hybrid |
| `Transolver` / `Transolver2D` | Physics-attention transformer (NOT `Transolver2d`) |
| `FEDONet2D` | Finite-element hybrid decoder (NOT `FEDONet2d`) |
| `AttentionEnhancedFNO2D` | FNO + attention (NOT `AttentionEnhancedFNO2d`) |
| `SSNO` | State-Space Neural Operator (NOT `SSNO1d`) |
| `EnergyFNO` | Energy-conserving FNO (NOT `EnergyConservingFNO1d`) |
| `MambaNO1d` | Mamba state-space operator |
| `UNO` | U-shaped Neural Operator |
| `WNO` | Wavelet Neural Operator |
| `DeepONet` / `PODDeepONet` | DeepONet variants |

---

## Loss Functions

| Key | When to use | Field syntax |
|-----|-------------|-------------|
| `l2_rel` | Default | `loss_type: l2_rel` |
| `h1` | Shock / gradient errors | `loss_type: h1`, `h1_alpha: 0.3` |
| `h1_adaptive` | H1 with auto-scaled alpha | `loss_type: h1_adaptive` |
| `h1_strong` | Very sharp fronts | `loss_type: h1_strong` |
| `h2` | 2nd derivative penalty (1D) | `loss_type: h2` |
| `spectral` | High-freq under-prediction | `loss_type: spectral` |
| `l1_rel` | Outlier robustness | `loss_type: l1_rel` |

---

## Training Features

| Feature | YAML field | CLI flag | Default |
|---------|-----------|----------|---------|
| EMA weights | `ema_decay: 0.999` | `--ema_decay 0.999` | off |
| Curriculum | `curriculum: true` | `--curriculum` | off |
| Snapshot ensemble | `snapshot_ensemble: 3` | `--snapshot_ensemble 3` | 1 |
| Early stopping | `patience: 5` | `--patience 5` | 5 |
| LR schedule | `lr_schedule: cosine` | `--lr_schedule cosine` | warmup_cosine |
| Augmentation | `augment: true` | `--augment` | off |
| Seed | `seed: 42` | `--seed 42` | 42 |

---

## Diagnosing Poor Results

```bash
grep "^diag_" logs/<name>.log      # spectral error, grad norms
python3 -m core.hypothesis --benchmark <bm>   # full analysis + suggestions
python3 -m core.hypothesis --benchmark <bm> --intervene <current_val>
uv run auto_suggest.py --benchmark <bm> --gaps  # SOTA gap table
```

Key diagnostic fields in logs:
- `diag_high_freq_error` > 0.3 → switch to `h1` or `spectral` loss
- `diag_grad_norm_max` > 10 → reduce `lr` or increase `grad_clip`
- `diag_early_stopped=True` → model converged; consider more budget or different architecture

---

## Hard Constraints

| Rule | Reason |
|------|--------|
| Never modify `data/prepare.py` | Corrupts ground-truth metric |
| Never hand-edit `results.json` | Use `core/tracker.py` |
| Never queue PINO | Endpoint formulation always diverges |
| RFNO is 1D only | Shape mismatch on 2D — crashes |
| AFNO: do not queue | 0.50–0.72 consistently — wrong spectral bias |
| SSNO: `hidden_dim ≤ 64`, `n_layers ≤ 4` | Diverges above these limits |
| 2D: `hidden_dim ≤ 32`, `n_layers ≤ 4`, `n_modes ≤ 12` | Apple Silicon memory ceiling |
| 2D: `budget_s ≥ 3600` | Minimum steps for convergence (auto-enforced) |
| 1D: `budget_s ≥ 1800` | Minimum steps for convergence (auto-enforced) |
| `name` globally unique | Dedup logic in autorun |
| Model keys case-sensitive | `FNO2D` not `FNO2d` — registry lookup fails otherwise |
| Log every queue change to `trajectories.jsonl` | RL replay buffer integrity |
| No new packages | `pyproject.toml` is frozen |
| Never `git add -A` | Stage specific files only |

---

## Papers Directory

`docs/papers/*.yaml` — SOTA targets and pending experiment ideas.

```
papers/
├── fno_2020.yaml          # FNO (0.0128 on NS2D)
├── rfno_2024.yaml         # Residual FNO (0.010 on KdV)
├── gnot_2023.yaml         # GNOT (0.0031 on Burgers — primary target)
└── ... (20+ papers)
```

```bash
uv run -m core.paper_registry --gaps      # SOTA gap table
uv run -m core.paper_registry --pending   # unimplemented ideas
```
