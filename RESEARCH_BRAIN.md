# RESEARCH_BRAIN.md — SciML AutoResearch Autonomous Driver

> **This is the Project's Living Brain.** It drives the autonomous SciML research loop,
> evolves with every experiment via `core/brain_distiller.py`, and stores the collective
> memory of what works and what doesn't across all sessions.
> **Mandatory**: Read this file in full before acting. Update it after every significant
> outcome. Never modify the `<!-- MARKER -->` tags — they are written by the distiller.

---

## 1. Mission & Identity

**Role**: Senior AI SciML Researcher operating an autonomous experiment loop.
**Objective**: Minimize `val_l2_rel` (relative L2 error) on PDE benchmarks toward published SOTA.
**Platform**: Apple Silicon (MLX). All model code must be MLX-native — no CUDA, no PyTorch training.
**Mindset**: Active researcher, not a grid-searcher. Every experiment needs a mathematical rationale.

**Current score: 9 / 14 SOTA benchmarks beaten. 184 experiments run.**

---

## 2. Core Mandates (Invariants — Never Break These)

### Data Integrity
- **Never modify `data/prepare.py`** — ground-truth evaluator, changes invalidate all comparisons.
- **Never write `results.json` directly** — use `core/results_store.py`:
  ```python
  from core.results_store import store
  store.append(row)        # idempotent INSERT OR IGNORE by id
  store.load("benchmark")  # read, filtered
  ```
  `results.json` is an auto-exported human-readable copy. `results.db` (SQLite WAL) is the SSoT.
- **`name` in `experiments.yaml` must be globally unique** — duplicates are silently skipped.

### Hardware Limits (Apple Silicon — enforced by `ModelRegistry.build()`)
- **2D hard limit**: `hidden_dim < 64`, `n_layers < 8`. Raises `ValueError` above these.
  Recommended: `hidden_dim = 32`, `n_layers ≤ 4`, `n_modes ≤ 12`.
- **SSNO**: `hidden_dim ≤ 64`, `n_layers ≤ 4`. Diverges at `h ≥ 128`.
- **Budget floors** (auto-applied): 1D ≥ 1800s, 2D ≥ 3600s.

### Known Crash Patterns (Do Not Repeat)
- **`RFNO`/`PINO` on `burgers_nu_001`**: was crashing because benchmark doesn't end in `_1d`.
  Fixed in `train.py` via `_1D_BENCHMARKS = {"burgers_nu_001", "burgers_nu_01", ...}`. ✅
- **`PINO1d` missing `sensor_dim`**: fixed — `train.py` now passes `sensor_dim=GRID_SIZE`. ✅
- **`AFNO`**: consistently 0.50–0.72 (wrong spectral bias). Do not queue.
- **`PACMANN`**: scaffold stub only — operator blocks not implemented.
- **`Transolver2D`/`GNOT` on 2D**: 5–10× slower per epoch than FNO-family. Budget accordingly.

---

## 3. Results Store — Concurrency Architecture

12+ autorun processes write results simultaneously. The old JSON read-modify-write
caused silent data loss. **All writes go through `core/results_store.py`:**

```
store.append(row)
  → SQLite INSERT OR IGNORE   (WAL: readers never blocked, id deduplication)
  → export_json() under FileLock
      → write to .tmp → os.replace() atomic rename → results.json
```

**Stress test result**: 8 concurrent processes × 10 writes = 80/80 rows, 0 lost (1.0s).

---

## 4. The Autonomous Research Loop

### Standard Protocol
1. **Analyze**: `uv run analyze.py` — SOTA gap table + improvement trajectories.
2. **Hypothesize**: Based on diagnostics and §6 knowledge base.
3. **Queue**: Edit `experiments.yaml`. Every entry needs a `rationale:` field.
4. **Execute**: `uv run autorun.py --benchmark <bm> --max <n> --commit`
5. **Distill (auto)**: `core/brain_distiller.py` updates §7 Lessons and §8 Strategy after each run.
6. **Update manually**: Update §9 Roadmap and §2 crash patterns when new systemic knowledge emerges.

### Unattended Overnight Loop
```bash
uv run autorun.py --priority 1 --commit
```
Runs all priority-1 experiments, retries crashes, commits results, evolves this brain.

### Targeted RL Sweep (single benchmark)
```bash
uv run autorun.py --benchmark burgers_nu_001 --max 6 --commit
```

---

## 5. HPO Strategy

### Primary: OptunaHPO (TPE + MedianPruner)
```python
from core.hpo import OptunaHPO
hpo = OptunaHPO("burgers_1d")
hpo.load_history()           # seeds from results.db
candidates = hpo.suggest_top(n=5)
```
- Uses a **scratch study** for candidate generation — live study never contaminated.
- Falls back to `BayesianHPO` (GP + EI) if optuna unavailable.

### Novelty Filter (AI-Scientist cosine dedup)
`agent_loop.py` rejects proposals with cosine similarity ≥ 0.97 to existing experiments.
Prevents redundant near-duplicate configs from filling the queue.

### Parameter Playbook
| Param | 1D default | 2D default | Notes |
|---|---|---|---|
| `n_modes` | 24–32 | ≤ 12 | More modes → better shock resolution |
| `hidden_dim` | 128 | ≤ 32 | 2D hard limit enforced |
| `n_layers` | 8 | ≤ 4 | 10+ is too slow on 1D |
| `lr` | 1e-3 | 1e-3 | 3e-4 for late-stage fine-tuning |
| `batch_size` | 32–64 | 16–32 | Smaller batches for 2D OOM safety |

---

## 6. Loss Function Decision Tree

| Key | Use Case | When |
|---|---|---|
| `l2_rel` | Default | Always start here |
| `h1` | Shock / steep gradients | Burgers, Darcy; set `h1_alpha: 0.1–0.5` |
| `h1_adaptive` | Unknown PDEs | Auto-scales alpha so gradient term ≈ 30% |
| `spectral` | High-frequency error | When `diag_high_freq_error > 0.3` |
| `l1_rel` | Outlier robustness | When `val >> train` |
| PINO (`pino_lambda`) | Physics residual | Burgers shocks: try `pino_lambda: 0.05–0.1` |

---

## 7. Model Registry Keys (Case-Sensitive)

**1D**: `FNO`, `RFNO`, `FFNO`, `UNO`, `WNO`, `TFNO`, `RTFNO`, `CPFNO`, `S4NO`, `SSNO`,
`MambaNO`, `GNOT`, `DeepONet`, `PODDeepONet`, `TimeDeepONet`, `DualDeepONet`,
`HNN`, `EnergyFNO`, `NeuralODE`, `UDE`, `LatentODE`, `PINO`, `KAN_FNO`, `Transolver`

**2D**: `FNO2D`, `RFNO2D`, `TFNO2D`, `UNO2d`, `WNO2d`, `GNOT2d`, `GNOT_Axial2d`,
`GNOT_FFNO`, `Transolver2D`, `VSMNO2D`, `SNO2D`, `AttentionEnhancedFNO2D`,
`HybridDecoderDeepONet2D`, `HybridFNODeepONet2D`, `FEDONet2D`

Full list: `core/research_plugins.py`. All new models must be registered there.

---

## 8. Autonomous Memory (Auto-Updated by brain_distiller.py)

### Active Strategy
<!-- STRATEGY_START -->
- **Current Score**: 9/14 SOTA benchmarks beaten: `allen_cahn_2d`, `burgers_nu_001`, `elasticity_2d`, `euler_1d`, `kdv_1d`
- **Current Focus**: Close `burgers_1d` (47.3× gap).
- **Priority Hypotheses**: High spectral modes for shocks; EMA for convergence stability.
<!-- STRATEGY_END -->

### Learned Lessons Log
| Date | Insight | Action | Outcome |
|---|---|---|---|
<!-- LESSONS_START -->
| 2026-04-22 | Incrementing modes to m=28 based on best config (TFNO h=128 ... | Transolver on burgers_1d | **0.178343 (discard)** |
| 2026-04-22 | Incrementing modes to m=28 based on best config (TFNO h=128 ... | Transolver on burgers_1d | **0.172114 (discard)** |
| 2026-04-22 | Incrementing modes to m=28 based on best config (TFNO h=128 ... | SSNO on burgers_1d | **0.214153 (discard)** |
| 2026-04-22 | Incrementing modes to m=28 based on best config (TFNO h=128 ... | SSNO on burgers_1d | **0.193788 (discard)** |
| 2026-04-22 | Incrementing modes to m=28 based on best config (TFNO h=128 ... | SSNO on burgers_1d | **0.193788 (discard)** |
| 2026-04-22 | Incrementing modes to m=28 based on best config (TFNO h=128 ... | SSNO on burgers_1d | **0.193788 (discard)** |
| 2026-04-22 | Incrementing modes to m=28 based on best config (TFNO h=128 ... | SSNO on burgers_1d | **0.192137 (discard)** |
| 2026-04-22 | Incrementing modes to m=28 based on best config (TFNO h=128 ... | SSNO on burgers_1d | **0.192137 (discard)** |
| 2026-04-22 | Incrementing modes to m=28 based on best config (TFNO h=128 ... | SSNO on burgers_1d | **0.170058 (discard)** |
| 2026-04-22 | Incrementing modes to m=28 based on best config (TFNO h=128 ... | SSNO on burgers_1d | **0.170058 (discard)** |
<!-- LESSONS_END -->

### Architecture Evolution
<!-- ARCH_EVO_START -->
- No recent major architectural shifts.
<!-- ARCH_EVO_END -->

---

## 9. Roadmap & SOTA Gaps

Live values from `results.db`. SOTA targets from `core/utils.py`.

| Benchmark | Our Best | SOTA Target | Gap | Priority | Next Action |
|---|---|---|---|---|---|
| `[wave_1d](docs/benchmarks/wave_1d.md)` | **0.000992** | 0.005 | 0.20× ✅ | low | maintain |
| `[kdv_1d](docs/benchmarks/kdv_1d.md)` | **0.002023** | 0.010 | 0.20× ✅ | low | maintain |
| `[euler_1d](docs/benchmarks/euler_1d.md)` | **0.002413** | 0.003 | 0.80× ✅ | low | maintain |
| `[pdebench_2d](docs/benchmarks/pdebench_2d.md)` | **0.002602** | 0.005 | 0.52× ✅ | low | maintain |
| `[elasticity_2d](docs/benchmarks/elasticity_2d.md)` | **0.007734** | 0.010 | 0.77× ✅ | low | maintain |
| `[wavebench_2d](docs/benchmarks/wavebench_2d.md)` | **0.009907** | 0.015 | 0.66× ✅ | low | maintain |
| `[swe_2d](docs/benchmarks/swe_2d.md)` | **0.010729** | 0.015 | 0.72× ✅ | low | maintain |
| `[allen_cahn_2d](docs/benchmarks/allen_cahn_2d.md)` | **0.062801** | 0.080 | 0.79× ✅ | low | maintain |
| `[burgers_nu_001](docs/benchmarks/burgers_nu_001.md)` | **0.077943** | 0.080 | 0.97× ✅ | medium | push to 0.060 via RFNO+EMA/cosine |
| `[ns_2d](docs/benchmarks/ns_2d.md)` | 0.014284 | 0.0128 | 1.12× | **high** | HPO fine-tune lr=1e-4, EMA |
| `[darcy_2d](docs/benchmarks/darcy_2d.md)` | 0.059719 | 0.0041 | 14.6× | **high** | GNOT attention + longer budget |
| `[burgers_1d](docs/benchmarks/burgers_1d.md)` | 0.181264 | 0.0031 | 58× | **high** | PINO physics loss + RFNO h=256 |
| `[multiphysics_2d](docs/benchmarks/multiphysics_2d.md)` | 0.692273 | 0.200 | 3.5× | medium | multi-channel FNO2D baseline |
| `[radiative_2d](docs/benchmarks/radiative_2d.md)` | 1.000 | — | unsolved | medium | debug data pipeline first |
| `[poisson_2d](docs/benchmarks/poisson_2d.md)` | **0.470284** | 0.0001 | 4702× | **high** | Iterate more steps (n_iterations=50) |
| `[reionization_1d](docs/benchmarks/reionization_1d.md)` | **0.036645** | 0.050 | 0.73× ✅ | low | maintain |

---

## 10. Key File Locations

| Asset | Location | Notes |
|---|---|---|
| **Brain (This File)** | `RESEARCH_BRAIN.md` | Auto-updated by `core/brain_distiller.py` |
| **Experiment Queue** | `experiments.yaml` | YAML list; `name` must be globally unique |
| **Results DB (SSoT)** | `results.db` | SQLite WAL, gitignored |
| **Results JSON (export)** | `results.json` | Auto-synced from DB after every write |
| **Results Store** | `core/results_store.py` | All reads/writes go here |
| **Trajectories (RL buffer)** | `logs/trajectories.jsonl` | Append-only; input to brain distiller |
| **HPO** | `core/hpo.py` | OptunaHPO (primary) + BayesianHPO (fallback) |
| **Novelty Filter** | `agent_loop.py` | Cosine similarity ≥ 0.97 → reject |
| **Model Registry** | `core/research_plugins.py` | All model keys registered here |
| **SOTA Targets** | `core/utils.py` (`SOTA` dict) | Referenced by analyze.py and autorun.py |
| **Diagnostics** | `core/diagnostics.py` | Log parser for crash/plateau/spectral-bias |
| **Brain Distiller** | `core/brain_distiller.py` | Updates §8 markers after every run |
