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

## Start Here

```bash
uv sync
uv run data/prefetch_data.py --skip-slow   # cache datasets first (always)

uv run analyze.py --papers                  # read current state vs SOTA
uv run auto_suggest.py                      # get ranked next actions
# edit experiments.yaml, then:
uv run autorun.py --priority 1 --commit
```

Repeat that loop. Full tool reference, diagnostic APIs, and scaffold workflow
are in `program.md`.

---

## Key File Locations

| What | Where |
|------|-------|
| Current results + SOTA gaps | `uv run analyze.py --papers` |
| What to do next | `uv run auto_suggest.py` |
| Experiment queue | `experiments.yaml` |
| All completed runs | `results.json` (SSoT — never hand-edit) |
| Models | `models/*.py` and `models/__init__.py` |
| SOTA targets | `docs/papers/*.yaml` and `docs/SOTA.md` |
| Unimplemented paper ideas | `uv run -m core.paper_registry --pending` |
| Run logs + spectral diagnostics | `logs/<name>.log` |
| RL replay buffer | `logs/trajectories.jsonl` (you must write here) |
| Ground-truth evaluator | `data/prepare.py` (read-only — never modify) |

---

## Non-Negotiable Rules

- **Never modify `data/prepare.py`**
- **Never hand-edit `results.json`** — use `core/tracker.py`
- **Never queue PINO** — always diverges
- **RFNO is 1D-only** — crashes on all 2D benchmarks
- **2D benchmarks**: `hidden_dim ≤ 32`, `n_layers ≤ 4`, `budget_s ≥ 480`
- **`name` in `experiments.yaml` must be globally unique**
- **Log to `logs/trajectories.jsonl`** on every queue change or model scaffold
- **No new packages** beyond `pyproject.toml`
- **Never `git add -A`** — stage specific files only

Full constraint table with reasons: `program.md`.
