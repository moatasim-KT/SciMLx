# autoresearch-sciml-mlx

Autonomous AI-driven research loop for **Scientific Machine Learning (SciML)**
on Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx) — no
PyTorch, no CUDA.

The system trains neural PDE solvers within a fixed 5-minute budget per
experiment, logs every result to a lineage-aware DAG, and iterates toward
published SOTA. A human or AI agent drives the loop by reading current state,
forming hypotheses, queuing experiments, and analyzing outcomes.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

---

## Quick Start

**Requirements:** Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/)

```bash
uv sync
uv run data/prefetch_data.py --skip-slow   # cache PDE datasets (~2 min)

uv run analyze.py --papers                  # current results vs SOTA
uv run auto_suggest.py                      # ranked next actions

# edit experiments.yaml, then run the queue
uv run autorun.py --priority 1 --commit
```

For a continuous automated run:
```bash
uv run autorun.py --auto --commit --max-auto-experiments 20 --max-auto-time 10800
```

---

## Documentation

| File | Purpose |
|------|---------|
| [`program.md`](./program.md) | Full operational guide — the loop, tools, constraints, scaffold workflow |
| [`CLAUDE.md`](./CLAUDE.md) | Entry point for Claude Code agents |
| [`GEMINI.md`](./GEMINI.md) | Entry point for Gemini and other external agents |
| [`docs/SOTA.md`](./docs/SOTA.md) | SOTA targets per benchmark |
| [`docs/papers/*.yaml`](./docs/papers/) | Paper registry with suggested experiments |
| [`WIKI.md`](./WIKI.md) | Architecture diagrams and system overview |

---

## Repository Layout

```
autoresearch-mlx/
├── train.py               # training harness — routes benchmark → model → trainer
├── autorun.py             # queue runner with crash recovery and git integration
├── agent_loop.py          # automated Bayesian HPO loop (optional)
├── analyze.py             # results analyzer and SOTA gap reporter
├── auto_suggest.py        # ranked next-experiment suggester
├── experiments.yaml       # declarative experiment queue
├── results.json           # SSoT — lineage DAG of all completed runs
├── core/
│   ├── trainer.py         # MLX JIT training loop + AdamW
│   ├── tracker.py         # write API for results.json
│   ├── hypothesis.py      # failure pattern detection
│   ├── hpo.py             # Gaussian Process Bayesian HPO
│   ├── scaffold.py        # gated model registration
│   ├── losses.py          # l2_rel, h1, spectral, l1_rel
│   └── research_plugins.py  # model + benchmark registry
├── models/                # 27 model implementations
├── data/
│   ├── prepare.py         # READ-ONLY ground-truth evaluator
│   ├── prefetch_data.py   # one-time dataset cache
│   ├── benchmarks_ext.py  # extended benchmark definitions
│   └── simulations/       # high-fidelity PDE solvers
├── dashboard/
│   ├── app.py             # FastAPI backend
│   └── ui/dashboard.html  # live dashboard
├── docs/
│   ├── papers/*.yaml      # paper registry
│   └── SOTA.md            # SOTA targets
└── logs/
    ├── trajectories.jsonl # RL replay buffer — agent reasoning log
    └── <name>.log         # per-experiment training logs
```

---

## Current Results

To see the latest benchmark results and SOTA gaps:

```bash
uv run analyze.py --papers
```

To see what to run next:

```bash
uv run auto_suggest.py
```

---

## Dashboard

```bash
python3 dashboard/app.py
# open dashboard/ui/dashboard.html in a browser
```

Live training progress, VRAM usage, lineage DAG, per-experiment log tails,
and kill controls.

---

## Hard Constraints

- `data/prepare.py` is read-only — it defines the ground-truth metric
- `results.json` is never hand-edited — use `core/tracker.py`
- 2D benchmarks require `hidden_dim ≤ 32`, `n_layers ≤ 4`, `budget_s ≥ 480`
- RFNO is 1D-only; PINO is broken — never queue either on 2D benchmarks

Full constraint table with reasons in [`program.md`](./program.md).

---

## Acknowledgments

- Andrej Karpathy for the autonomous research concept
- [MLX](https://github.com/ml-explore/mlx) team at Apple


<!-- STRUCTURE_START -->
```text
autoresearch-mlx/
├── agents/
│   └── skills/
│       └── SciMLx/
│           └── SKILL.md
├── core/
│   ├── __init__.py
│   ├── diagnostics.py
│   ├── hpo.py
│   ├── hypothesis.py
│   ├── loader.py
│   ├── losses.py
│   ├── paper_registry.py
│   ├── readme_hook.py
│   ├── research_plugins.py
│   ├── scaffold.py
│   ├── tracker.py
│   ├── trainer.py
│   ├── utils.py
│   └── viz.py
├── dashboard/
│   ├── ui/
│   │   └── dashboard.html
│   ├── app.py
│   └── monitor.py
├── data/
│   ├── simulations/
│   │   ├── AGENTS.md
│   │   ├── __init__.py
│   │   ├── allen_cahn.py
│   │   ├── elasticity.py
│   │   ├── euler1d.py
│   │   ├── multiphysics.py
│   │   ├── ns_etdrk4.py
│   │   ├── pdebench.py
│   │   ├── radiative.py
│   │   ├── shallow_water.py
│   │   └── wavebench.py
│   ├── benchmarks_ext.py
│   ├── prefetch_data.py
│   └── prepare.py
├── docs/
│   ├── papers/
│   │   ├── afno_2022.yaml
│   │   ├── augmentation_2023.yaml
│   │   ├── curriculum_2009.yaml
│   │   ├── deeponet_2021.yaml
│   │   ├── ensemble_uq_2023.yaml
│   │   ├── ffno_2023.yaml
│   │   ├── fno_2020.yaml
│   │   ├── gnot_2023.yaml
│   │   ├── h1_loss.yaml
│   │   ├── hnn_2019.yaml
│   │   ├── inverse_pinn_2023.yaml
│   │   ├── mambano_2024.yaml
│   │   ├── memno_2025.yaml
│   │   ├── mppde_2022.yaml
│   │   ├── neural_ode_ude_2020.yaml
│   │   ├── physicsnemo_2024.yaml
│   │   ├── pikan_2025.yaml
│   │   ├── pino_2021.yaml
│   │   ├── rfno_2024.yaml
│   │   ├── ssm_s4_2022.yaml
│   │   ├── tfno_2022.yaml
│   │   ├── time_marching_deeponet_2025.yaml
│   │   ├── transolver_2024.yaml
│   │   ├── uno_2022.yaml
│   │   └── wno_2022.yaml
│   ├── LICENSE
│   ├── LITERATURE.md
│   ├── SOTA.md
│   └── TERMINOLOGY.md
├── models/
│   ├── AGENTS.md
│   ├── __init__.py
│   ├── afno.py
│   ├── attention_fno.py
│   ├── axial_attention.py
│   ├── chebyshev_kan.py
│   ├── deeponet.py
│   ├── fedonet.py
│   ├── fno.py
│   ├── gnot.py
│   ├── hano.py
│   ├── hnn.py
│   ├── hybrid_decoder_deeponet.py
│   ├── hybrid_fno_deeponet.py
│   ├── kan.py
│   ├── mamba_no.py
│   ├── mem_no.py
│   ├── neural_ode.py
│   ├── pacmann.py
│   ├── pinn.py
│   ├── s4d.py
│   ├── sno.py
│   ├── ssno.py
│   ├── tfno.py
│   ├── time_deeponet.py
│   ├── transolver.py
│   ├── vsmno.py
│   └── wno.py
├── notebooks/
│   └── colab_experiments.ipynb
├── AGENTS.md
├── CLAUDE.md
├── CODE_INDEX.json
├── GEMINI.md
├── README.md
├── SKILL.md
├── WIKI.md
├── agent_loop.py
├── analyze.py
├── auto_suggest.py
├── autorun.py
├── experiments.yaml
├── model_architectures.md
├── program.md
├── pyproject.toml
├── results.json
├── train.py
└── uv.lock
```
<!-- STRUCTURE_END -->
