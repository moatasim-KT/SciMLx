# autoresearch-sciml-mlx

Autonomous AI-driven research loop for **Scientific Machine Learning (SciML)**
on Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx) — no
PyTorch, no CUDA.

The system trains neural PDE solvers within a fixed budget per experiment
(30 min for 1D, 60 min for 2D — auto-enforced), logs every result to a
lineage-aware DAG, and iterates toward published SOTA. A human or AI agent
drives the loop by reading current state, forming hypotheses, queuing
experiments, and analyzing outcomes.

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

For a fully autonomous unattended run:
```bash
uv run autorun.py --auto --commit \
    --max-auto-experiments 50 \
    --max-auto-time 86400    # 24-hour cap
```

`--auto` is a non-recursive iterative loop: runs queue → replenishes via
`auto_suggest --generate --write-yaml` → falls back to `BayesianHPO` → stops
after 3 idle rounds or the time/experiment cap.

Optional `train.py` flags:
```bash
uv run train.py --lr_schedule cosine    # warmup_cosine | cosine | onecycle | none
uv run train.py --seed 123              # reproducibility seed (default: 42)
uv run train.py --ema_decay 0.999       # EMA shadow weights (3–8% free improvement)
uv run train.py --patience 5            # early stopping (10%-budget evals)
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
│   ├── losses.py          # l2_rel, h1, spectral, l1_rel (1D + 2D native)
│   ├── diagnostics.py     # spectral bias, grad norm extraction, failure classifier
│   ├── mlflow_integration.py  # optional MLflow run tracking
│   ├── model_versioning.py    # model checkpoint versioning
│   └── research_plugins.py  # model + benchmark registry
├── models/                # 48+ model implementations
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

## New Capabilities (Sessions 1–10)

| Feature | How to use |
|---------|-----------|
| EMA weights | `ema_decay: 0.999` in YAML or `--ema_decay 0.999` — 3–8% free improvement |
| Adaptive H1 loss | `loss_type: h1_adaptive` — auto-scales α to target 30% gradient contribution |
| 2D curriculum | `curriculum: true` — spectral masking from k=2→N//4 over first 30% of budget |
| Weighted snapshot ensemble | `snapshot_ensemble: 3` — inverse-val-error weighting across checkpoints |
| Early stopping | `patience: 5` — halts if no improvement for N consecutive 10%-budget evals |
| Budget floors | Auto-enforced: 1D ≥ 1800 s, 2D ≥ 3600 s — no manual config needed |
| Plateau detection | Benchmarks with < 2% relative improvement over 5 runs get deprioritized |
| Iterative auto loop | `--auto` is a stack-safe `while True` loop; queue replenished via `auto_suggest` |

## Hard Constraints

- `data/prepare.py` is read-only — it defines the ground-truth metric
- `results.json` is never hand-edited — use `core/tracker.py`
- 2D benchmarks: `hidden_dim ≤ 32`, `n_layers ≤ 4`, `n_modes ≤ 12` (memory ceiling)
- Budget floors auto-enforced: 1D → 1800 s (30 min), 2D → 3600 s (60 min)
- Model registry keys are case-sensitive: `FNO2D` not `FNO2d`, `SSNO` not `SSNO1d`
- RFNO is 1D-only; PINO always diverges — never queue either
- AFNO: do not queue — consistently 0.50–0.72 spectral bias
- SSNO: `hidden_dim ≤ 64`, `n_layers ≤ 4` only — diverges at h≥128

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
│   ├── mlflow_integration.py
│   ├── model_versioning.py
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
├── mlruns/
│   ├── 0/
│   │   └── meta.yaml
│   ├── 1/
│   │   ├── 084f1fb5b959456b8baf30f508c2c76f/
│   │   │   └── artifacts/
│   │   │       └── validation_ssno_burgers_best.npz
│   │   ├── 0d9843e5ed4f4c6696b4e751083c2cdd/
│   │   │   └── artifacts/
│   │   │       ├── mambano_burgers_curriculum_adapt.log
│   │   │       └── mambano_burgers_curriculum_adapt_best.npz
│   │   ├── 0da8cee76c0940e7a11f61401ccc022f/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_baseline_h128_l8_m24.log
│   │   │       └── fno_burgers_baseline_h128_l8_m24_best.npz
│   │   ├── 17e7b93e132a475fb39531e06aaee960/
│   │   │   └── artifacts/
│   │   │       └── afno_fix_burgers_v2_best.npz
│   │   ├── 28a02f8a1dd146c6ae80f5ebf0f65ab4/
│   │   │   └── artifacts/
│   │   │       └── afno_fix_v3_best.npz
│   │   ├── 2a133368cbbb4025802d02d1f79c1e61/
│   │   │   └── artifacts/
│   │   │       └── validation_curriculum_smooth_best.npz
│   │   ├── 3bc9ecb103cf4d7d894af3a9dd09b478/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_onecycle_aug_h128_l8_m24.log
│   │   │       └── fno_burgers_onecycle_aug_h128_l8_m24_best.npz
│   │   ├── 3ff8598ac74844fb8ef5c10b0d0517e8/
│   │   │   └── artifacts/
│   │   │       └── validation_pino_fix_best.npz
│   │   ├── 4d95bbd7cc9447e2a98362d265f69b95/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_curriculum_ema_h128_l8_m24.log
│   │   │       └── fno_burgers_curriculum_ema_h128_l8_m24_best.npz
│   │   ├── 52096fa8b71f452883fb1533c72045b3/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_curriculum_h128_l8_m24.log
│   │   │       └── fno_burgers_curriculum_h128_l8_m24_best.npz
│   │   ├── 54625101ce2f4fed8363ecf8c6f11b4e/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_onecycle_aug_h128_l8_m24.log
│   │   │       └── fno_burgers_onecycle_aug_h128_l8_m24_best.npz
│   │   ├── 5b0c5120f3164fb69e286941440c18c8/
│   │   │   └── artifacts/
│   │   │       └── validation_ensemble_uq_best.npz
│   │   ├── 82983239f8fe4ae889d7ab5f2ef9d183/
│   │   │   └── artifacts/
│   │   │       └── mambano_burgers_h128_l8_h1_best.npz
│   │   ├── 9410178ddfd7418b8dedb23f859c2d87/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_onecycle_aug_h128_l8_m24.log
│   │   │       └── fno_burgers_onecycle_aug_h128_l8_m24_best.npz
│   │   ├── 96c5b66cac274f6cb37a5ea71e5577c6/
│   │   │   └── artifacts/
│   │   │       ├── mambano_burgers_curriculum.log
│   │   │       └── mambano_burgers_curriculum_best.npz
│   │   ├── b48aca5c6f9c42d4bfe9f37e38dc1ab2/
│   │   │   └── artifacts/
│   │   │       └── ffno_burgers_test_best.npz
│   │   ├── bf4a87c76f2a4107a71f1a281d7f27e5/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_onecycle_aug_h128_l8_m24.log
│   │   │       └── fno_burgers_onecycle_aug_h128_l8_m24_best.npz
│   │   ├── ca1b039eecd6409b839b073ab3eb404f/
│   │   │   └── artifacts/
│   │   │       └── afno_fix_v4_best.npz
│   │   ├── e583018f152f4544bb845bdd9e553d9d/
│   │   │   └── artifacts/
│   │   │       └── repro_afno_bias_best.npz
│   │   ├── ec87c364298a43abb3835ae27d49bc75/
│   │   │   └── artifacts/
│   │   │       ├── fno_burgers_ema_h128_l8_m24.log
│   │   │       └── fno_burgers_ema_h128_l8_m24_best.npz
│   │   └── ff0b719f90d149d49d96bc8b9b27e78d/
│   │       └── artifacts/
│   │           └── afno_heavy_test_best.npz
│   ├── 2/
│   │   ├── 101fe0b7688e4303a829a3dcf9d7bfa8/
│   │   │   └── artifacts/
│   │   │       ├── ns2d_fno_h32_l4_m8.log
│   │   │       └── ns2d_fno_h32_l4_m8_best.npz
│   │   └── d6877b1b2654460aaae90343a0496d8e/
│   │       └── artifacts/
│   │           └── validation_ns2d_cleanup_best.npz
│   ├── 3/
│   │   ├── 2800e885ed3f406cbd1b857dfa1f40dd/
│   │   │   └── artifacts/
│   │   │       └── smoke_test_2d_best.npz
│   │   ├── 303d458e47d14da7b3387f5e7b79c978/
│   │   │   └── artifacts/
│   │   │       ├── rfno2d_darcy2d_h32_l4_m8.log
│   │   │       └── rfno2d_darcy2d_h32_l4_m8_best.npz
│   │   ├── 365d5e0de857483f977d9d3501148397/
│   │   │   └── artifacts/
│   │   │       ├── rfno2d_darcy2d_h32_l4_m8_adapt.log
│   │   │       └── rfno2d_darcy2d_h32_l4_m8_adapt_best.npz
│   │   ├── 569846757b1e4791a71c2ccb9801c281/
│   │   │   └── artifacts/
│   │   │       ├── fedonet2d_darcy_h32_l2_m8_h1.log
│   │   │       └── fedonet2d_darcy_h32_l2_m8_h1_best.npz
│   │   ├── 5bb14a75e85143e29f7f25087212e71a/
│   │   │   └── artifacts/
│   │   │       ├── fedonet2d_darcy_h32_l4.log
│   │   │       └── fedonet2d_darcy_h32_l4_best.npz
│   │   ├── 7b6b50efb79d4eefaba32ec67f79cfd1/
│   │   │   └── artifacts/
│   │   │       ├── fno2d_darcy_h32_l4_m12_h1_aug.log
│   │   │       └── fno2d_darcy_h32_l4_m12_h1_aug_best.npz
│   │   ├── 7fc7b67f422d48459b86cf9fda4de340/
│   │   │   └── artifacts/
│   │   │       ├── fno2d_darcy_h32_l4_m8_h1.log
│   │   │       └── fno2d_darcy_h32_l4_m8_h1_best.npz
│   │   ├── 968bf7023a124d728e9a769ebc59a12b/
│   │   │   └── artifacts/
│   │   │       ├── transolver2d_darcy_s16_h24_l3.log
│   │   │       └── transolver2d_darcy_s16_h24_l3_best.npz
│   │   ├── abc12a20d416466b95ec4e8cc2f02cc2/
│   │   │   └── artifacts/
│   │   │       ├── fedonet2d_darcy_h32_l2_m8_h1_adapt.log
│   │   │       └── fedonet2d_darcy_h32_l2_m8_h1_adapt_best.npz
│   │   ├── b0b654187e3f4bf2bc185394e871a6b3/
│   │   │   └── artifacts/
│   │   │       ├── fno2d_darcy_h48_l6_m12_aug.log
│   │   │       └── fno2d_darcy_h48_l6_m12_aug_best.npz
│   │   ├── b7cb04e62ee74c7fae9b5e8326476472/
│   │   │   └── artifacts/
│   │   │       ├── autogen_darcy_2d_fno2d_h1_4991.log
│   │   │       └── autogen_darcy_2d_fno2d_h1_4991_best.npz
│   │   ├── d26aaa4bbf92492896d0150798bd90ae/
│   │   │   └── artifacts/
│   │   │       ├── transolver2d_darcy_h32_l4_s32_h1.log
│   │   │       └── transolver2d_darcy_h32_l4_s32_h1_best.npz
│   │   └── de4538a2e33a4120a0d19c01fea7c051/
│   │       └── artifacts/
│   │           ├── transolver2d_darcy_h32_l4_s32_h1_adapt.log
│   │           └── transolver2d_darcy_h32_l4_s32_h1_adapt_best.npz
│   ├── 4/
│   │   └── 7379f0f26da342ba8b2cf5512f0e00de/
│   │       └── artifacts/
│   │           ├── autogen_rayleigh_benard_2d_fno2d_h1adapt_2746.log
│   │           └── autogen_rayleigh_benard_2d_fno2d_h1adapt_2746_best.npz
│   ├── 415139728503581214/
│   │   ├── f487c4bd48de4f4b98dd180188324a57/
│   │   │   ├── artifacts/
│   │   │   ├── metrics/
│   │   │   │   ├── training_seconds
│   │   │   │   └── val_l2_rel
│   │   │   ├── params/
│   │   │   │   ├── hidden_dim
│   │   │   │   ├── lr
│   │   │   │   └── n_layers
│   │   │   ├── tags/
│   │   │   │   ├── benchmark
│   │   │   │   ├── exp_name
│   │   │   │   ├── mlflow.runName
│   │   │   │   ├── mlflow.source.name
│   │   │   │   ├── mlflow.source.type
│   │   │   │   ├── mlflow.user
│   │   │   │   └── model
│   │   │   └── meta.yaml
│   │   └── meta.yaml
│   ├── 5/
│   │   ├── 08d3eb38ad3543e3b3ea72b543aa0fd3/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h128_m32_l8_f1.log
│   │   │       └── rfno_kdv_h128_m32_l8_f1_best.npz
│   │   ├── 12244133c4c440a0bbdd13e6c80a91f9/
│   │   │   └── artifacts/
│   │   │       ├── hnn_kdv_h64_l4_f1_adapt.log
│   │   │       └── hnn_kdv_h64_l4_f1_adapt_best.npz
│   │   ├── 16fba21f31fe40a1842e6ae9d8cee6fd/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h128_m24_l10_f1.log
│   │   │       └── rfno_kdv_h128_m24_l10_f1_best.npz
│   │   ├── 206e3290c9a44f508b4308a471cd7f13/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h256_m24_l8_f1.log
│   │   │       └── rfno_kdv_h256_m24_l8_f1_best.npz
│   │   ├── 49ccb8972c394db28edb90b58067b28f/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h128_m24_l12_f1.log
│   │   │       └── rfno_kdv_h128_m24_l12_f1_best.npz
│   │   ├── 6b087efc74dc408ab765f3f512c96f29/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h128_m24_l10_f1_adapt.log
│   │   │       └── rfno_kdv_h128_m24_l10_f1_adapt_best.npz
│   │   ├── 72998ac24aff4c4fa01764967d39643a/
│   │   │   └── artifacts/
│   │   │       ├── rtfno_kdv_h128_l10_m24_f1_r2.log
│   │   │       └── rtfno_kdv_h128_l10_m24_f1_r2_best.npz
│   │   ├── 977c50210e144f03be4b54556a353332/
│   │   │   └── artifacts/
│   │   │       ├── rtfno_kdv_h128_l10_m24_f1_adapt.log
│   │   │       └── rtfno_kdv_h128_l10_m24_f1_adapt_best.npz
│   │   ├── a52e0e23a9ed48308082fd721c02f38d/
│   │   │   └── artifacts/
│   │   │       ├── fno_kdv_h256_m32_l8_f1.log
│   │   │       └── fno_kdv_h256_m32_l8_f1_best.npz
│   │   ├── b475d5c9b9d045ff98bca3c676a3ef2e/
│   │   │   └── artifacts/
│   │   │       ├── hnn_kdv_h64_l4_f1.log
│   │   │       └── hnn_kdv_h64_l4_f1_best.npz
│   │   ├── c3d252ccc6844db3ae9feb04346dedc4/
│   │   │   └── artifacts/
│   │   │       ├── rfno_kdv_h128_m24_l12_f1.log
│   │   │       └── rfno_kdv_h128_m24_l12_f1_best.npz
│   │   ├── c486e0920f374ab6aff09baa8041404f/
│   │   │   └── artifacts/
│   │   │       ├── ffno_kdv_h256_m32_l8_f1.log
│   │   │       └── ffno_kdv_h256_m32_l8_f1_best.npz
│   │   ├── c9bbe68c2b1749e8abdb2209e76e85c0/
│   │   │   └── artifacts/
│   │   │       ├── tfno_kdv_h128_l8_m24_f1.log
│   │   │       └── tfno_kdv_h128_l8_m24_f1_best.npz
│   │   ├── cbc7e7b893394cc0b47f047282caed66/
│   │   │   └── artifacts/
│   │   │       ├── energy_fno_kdv_h128_l8_m24_f1_r1.log
│   │   │       └── energy_fno_kdv_h128_l8_m24_f1_r1_best.npz
│   │   └── fa8e55dc92e746a68cf779fbc1e87755/
│   │       └── artifacts/
│   │           ├── energy_fno_kdv_h128_l8_m24_f1_r1.log
│   │           └── energy_fno_kdv_h128_l8_m24_f1_r1_best.npz
│   └── 6/
│       ├── 41709d212b0d41b4a3fe1c6e07a94f22/
│       │   └── artifacts/
│       │       ├── ssno_wave_h64_l4_m16_f1.log
│       │       └── ssno_wave_h64_l4_m16_f1_best.npz
│       ├── 4930a7ed6016418c825e16a034ba57a1/
│       │   └── artifacts/
│       │       ├── energy_fno_wave_h64_l8_m24_f1.log
│       │       └── energy_fno_wave_h64_l8_m24_f1_best.npz
│       ├── 6af3c131a637401ea11f8b11a89ad060/
│       │   └── artifacts/
│       │       ├── energy_fno_wave_h64_l8_m24_f1_adapt.log
│       │       └── energy_fno_wave_h64_l8_m24_f1_adapt_best.npz
│       ├── a406f7d4587e4f9b8f608af03bcba888/
│       │   └── artifacts/
│       │       ├── ssno_wave_h64_l4_m16_f1_adapt.log
│       │       └── ssno_wave_h64_l4_m16_f1_adapt_best.npz
│       ├── c562ff4728c746aba70fdd7285aa0984/
│       │   └── artifacts/
│       │       ├── energy_fno_wave_h64_l8_m24_f1.log
│       │       └── energy_fno_wave_h64_l8_m24_f1_best.npz
│       └── f4175ebb4d344ef6b87755649bdeabb2/
│           └── artifacts/
│               ├── energy_fno_wave_h64_l8_m24_f1_adapt.log
│               └── energy_fno_wave_h64_l8_m24_f1_adapt_best.npz
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
├── scripts/
│   ├── backfill_model_registry.py
│   ├── dvc_train.py
│   ├── gen_arch_nanobanana.py
│   └── gen_arch_viz.py
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
├── dvc.yaml
├── experiments.yaml
├── identify_missing.py
├── mlflow.db
├── model_architectures.md
├── model_registry.json
├── params.yaml
├── program.md
├── pyproject.toml
├── results.json
├── test_hf.py
├── test_openai.py
├── train.py
└── uv.lock
```
<!-- STRUCTURE_END -->
