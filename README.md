# SciMLx: Agent-Driven Autoresearch for Neural Operators

SciMLx is a self-driving experiment harness designed for high-performance neural operator research. By bridging the gap between scientific literature and high-fidelity simulation, SciMLx enables autonomous discovery and optimization of PDE solvers across complex physical domains.

## Accelerating Scientific Discovery
The core of SciMLx is an intelligent, agent-driven loop that automates the entire research workflow: from literature-informed hypothesis generation and neural architecture scaffolding to automated training, rigorous benchmarking, and experiment refinement.

Enabled by high-performance neural operator research, the platform provides:
- **Autonomous Research Campaigns**: Overnight orchestration of train-evaluate-diagnose cycles.
- **Advanced Operator Architectures**: Access to 30+ cutting-edge models, including FNO, MambaNO, and KANs.
- **Hardware-Accelerated Benchmarking**: 15+ high-fidelity PDE simulation suites optimized for NVIDIA CUDA.
- **Scientific Implementation (SI) Layer**: Built-in support for dimensional analysis (`units.py`), Lie Algebra foundations (`lie_math.py`), and frequency-aware spectral loss modulation.

## Quick Start
```bash
# Clone the repository
git clone https://github.com/moatasimfarooque/autoresearch-mlx scimlx
cd scimlx

# Install dependencies
uv sync

# Run a training session on a benchmark
uv run train.py --benchmark burgers_1d --model FNO

# Launch the autonomous research loop
uv run autorun.py --auto --commit
```

## Documentation
For deeper insights into the project, consult our technical resources:
- [Architecture](./docs/ARCHITECTURE.md)
- [Benchmarks](./docs/BENCHMARKS.md)
- [Research Brain](./RESEARCH_BRAIN.md)
- [SOTA Report](./docs/SOTA.md)

## Contributions
We welcome contributions from the scientific and engineering community. Whether you are improving operator architectures, adding new PDE benchmarks, or refining the autonomous agent loop, your input helps advance the state of scientific machine learning. Please refer to `CONTRIBUTING.md` for our submission guidelines.

## Authors
- Moatasim Farooque

## Acknowledgments
We thank the open-source community for the foundational research in neural operators, PDE benchmarking, and agentic workflows that have enabled the development of SciMLx.

## License
This project is released under an Open License. Please see the [LICENSE](./docs/LICENSE) file for details.


<!-- STRUCTURE_START -->
```text
autoresearch-mlx/
├── agents/
│   └── skills/
│       └── SciMLx/
│           └── SKILL.md
├── artifacts/
│   └── presentation/
│       ├── assets/
│       │   └── gen/
│       │       └── workflow.png
│       └── src/
├── core/
│   ├── __init__.py
│   ├── arxiv_agent.py
│   ├── audio.py
│   ├── brain_distiller.py
│   ├── deployment.py
│   ├── device.py
│   ├── diagnostics.py
│   ├── dp_federated.py
│   ├── heat_kernels.py
│   ├── hpo.py
│   ├── hypothesis.py
│   ├── lie_math.py
│   ├── loader.py
│   ├── losses.py
│   ├── mlflow_integration.py
│   ├── model_versioning.py
│   ├── oracle_constants.py
│   ├── paper_registry.py
│   ├── readme_hook.py
│   ├── research_plugins.py
│   ├── results_store.py
│   ├── scaffold.py
│   ├── spectral_governor.py
│   ├── tracker.py
│   ├── trainer.py
│   ├── units.py
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
│   ├── benchmarks/
│   │   ├── allen_cahn_2d.md
│   │   ├── burgers_1d.md
│   │   ├── burgers_nu_001.md
│   │   ├── burgers_nu_01.md
│   │   ├── darcy_2d.md
│   │   ├── elasticity_2d.md
│   │   ├── euler_1d.md
│   │   ├── kdv_1d.md
│   │   ├── mhd_2d.md
│   │   ├── multiphysics_2d.md
│   │   ├── ns_2d.md
│   │   ├── ns_hre_2d.md
│   │   ├── pdebench_2d.md
│   │   ├── poisson_2d.md
│   │   ├── radiative_2d.md
│   │   ├── reionization_1d.md
│   │   ├── swe_2d.md
│   │   ├── wave_1d.md
│   │   └── wavebench_2d.md
│   ├── maestro/
│   │   ├── plans/
│   │   │   └── archive/
│   │   │       ├── 2026-04-27-asil-pipeline-design.md
│   │   │       ├── 2026-04-27-asil-pipeline-impl-plan.md
│   │   │       ├── 2026-04-27-multi-backend-integration.md
│   │   │       ├── 2026-04-27-scimlx-vision-refactor-design.md
│   │   │       ├── 2026-04-27-scimlx-vision-refactor-impl-plan.md
│   │   │       ├── 2026-04-30-readme-overhaul-design.md
│   │   │       ├── 2026-04-30-readme-overhaul-impl-plan.md
│   │   │       ├── 2026-04-30-readme-scientific-overhaul-design.md
│   │   │       └── 2026-04-30-readme-scientific-overhaul-impl-plan.md
│   │   └── state/
│   │       ├── archive/
│   │       │   ├── 2026-04-30-readme-scientific-overhaul.md
│   │       │   └── 2026-04-30-update-readme-md.md
│   │       └── 2026-04-30-implement-improvements.design-gate.json
│   ├── papers/
│   │   ├── afno_2022.yaml
│   │   ├── augmentation_2023.yaml
│   │   ├── cosmic_reionization_pinn_2023.yaml
│   │   ├── curriculum_2009.yaml
│   │   ├── deeponet_2021.yaml
│   │   ├── ensemble_uq_2023.yaml
│   │   ├── fbpinn_2023.yaml
│   │   ├── ffno_2023.yaml
│   │   ├── fno_2020.yaml
│   │   ├── gnot_2023.yaml
│   │   ├── h1_loss.yaml
│   │   ├── hnn_2019.yaml
│   │   ├── inverse_pinn_2023.yaml
│   │   ├── mambano_2024.yaml
│   │   ├── memno_2025.yaml
│   │   ├── modal_pinn_2024.yaml
│   │   ├── mppde_2022.yaml
│   │   ├── neural_ode_ude_2020.yaml
│   │   ├── neural_pde_solver_2023.yaml
│   │   ├── packed_ensemble_2023.yaml
│   │   ├── pdebench_2024.yaml
│   │   ├── physicsnemo_2024.yaml
│   │   ├── pikan_2025.yaml
│   │   ├── pinnacle_2024.yaml
│   │   ├── pino_2021.yaml
│   │   ├── pod_dl_rom_2023.yaml
│   │   ├── rfno_2024.yaml
│   │   ├── sar_2026.yaml
│   │   ├── spline_pinn_2024.yaml
│   │   ├── ssm_s4_2022.yaml
│   │   ├── tfno_2022.yaml
│   │   ├── time_marching_deeponet_2025.yaml
│   │   ├── transolver_2024.yaml
│   │   ├── uno_2022.yaml
│   │   └── wno_2022.yaml
│   ├── proposals/
│   │   ├── TEMPLATE.md
│   │   ├── test_proposal.md
│   │   └── test_refactor_scaffold.md
│   ├── ARCHITECTURE.md
│   ├── BENCHMARKS.md
│   ├── CONTRIBUTING.md
│   ├── LICENSE
│   ├── LITERATURE.md
│   ├── SOTA.md
│   ├── TERMINOLOGY.md
│   └── VISION_2026.md
├── models/
│   ├── layers/
│   │   └── equivariant.py
│   ├── __init__.py
│   ├── afno.py
│   ├── attention_fno.py
│   ├── axial_attention.py
│   ├── chebyshev_kan.py
│   ├── deeponet.py
│   ├── dualmodeltest_mlx.py
│   ├── dualmodeltest_torch.py
│   ├── fedonet.py
│   ├── fno.py
│   ├── gato.py
│   ├── gnot.py
│   ├── hano.py
│   ├── hnn.py
│   ├── hybrid_decoder_deeponet.py
│   ├── hybrid_fno_deeponet.py
│   ├── kan.py
│   ├── kan_refiner.py
│   ├── mamba_no.py
│   ├── mambafno.py
│   ├── mambafno_mlx.py
│   ├── mambafno_torch.py
│   ├── mem_no.py
│   ├── mff.py
│   ├── mff_mlx.py
│   ├── mff_torch.py
│   ├── neural_ode.py
│   ├── pacmann.py
│   ├── pinn.py
│   ├── s4d.py
│   ├── sar.py
│   ├── sno.py
│   ├── ssno.py
│   ├── testnet_mlx.py
│   ├── testnet_torch.py
│   ├── tfno.py
│   ├── time_deeponet.py
│   ├── transolver.py
│   ├── vsmno.py
│   └── wno.py
├── notebooks/
├── sciml_mlx/
│   ├── core/
│   ├── dashboard/
│   ├── data/
│   │   └── simulations/
│   ├── models/
│   └── results.db
├── scripts/
│   ├── gcp/
│   │   ├── setup_vm.sh
│   │   └── submit_vertex.py
│   ├── maintenance/
│   │   ├── backfill_model_registry.py
│   │   ├── gen_arch_nanobanana.py
│   │   └── gen_arch_viz.py
│   ├── asil_ideate.py
│   ├── asil_scaffold.py
│   └── dvc_train.py
├── tests/
│   ├── integration/
│   │   ├── test_asil_loop.py
│   │   ├── test_federated.py
│   │   ├── test_gato.py
│   │   ├── test_mff.py
│   │   ├── test_parity.py
│   │   └── test_si_modules.py
│   ├── smoke/
│   └── unit/
│       ├── test_arxiv_agent.py
│       ├── test_heat_kernels.py
│       ├── test_lie_math.py
│       ├── test_oracle_constants.py
│       ├── test_spectral_governor.py
│       └── test_units.py
├── Dockerfile
├── README.md
├── RESEARCH_BRAIN.md
├── agent_loop.py
├── analyze.py
├── auto_suggest.py
├── autorun.py
├── dvc.yaml
├── experiments.yaml
├── generate_diagram.py
├── mlflow.db
├── model_registry.json
├── params.yaml
├── pyproject.toml
├── results.db
├── results.json
├── train.py
└── uv.lock
```
<!-- STRUCTURE_END -->
