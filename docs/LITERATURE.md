# SciML Literature

## Foundational Neural Operators

- **FNO** — Li et al. (2020) "Fourier Neural Operator for Parametric Partial
  Differential Equations" arXiv:2010.08895
  _Baseline architecture in this repo._

- **DeepONet** — Lu et al. (2019/2021) "Learning nonlinear operators via
  DeepONet based on the universal approximation theorem of operators"
  Nature Machine Intelligence 2021.

- **PINNs** — Raissi et al. (2019) "Physics-informed neural networks: A deep
  learning framework for solving forward and inverse problems involving
  nonlinear PDEs" Journal of Computational Physics.

- **Neural ODE** — Chen et al. (2018) "Neural Ordinary Differential Equations"
  NeurIPS 2018.  Foundation for latent dynamics models.

## Architecture Advances

- **U-NO (UNO)** — Rahman et al. (2022) "U-NO: U-shaped Neural Operators"
  arXiv:2204.11127.  Encoder-decoder FNO with skip connections;
  ~20% better than flat FNO on standard benchmarks.
  _Implemented as `UNO1d` in `models/fno.py`._

- **WNO** — Tripura & Chakraborty (2022) "Wavelet Neural Operator for solving
  parametric PDEs in computational mechanics" arXiv:2205.02191.
  Replaces Fourier modes with Haar wavelets; handles non-periodic BCs.
  _Implemented as `WNO1d` in `models/wno.py`._

- **AFNO** — Guibas et al. (2022) "Adaptive Fourier Neural Operators: Efficient
  Token Mixers for Transformers" ICLR 2022.  Block-diagonal MLPs in Fourier
  space with softshrink sparsity.

- **Geo-FNO** — Li et al. (2023) "Fourier Neural Operator with Learned
  Deformations for PDEs on General Geometries" JMLR 2023.  Extends FNO to
  irregular meshes via input deformation.

- **GNOT** — Hao et al. (2023) "GNOT: A General Neural Operator Transformer
  for operator learning" ICML 2023.  Attention-based operator with
  heterogeneous inputs.

- **Transolver** — Wu et al. (2024) "Transolver: A Fast Transformer Solver for
  PDEs on General Geometries" ICML 2024.

## Physics-Informed Approaches

- **SplinePINN** — Jaggi et al. (2024) "3D Steady-State SplinePINNs for solving 
  Navier-Stokes" EPFL ML4Science.
  Integrates splines into PINNs for better local gradient control.

- **PINO** — Li et al. (2021) "Physics-Informed Neural Operator for Learning
  Partial Differential Equations" arXiv:2111.03794.  Adds PDE residual loss to
  FNO; consistently improves data efficiency.

- **PI-DeepONet** — Wang et al. (2022) "Improved architectures and training
  algorithms for deep operator networks" Journal of Scientific Computing.

- **FBPINN** — Moseley et al. (2023) "Finite Basis Physics-Informed Neural
  Networks (FBPINNs): a scalable domain decomposition approach for solving
  differential equations" Advances in Computational Mathematics.  Overlapping
  subdomain decomposition; best PINN variant for NS in PINNacle 2024 (0.0245).
  _Source: MLiS-cited PINNacle benchmark._

- **Self-Supervised PINN** — Basir et al. (2022) "Physics and Equality
  Constrained Artificial Neural Networks: application to partial differential
  equations" SIAM Journal.

## Benchmarks & Evaluation

- **PDEBench (2024 update)** — Takamoto et al. (2022/2024) "PDEBench: An
  Extensive Benchmark for Scientific Machine Learning" NeurIPS 2022; 2024
  update adds new FNO/U-Net/PINN baselines, fRMSE metric, foundation model
  comparisons. FNO: Burgers 0.001, KdV 0.040, Wave 0.042, NS-INS 0.010.
  _Source: MLiS 2024 benchmark reference._

- **PINNacle** — Hao et al. (2024) "PINNacle: A Comprehensive Benchmark of
  Physics-Informed Neural Networks for Solving PDEs" NeurIPS 2024 Datasets
  & Benchmarks. Evaluates 10 PINN variants on 20+ PDEs. Key result: FBPINN
  best on NS (0.0245); all methods fail Allen-Cahn (~0.50); Burgers ~0.013.
  arXiv:2306.08827.  _Source: MLiS conference series._

- **MCNP Operator Benchmark** — Kovachki et al. (2021) "Neural Operator:
  Learning Maps Between Function Spaces" arXiv:2108.08481.

## Latent Dynamics & Multi-Scale

- **LOCA** — Kissas et al. (2022) "Learning operators with coupled attention"
  JMLR 2022.  Attention-based operator learning for multi-physics.

- **CORAL** — Serrano et al. (2023) "CORAL: Continuous Representation Learning
  in Reduced Space for PDEs" NeurIPS 2023.

- **MP-PDE** — Brandstetter et al. (2022) "Message Passing Neural PDE Solvers"
  ICLR 2022.  Graph-based solver on unstructured meshes.

## Uncertainty Quantification

- **Ensemble FNO** — Rahman et al. (2023) "Neural Operator Ensembles for
  Uncertainty Quantification in PDEs."

- **Bayesian DeepONet** — Lin et al. (2021) "Operator learning for predicting
  multiscale bubble growth dynamics" Journal of Chemical Physics.

## MLiS Conference (Machine Learning in Science)

Annual interdisciplinary conference hosted in Tübingen.  Key results pipeline:

- **MLiS 2024** (Aug 21–22, Glasgow) — Themes: deep learning for brain health,
  AI for computational science, ML for animal behaviour, multi-objective
  optimisation for sustainability (GALLANT project), invertible networks for
  inverse problems.

- **MLiS 2025** (Sep 30 – Oct 1, Tübingen) — Invited talks:
  - Mario Krenn: "Towards an Artificial Muse for New Ideas in Science"
  - Harry Scells: "Automating Systematic Reviews"
  - Philine Widmer: "AI in Economics: Research Tool and Research Subject"
  _Proceedings will include AI4Science and NeurOp submissions._

- **NeurIPS ML4PS workshop** (ongoing) — "Machine Learning and the Physical
  Sciences"; primary venue for neural operator + PINN results.

## Learned Preconditioners

- **L-BFGS Preconditioner** — Lötzsch et al. (2022) "Learning to solve PDEs
  with finite elements" arXiv:2207.05398.

- **NeuralIF** — Tagasovska et al. (2023) "NeuralIF: Neural Incomplete
  Factorization Preconditioners" arXiv:2309.12361.
9.12361.
