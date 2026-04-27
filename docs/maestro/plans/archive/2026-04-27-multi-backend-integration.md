---
title: "Multi-Backend (CUDA & MLX) Integration"
created: "2026-04-27T08:30:00.000Z"
status: "draft"
total_phases: 5
estimated_files: 8
task_complexity: "medium"
---

# Multi-Backend Integration Plan

## Objective
Integrate native **Apple Silicon (MLX)** support alongside the existing **NVIDIA CUDA (PyTorch)** framework. This will allow the SciMLx framework and ASIL pipeline to run optimally on both platforms by selecting the best backend at runtime.

## Approach: A (Unified Dispatch)
- Use **PyTorch** for NVIDIA (CUDA) and CPU-only systems.
- Use **MLX** for Apple Silicon hardware.
- Maintain separate optimized trainers and model stubs for each framework.

## Phase 1: Multi-Backend Core Foundation
### Objective
Establish the hardware detection and framework selection logic.

### Agent: `coder`
### Files to Modify
- `core/device.py` — Add hardware detection logic to define `FRAMEWORK` (`'torch'` or `'mlx'`).
- `core/device.py` — Implement unified `to_array()` and `to_framework_device()` helpers.

---

## Phase 2: Framework-Agnostic Data Loading
### Objective
Ensure data loaders return the correct array/tensor type for the active backend.

### Agent: `coder`
### Files to Modify
- `data/prepare.py` — Update `make_dataloader` and `evaluate_l2_rel` to support MLX arrays.
- `data/benchmarks_ext.py` — (If necessary) Ensure compatibility with MLX arrays.

---

## Phase 3: Multi-Backend Trainer
### Objective
Implement a native MLX trainer while preserving the PyTorch trainer.

### Agent: `coder`, `tester`
### Files to Modify
- `core/trainer.py` — Refactor `Trainer` into `BaseTrainer`, `TrainerTorch`, and `TrainerMLX`.
- `core/trainer.py` — Implement `TrainerMLX.train()` using `mx.value_and_grad` and MLX optimizers.

---

## Phase 4: Dual-Backend Scaffolding
### Objective
Update the scaffolding logic to generate framework-optimized code.

### Agent: `coder`
### Files to Modify
- `core/scaffold.py` — Add MLX-native model templates.
- `core/scaffold.py` — Update `generate_stub` and `ModelGate.validate()` to handle both frameworks.

---

## Phase 5: ASIL Pipeline & Documentation
### Objective
Enable hardware-aware scaffolding in the ASIL pipeline and document the feature.

### Agent: `coder`, `technical_writer`
### Files to Modify
- `scripts/asil_scaffold.py` — Detect backend and pass to scaffolding logic.
- `README.md` — Add documentation for multi-backend support.
- `RESEARCH_BRAIN.md` — Update hardware limits and mandates for MLX.

## Verification
- Run `python train.py` on both NVIDIA (if available) and Apple Silicon (if available) and verify it selects the correct backend.
- Run `tests/test_asil_loop.py` to ensure the ASIL loop still functions correctly.
- Verify that `models/` stubs generated on Mac use `import mlx.nn as nn`.
