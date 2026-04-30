---
title: "SciMLx 2026 Vision — Multi-Backend Refactor Implementation Plan"
design_ref: "docs/maestro/plans/2026-04-27-scimlx-vision-refactor-design.md"
created: "2026-04-27T11:30:00.000Z"
status: "draft"
total_phases: 6
estimated_files: 30
task_complexity: "complex"
---

# SciMLx 2026 Vision — Multi-Backend Refactor Implementation Plan

## Plan Overview

- **Total phases**: 6
- **Agents involved**: `devops_engineer`, `coder`, `refactor`, `tester`, `technical_writer`
- **Estimated effort**: Complex refactor spanning core logic, math foundations, and 20+ new model/utility modules.

## Dependency Graph

```text
Phase 1: Branch & Core Dispatch
    |
Phase 2: Math Tier (SciMLTensor, LieAlgebra)
    |
Phase 3: Dual-Backend Scaffolder
    |
Phase 4: SI Module Implementation (Batch A: Foundations & HPO)
    |
Phase 5: SI Module Implementation (Batch B: Models & Production)
    |
Phase 6: Integration, Testing & Docs
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|-------|--------|-----------|-------------|-------|
| 1     | Phase 1 | Sequential | 1 | Foundation & Branching |
| 2     | Phase 2, 3 | Parallel | 2 | Core Math & Scaffolding |
| 3     | Phase 4, 5 | Parallel | 4+ | Module Implementation |
| 4     | Phase 6 | Sequential | 2 | Final Polish |

## Phase 1: Foundation & Branching

### Objective
Establish the dedicated feature branch and refactor `core/device.py` to support robust multi-backend dispatch.

### Agent: `devops_engineer`
### Parallel: No

### Files to Create
- None

### Files to Modify
- `core/device.py` — Add `get_framework_backend()` and explicit `MPS` vs `CUDA` vs `MLX` logic.
- `pyproject.toml` — Ensure `torch` and `mlx` (optional) dependencies are correctly grouped.

### Implementation Details
1. Create branch `feat/SI-MultiBackend-Architecture`.
2. Refactor `core/device.py` to expose `FRAMEWORK` constant and `to_array()` helper that handles `mx.array` vs `torch.tensor` conversions.

### Validation
- `python -c "from core.device import FRAMEWORK; print(FRAMEWORK)"`
- `git branch` output.

### Dependencies
- Blocked by: None
- Blocks: Phase 2, 3

---

## Phase 2: Scientific Math Tier

### Objective
Implement the backend-agnostic math abstractions for unit-aware tensors and Lie Algebra foundations.

### Agent: `coder`
### Parallel: Yes

### Files to Create
- `core/lie_math.py` — High-level Lie Algebra operations using NumPy/SciPy.
- `core/heat_kernels.py` — Mesh-based heat kernel signature calculation.

### Files to Modify
- `core/units.py` — Refactor `SciMLTensor` to inherit from a generic wrapper and delegate to `torch` or `mlx` based on `FRAMEWORK`.

### Implementation Details
1. Refactor `SciMLTensor` to support `mx.array` wrapping for MLX backend.
2. Implement `LieLatentSpace` logic in `core/lie_math.py`.

### Validation
- `pytest tests/unit/test_units.py` (to be created)
- `pytest tests/unit/test_lie_math.py` (to be created)

### Dependencies
- Blocked by: Phase 1
- Blocks: Phase 4

---

## Phase 3: Dual-Backend Scaffolder Update

### Objective
Upgrade the scaffolding logic to automatically generate both Torch and MLX stubs for new operator proposals.

### Agent: `refactor`
### Parallel: No

### Files to Modify
- `core/scaffold.py` — Update `generate_stub` to use a template engine that produces `models/{name}_torch.py` and `models/{name}_mlx.py`.
- `scripts/asil_scaffold.py` — Update to trigger dual-backend generation.

### Implementation Details
1. Define `TORCH_TEMPLATE` and `MLX_TEMPLATE` inside `core/scaffold.py`.
2. Update `ModelGate` to smoke-test both stubs.

### Validation
- `python scripts/asil_scaffold.py --proposal docs/proposals/test_proposal.md`
- Verify both `.py` files exist in `models/`.

### Dependencies
- Blocked by: Phase 1
- Blocks: Phase 5

---

## Phase 4: SI Modules — Foundation & HPO (Batch A)

### Objective
Implement the first batch of Scientific Intelligence modules focused on foundations and agentic HPO.

### Agent: `coder`
### Parallel: Yes

### Files to Create
- `core/spectral_governor.py` — Framework-agnostic Spectral Bias logic.
- `core/oracle_constants.py` — Buckingham Pi Theorem analyzer.
- `core/arxiv_agent.py` — Refactored ArXiv integration.

### Implementation Details
1. Move `SpectralBiasGovernor` from `core/trainer.py` to `core/spectral_governor.py`.
2. Implement `MutualInformationScore` in `core/oracle_constants.py`.

### Validation
- Unit tests for each new core module.

### Dependencies
- Blocked by: Phase 2
- Blocks: Phase 6

---

## Phase 5: SI Modules — Models & Production (Batch B)

### Objective
Implement the second batch of modules focused on production-grade operators and deployment.

### Agent: `coder`
### Parallel: Yes

### Files to Create
- `models/mff.py` — Multi-Fidelity Fusion (Refactor).
- `models/gato.py` — Geometry-Aware Transformer Operator.
- `core/dp_federated.py` — Differential Privacy logic.

### Implementation Details
1. Implement `GeometricAttention` using the heat kernels from Phase 2.
2. Refactor existing `mff.py` to support the dual-backend stub pattern.

### Validation
- Integration tests verifying `out.shape` for new models.

### Dependencies
- Blocked by: Phase 2, 3
- Blocks: Phase 6

---

## Phase 6: Integration, Testing & Docs

### Objective
Finalize the test hierarchy migration, perform cross-backend parity checks, and update documentation.

### Agent: `tester`, `technical_writer`
### Parallel: No

### Files to Modify
- `README.md` — Update with Multi-Backend usage.
- `docs/ARCHITECTURE.md` — Document the 3-tier SI layer.
- `tests/` — Move files to `tests/unit/` and `tests/integration/`.

### Implementation Details
1. Move existing tests to the new directory structure.
2. Implement `tests/integration/test_parity.py` that asserts `torch_out ≈ mlx_out`.

### Validation
- `pytest tests/` (full suite pass).

### Dependencies
- Blocked by: Phase 4, 5
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `core/device.py` | 1 | Multi-backend dispatch logic. |
| 2 | `core/units.py` | 2 | Refactored SciMLTensor. |
| 3 | `core/lie_math.py` | 2 | Lie Algebra foundations. |
| 4 | `core/scaffold.py` | 3 | Dual-backend stub templates. |
| 5 | `core/spectral_governor.py" | 4 | Middleware for frequency-aware loss. |
| 6 | `core/oracle_constants.py` | 4 | Dimensional analysis agent. |
| 7 | `models/gato.py` | 5 | Geometry-aware operator implementation. |

## Risk Classification

| Phase | Risk | Rationale |
|-------|------|-----------|
| 1 | HIGH | Core infrastructure change; potential to break existing experiments. |
| 2 | MEDIUM | Math complexity in unit-aware tensor wrapping. |
| 4 | MEDIUM | Many small modules requiring precise scientific logic. |

## Execution Profile

```text
Execution Profile:
- Total phases: 6
- Parallelizable phases: 4 (Phase 2, 3, 4, 5)
- Sequential-only phases: 2 (Phase 1, 6)
- Estimated parallel wall time: ~4 hours
- Estimated sequential wall time: ~8 hours

Note: Native parallel execution currently runs agents in autonomous mode.
All tool calls are auto-approved without user confirmation.
```
