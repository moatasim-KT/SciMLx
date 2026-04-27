---
title: "ASIL Pipeline Implementation Plan"
design_ref: "docs/maestro/plans/2026-04-27-asil-pipeline-design.md"
created: "2026-04-27T07:55:00.000Z"
status: "draft"
total_phases: 5
estimated_files: 8
task_complexity: "medium"
---

# ASIL Pipeline Implementation Plan

## Plan Overview

- **Total phases**: 5
- **Agents involved**: `technical_writer`, `architect`, `coder`, `tester`, `code_reviewer`
- **Estimated effort**: Moderate implementation effort focusing on CLI script development and framework integration.

## Dependency Graph

```text
Phase 1 (Foundation)
    |
Phase 2 (Ideation Engine)
    |
Phase 3 (Scaffolding Bridge)
    |
Phase 4 (Validation & Integration)
    |
Phase 5 (Documentation & Polish)
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|-------|--------|-----------|-------------|-------|
| 1     | Phase 1 | Sequential | 1 | Protocol & Templates |
| 2     | Phase 2, 3 | Sequential | 2 | Core Scripting |
| 3     | Phase 4 | Sequential | 1 | E2E Testing |
| 4     | Phase 5 | Sequential | 1 | Documentation |

## Phase 1: Protocol & Templates Foundation

### Objective
Establish the directory structure and standardized templates for Research Proposals.

### Agent: `technical_writer`
### Parallel: No

### Files to Create
- `docs/proposals/TEMPLATE.md` — Standardized Markdown template for "Novel Idea Proposals" including Hypothesis, Grounding, and Scaffolding Specs.
- `docs/proposals/.gitkeep` — Initialize the proposals directory.

### Implementation Details
The template must include a frontmatter section for metadata (target PDE, estimated complexity, citations) to allow for programmatic parsing in Phase 3.

### Validation
- Verify directory exists: `ls docs/proposals/`
- Verify template content follows the Deep design rationale markers.

### Dependencies
- Blocked by: None

---

## Phase 2: Ideation Engine Implementation

### Objective
Implement the `scripts/asil_ideate.py` script that synthesizes research into novel proposals.

### Agent: `architect` (Design), `coder` (Implementation)
### Parallel: No

### Files to Create
- `scripts/asil_ideate.py` — CLI script that fetches ArXiv data, cross-references `RESEARCH_BRAIN.md`, and generates a `docs/proposals/YYYY-MM-DD-novel-idea.md` artifact.

### Implementation Details
- Import `update_paper_registry` from `core/brain_distiller.py` for ArXiv fetching.
- Use a high-reasoning LLM prompt (reflecting the "Artistic Scientist" persona) to synthesize novel architectural combinations.
- Rationale annotations: The script must include a `--novelty` flag to control the degree of architectural divergence.

### Validation
- Run `python scripts/asil_ideate.py --keywords "Navier-Stokes" --novelty high`
- Confirm a valid Proposal MD is generated in `docs/proposals/`.

### Dependencies
- Blocked by: Phase 1

---

## Phase 3: Scaffolding Bridge Integration

### Objective
Connect approved proposals to the `scaffold.py` and `brain_distiller.py` automation logic.

### Agent: `coder`
### Parallel: No

### Files to Create
- `scripts/asil_scaffold.py` — CLI script that takes an approved Proposal MD and triggers the model generation and registration pipeline.

### Implementation Details
- Parse the Proposal MD to generate a temporary `docs/papers/` YAML.
- Call `ModelGate.validate()` and `ModelGate.register_and_queue()` from `core/scaffold.py`.
- Integration Seam: Ensure the script appends the new hypothesis to the "Hypothesis Tracking" section of `RESEARCH_BRAIN.md`.

### Validation
- Run `python scripts/asil_scaffold.py --proposal docs/proposals/test-proposal.md`
- Verify model code exists in `models/` and entry exists in `experiments.yaml`.

### Dependencies
- Blocked by: Phase 2

---

## Phase 4: Validation & Integration Testing

### Objective
Ensure the end-to-end ASIL loop is robust and framework-compliant.

### Agent: `tester`
### Parallel: No

### Files to Create
- `tests/test_asil_loop.py` — Integration test covering fetching -> ideation -> scaffolding -> registration.

### Validation
- Run `pytest tests/test_asil_loop.py`
- Verify clean cleanup of test models and experiment entries.

### Dependencies
- Blocked by: Phase 3

---

## Phase 5: Documentation & Polish

### Objective
Finalize user-facing documentation and the "Agentic Scientist" skill configuration.

### Agent: `technical_writer`
### Parallel: No

### Files to Modify
- `README.md` — Add section on "Agentic Scientist Mode (ASIL)".
- `RESEARCH_BRAIN.md` — Initialize the "Hypothesis Tracking" section.

### Validation
- Verify all links and instructions are correct.

### Dependencies
- Blocked by: Phase 4

---

## File Inventory

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `docs/proposals/TEMPLATE.md` | 1 | Standardize Proposal format |
| 2 | `scripts/asil_ideate.py` | 2 | Autonomous ideation script |
| 3 | `scripts/asil_scaffold.py` | 3 | Scaffolding bridge |
| 4 | `tests/test_asil_loop.py` | 4 | Integration testing |
| 5 | `README.md` | 5 | User documentation |

## Risk Classification

| Phase | Risk | Rationale |
|-------|------|-----------|
| 2 | MEDIUM | High reliance on LLM creativity and ArXiv API stability. |
| 3 | MEDIUM | Fragile integration with scaffolding registration if schema drifts. |

## Execution Profile

```text
Execution Profile:
- Total phases: 5
- Parallelizable phases: 0 (Strict sequential dependency for foundation and core logic)
- Sequential-only phases: 5
- Estimated parallel wall time: N/A
- Estimated sequential wall time: ~4-6 hours of agent execution

Note: This implementation builds the infrastructure for the Human-Guided (Mode A) pipeline.
```

## Cost Estimation

| Phase | Agent | Model | Est. Input | Est. Output | Est. Cost |
|-------|-------|-------|-----------|------------|----------|
| 1 | `technical_writer` | gemini-2.5-flash | 1,000 | 500 | $0.01 |
| 2 | `coder` | gemini-2.5-pro | 10,000 | 2,000 | $0.18 |
| 3 | `coder` | gemini-2.5-pro | 8,000 | 1,500 | $0.14 |
| 4 | `tester` | gemini-2.5-flash | 5,000 | 1,000 | $0.01 |
| 5 | `technical_writer` | gemini-2.5-flash | 2,000 | 1,000 | $0.01 |
| **Total** | | | **26,000** | **6,000** | **$0.35** |
