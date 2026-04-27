---
title: "Agentic Scientist Ideation Loop (ASIL) Pipeline"
created: "2026-04-27T07:45:00.000Z"
status: "approved"
authors: ["TechLead", "User"]
type: "design"
design_depth: "deep"
task_complexity: "medium"
---

# ASIL Pipeline Design Document

## Problem Statement

The current SciMLx framework is a powerful repository of neural operator benchmarks, but it lacks a proactive, creative mechanism for architectural evolution. Bridging the massive SOTA gaps identified in `RESEARCH_BRAIN.md` requires more than just re-implementing existing papers; it demands an **Agentic Scientist** capable of synthesizing novel, unexplored "Scientific Intelligence" models. The challenge is to build a pipeline that empowers agents to act as artistic researchers—generating groundbreaking ideas—while maintaining the scientific rigor and human-guided oversight required for production-grade physics solvers.

**Rationale Annotations:**
- **Agentic Scientist Persona** — *Chosen to satisfy the requirement for an expert, creative, and artistic researcher who goes beyond keyword monitoring to invent novel ideas.*
- **Human-Guided Gate (Mode A)** — *Balanced against the desire for novelty to ensure that "unexplored" ideas are grounded in physical consistency before significant compute is expended.*

## Requirements

### Functional Requirements

1. **REQ-1: Novel Ideation Engine** — The agent must autonomously search for and synthesize research from ArXiv and `RESEARCH_BRAIN.md` to propose novel architectures that have not been explicitly documented.
2. **REQ-2: Structured Research Proposals** — For every novel idea, the agent must generate a "Research Proposal" document including the hypothesis, research grounding, and proposed implementation strategy.
3. **REQ-3: Automated Paper Distillation** — The system must convert research papers (PDF/URL) into the canonical `SciMLx` brain-distillation YAML format using `core/brain_distiller.py`.
4. **REQ-4: Intelligent Scaffolding** — Upon human approval, the agent must generate a functional model plugin in `models/` that adheres to framework interfaces.

### Non-Functional Requirements

1. **NFR-1: Artistic Scientific Rigor** — While the agent is encouraged to be "artistic," all proposed models must attempt to respect physical symmetries and conservation laws.
2. **NFR-2: Knowledge Persistence** — Every research attempt, whether successful or failed, must be logged in `RESEARCH_BRAIN.md` to build the system's long-term intelligence.

### Constraints

- **Human-in-the-Loop** — No code execution or training on "Novel Ideas" may occur without explicit human approval of the Research Proposal.
- **PyTorch/CUDA Focus** — All generated code must prioritize the high-performance CUDA backend as outlined in `ARCHITECTURE.md`.

## Approach

### Selected Approach

**Agentic Scientist Ideation Loop (ASIL)**

The ASIL approach transforms the agent into an active research partner. It Cross-references SOTA gaps with ArXiv releases for "Artistic Hybridization"—proposing non-obvious architectural combinations to solve specific physics challenges.

### Alternatives Considered

#### Standard Distiller
- **Description**: One-to-one mapping of existing papers to code.
- **Pros**: High grounding, lower complexity.
- **Cons**: No novelty, fails the "Artistic" requirement.
- **Rejected Because**: It focuses solely on existing research and doesn't propose novel unexplored ideas.

### Decision Matrix

| Criterion | Weight | Approach A (ASIL) | Approach B (Standard) |
|-----------|--------|-------------------|-----------------------|
| Novelty/Creativity | 40% | 5: Explicitly encourages synthesis | 2: Limited to re-implementation |
| Scientific Rigor | 30% | 4: Proposal gate grounding | 5: Grounded in peer-reviewed code |
| Effort/Complexity | 10% | 3: Complex orchestration | 5: Straightforward mapping |
| SOTA Gap Impact | 20% | 5: Potential for breakthroughs | 3: Incremental improvements |
| **Weighted Total** | | **4.5** | **3.4** |

## Architecture

### Component Diagram

```text
[ArXiv / SOTA] ----> [Agentic Scientist] <----> [RESEARCH_BRAIN.md]
                            |
                            v
                  [Novel Research Proposal] <---- (User Approval Gate)
                            |
           +----------------+----------------+
           |                                 |
           v                                 v
   [Brain Distiller] ---------------> [Model Scaffolder]
     (YAML Spec)                         (Python/PyTorch)
           |                                 |
           +----------------+----------------+
                            |
                            v
                        [models/] <--- (Final Plugin)
```

### Data Flow

Research inputs from ArXiv are synthesized by the Agentic Scientist into a Proposal. Once the user approves the Proposal via the gate, the Brain Distiller generates a YAML specification which is then used by the Scaffolder to produce PyTorch code in the `models/` directory.

### Key Interfaces

- **IdeaProposal**: Markdown/YAML artifact with hypothesis and citations. (REQ-1, REQ-2)
- **DistillationYAML**: Standard format for `brain_distiller.py`. (REQ-3)

## Agent Team

| Phase | Agent | Parallel | Deliverables |
|-------|-------|----------|--------------|
| 1 | `architect` | No | System specs, integration plan |
| 2 | `technical_writer` | No | Proposal templates, protocols |
| 3 | `coder` | No | Automation scripts (Fetch/Scaffold) |
| 4 | `tester` | No | End-to-end validation |
| 5 | `code_reviewer` | No | Quality gate |

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Architectural Hallucination | HIGH | MEDIUM | Proposal gate requires cited grounding. |
| Physical Inconsistency | HIGH | LOW | Enforce symmetry-strict scaffolding. |
| Compute Runaway | MEDIUM | MEDIUM | Human review of Complexity Estimates. |

## Success Criteria

1. Agent successfully generates a Research Proposal linking two disparate concepts.
2. Human can trigger scaffolding of a functional model in < 5 minutes.
3. Every proposal is logged in `RESEARCH_BRAIN.md`.
4. Generated models pass framework compliance checks.
