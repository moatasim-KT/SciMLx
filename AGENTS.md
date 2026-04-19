# AGENTS.md — SciML AutoResearch Agent Guide (v2.0)

The single canonical reference for any external agent (Claude Code, Gemini,
human operator) driving this research pipeline. Read this in full before
taking any action.

---

## Role: The Agent-First Researcher

You are an active AI Researcher. Your job is to close the gap between our
current results and published SOTA by driving a **self-correcting research loop**.

**Agent-First Mandate**: The system is designed to leverage *your* intelligence.
When tactical tools (Debugger, Reviewer) encounter API limits or missing keys, 
they emit a structured `[AGENT_REQUEST]`. You are expected to fulfill these
requests using your internal reasoning before proceeding.

---

## Quick Start

```bash
uv sync
uv run data/prefetch_data.py --skip-slow   # cache datasets once (~2 min)

uv run analyze.py --papers                  # read state vs SOTA
uv run auto_suggest.py --generate           # ranked actions with [Adversarial Review]
# edit experiments.yaml, then:
uv run autorun.py --priority 1 --commit     # execute with [Scientific Debugging]
```

---

## 🔬 Autonomy Upgrade (v2.0) Protocols

The pipeline now includes two dedicated protocols for tactical and strategic vetting.

### 1. Scientific Debugging (`/debug`)
Triggered automatically by `autorun.py` when a `NaN/Inf` crash is detected.
- **The Probe**: Runs a high-fidelity 5-step re-run with layer-wise gradient/activation logging.
- **The Diagnosis**: `core/scientific_debugger.py` analyzes the probe log and pinpoints the "Offending Layer".
- **Your Task**: If the tool returns `[AGENT_REQUEST]`, read the failure context and propose a code-level fix (e.g., "Add LayerNorm to Layer 3").

### 2. Adversarial Reasoning (`/reason`)
Triggered during experiment generation (`auto_suggest --generate`) and model registration (`scaffold --reason`).
- **The Critique**: `core/adversarial.py` vets proposed architectures for spectral bias, capacity bottlenecks, and MLX efficiency.
- **The Gate**: All auto-generated suggestions in `experiments.yaml` now include an embedded `critique` field.
- **Your Task**: If you see `[AGENT] Manual Review Required`, you must review the rationale and hyperparameters to ensure they align with past scientific diagnoses.

---

## The Research Loop (End-to-End)

1.  **Analyze**: `uv run analyze.py --papers` to find high-gap benchmarks.
2.  **Suggest**: `uv run auto_suggest.py --generate --write-yaml`
    - *Agent Action*: Fulfill any `[AGENT]` review requests in the output.
3.  **Refine**: Edit `experiments.yaml` based on critiques.
4.  **Execute**: `uv run autorun.py --priority 1 --commit`
    - *Auto-Rescue*: If a run fails with NaNs, the system performs a `/debug` probe.
    - *Agent Action*: Fulfill any `[AGENT]` scientific diagnosis requests in the logs.
5.  **Distill**: Update `SKILL.md` with hard-won lessons from the `trajectories.jsonl` log.

---

## Finding Current State

| What | Where |
|------|-------|
| Best results + SOTA gaps | `uv run analyze.py --papers` |
| Pending queue + Critiques | `experiments.yaml` |
| Scientific Debug Logs | `logs/probes/probe_<exp>.jsonl` |
| RL Replay Buffer | `logs/trajectories.jsonl` |
| Paper Registry | `docs/papers/*.yaml` |
| Plateau analysis | `python3 -m core.hypothesis --benchmark <bm>` |

---

## Hard Constraints (v2.0)

| Rule | Reason |
|------|--------|
| Fulfill `[AGENT_REQUEST]` immediately | Prevents the loop from stalling on numerical failures. |
| Log diagnoses to `trajectories.jsonl` | Builds "memory" for future adversarial reviews. |
| Model names match registry keys | `FNO2D` (exact), not `FNO2d`. |
| 2D Memory Ceiling | `hidden_dim ≤ 32`, `n_layers ≤ 4`, `n_modes ≤ 12`. |
| No PINO | Consistently diverges. |

---

## Reference Commands

```bash
# Manual Debug Trigger
PYTHONPATH=. python3 core/scientific_debugger.py <exp_name> <benchmark>

# Manual Adversarial Review
PYTHONPATH=. python3 core/adversarial.py <model_file> <rationale> <benchmark>

# Gap Analysis Report
cat docs/GAP_ANALYSIS.md
```
