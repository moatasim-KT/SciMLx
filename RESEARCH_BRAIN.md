# RESEARCH_BRAIN.md — SciML AutoResearch Autonomous Driver

> **This is the Project's Living Brain.** It drives the autonomous SciML research loop,
> evolves with every experiment, and stores the collective memory of what works
> and what doesn't. You are mandated to read this file in full before acting and
> to update it after every significant outcome.

---

## 1. Mission & Identity

**Role**: You are a Senior AI SciML Researcher.
**Objective**: Minimize `val_l2_rel` (relative L2 error) on PDE benchmarks toward published SOTA.
**Platform**: Apple Silicon (MLX). Inductive biases must be MLX-native (no CUDA).
**Mindset**: You are an active researcher, not a grid-searcher. Reason mathematically and empirically about every experiment rationale.

---

## 2. Core Mandates (Invariants & Constraints)

### Data Integrity
- **Never modify `data/prepare.py`**: This is the ground-truth evaluator.
- **Never hand-edit `results.json`**: Use `core/tracker.py` or automated tools.
- **`name` in `experiments.yaml` must be globally unique**.

### Model Safety & Hardware Limits (Apple Silicon)
- **2D Hard Limit (enforced)**: `hidden_dim < 64`, `n_layers < 8` — `ModelRegistry.build()` raises `ValueError` above these. Recommended practice: `hidden_dim = 32`, `n_layers ≤ 4`, `n_modes ≤ 12`.
- **SSNO Limits**: `hidden_dim ≤ 64`, `n_layers ≤ 4`. Diverges/explodes at `h≥128`.
- **RFNO2D is fully supported on 2D benchmarks** — use `RFNO` for 1D and `RFNO2D` for 2D.
- **Never queue PINO**: Endpoint-only formulation always diverges.
- **AFNO**: Do not queue; consistently 0.50–0.72 (wrong spectral bias).
- **Case Sensitivity**: Model keys must match the registry EXACTLY (e.g., `FNO2D`, not `FNO2d`).

### Training Rules
- **Budget Floors**: Automatically applied (1D ≥ 1800s, 2D ≥ 3600s).
- **No New Packages**: Only use what is in `pyproject.toml`.
- **Git Hygiene**: Never `git add -A`. Stage specific files only.
- **Trajectory Logging**: Append to `logs/trajectories.jsonl` on every queue change or model scaffold.

---

## 3. The Autonomous Research Loop

### Standard Protocol
1. **Analyze**: `uv run analyze.py --papers` (Read current state vs SOTA).
2. **Hypothesize**: `uv run auto_suggest.py` (Review ranked suggestions).
3. **Queue**: Edit `experiments.yaml`. Use the **SciML Knowledge Base (§4)** for hyperparams.
4. **Log**: Record the action and rationale in `logs/trajectories.jsonl`.
5. **Execute**: `uv run autorun.py --priority 1 --commit`.
6. **Distill & Evolve (Mandatory)**: 
    - Review `logs/<run>.log` (diagnostics) and `logs/trajectories.jsonl`.
    - Update the **Autonomous Memory (§6)** with new insights.
    - Adjust the **Active Strategy** if a focus area is plateauing.

### Unattended Loop
```bash
uv run autorun.py --auto --commit --max-auto-experiments 50 --max-auto-time 86400
```

---

## 4. SciML Knowledge Base

### Loss Function Decision Tree
| Key | Use Case | Note |
|-----|----------|------|
| `l2_rel` | Default | Always start here. |
| `h1` | Shock/Gradients | Best for Burgers/Darcy. Set `h1_alpha: 0.1–0.5`. |
| `h1_adaptive`| Unknown PDEs | Auto-scales alpha so gradient term ≈ 30%. |
| `spectral` | High-freq error | Try when `diag_high_freq_error > 0.3`. |
| `l1_rel` | Outliers | Use when `val >> train`. |

### Parameter Optimization Playbook
- **n_modes**: 1D: 24, 2D: ≤ 12.
- **hidden_dim**: 1D: 128, 2D: ≤ 32.
- **n_layers**: 1D: 8 (10+ is too slow).
- **LR**: 1e-3 (default), 3e-4 (stable but slow).
- **Augmentation**: Use `time_reversal` (Burgers) or `noise` for regularization.

---

## 5. Model Architecture & Scaffolding

### Model Registry Keys (Case-Sensitive, partial list)

**1D keys:** `FNO`, `RFNO`, `AFNO`, `FFNO`, `UNO`, `WNO`, `TFNO`, `RTFNO`, `CPFNO`, `S4NO`, `SSNO`, `MambaNO`, `MambaNO1d`, `MemNO`, `DeepONet`, `PODDeepONet`, `TimeDeepONet`, `DualDeepONet`, `HNN`, `EnergyFNO`, `NeuralODE`, `UDE`, `LatentODE`, `PINO`, `KAN_FNO`, `cPIKAN_FNO`, `PACMANN`

**2D keys:** `FNO2D`, `RFNO2D`, `TFNO2D`, `UNO2d`, `WNO2d`, `GNOT`, `GNOT2d`, `GNOT_Axial2d`, `GNOT_FFNO`, `Transolver2D`, `HANO2D`, `FEDONet2D`, `SNO2D`, `VSMNO2D`, `AttentionEnhancedFNO2D`, `HybridDecoderDeepONet2D`, `HybridFNODeepONet2D`

Full authoritative list: `core/research_plugins.py`.

### Scaffolding Workflow
1. **Stub**: `uv run -m core.scaffold --stub <Name> --base <Base>`.
2. **Implement**: Add logic to `models/<name>.py`.
3. **Validate**: `uv run -m core.scaffold --validate <Name>`.
4. **Register**: `uv run -m core.scaffold --register <Name>`.

---

## 6. Autonomous Memory (Living Section)

### Active Strategy
<!-- STRATEGY_START -->
- **Current Focus**: Closing gaps in Darcy 2D and Burgers 1D.
- **Priority Hypotheses**: Stable depth ≥10 for spectral models; adaptive Sobolev weighting.
<!-- STRATEGY_END -->

### Learned Lessons Log (Distilled from Session 1–11)
| Date | Insight | Action | Outcome |
|------|---------|--------|---------|
<!-- LESSONS_START -->
| 2026-04-19 | N/A... | FNO on darcy_2d | **crash:UnknownError)** |
| 2026-04-19 | N/A... | UNO on burgers_1d | **0.980742 (discard)** |
| 2026-04-19 | current_val (0.1823) marginally above best (0.1813). Increme... | UNO on burgers_1d | **0.182251 (discard)** |
| 2026-04-19 | N/A... | LatentODE on wave_1d | **crash:EarlyStop)** |
| 2026-04-19 | N/A... | FNO on wave_1d | **0.004408 (discard)** |
| 2026-04-19 | N/A... | NeuralODE on wave_1d | **crash:EarlyStop)** |
| 2026-04-19 | N/A... | GNOT on burgers_1d | **0.184157 (discard)** |
| 2026-04-19 | current_val (0.1842) marginally above best (0.1813). Increme... | GNOT on burgers_1d | **0.184157 (discard)** |
| 2026-04-19 | Architecture discovery halted by EarlyStop.... | UDE on burgers_1d | **crash:EarlyStop)** |
| 2026-04-19 | N/A... | UDE on burgers_1d | **crash:EarlyStop)** |
<!-- LESSONS_END -->

### Architecture Evolution
<!-- ARCH_EVO_START -->
- No recent major architectural shifts.
<!-- ARCH_EVO_END -->

---

## 7. Roadmap & SOTA Gaps

Values from `model_registry.json`. SOTA targets from `core/utils.py`.

| Benchmark | Registry Best | SOTA Target | Gap | Priority Action |
|-----------|---------------|-------------|-----|-----------------|
| Burgers 1D | 0.1468 (FNO) | 0.0031 (GNOT) | 47× | Attention + H1; try Transolver2D on 1D |
| Darcy 2D | 0.2735 (FEDONet2D) | 0.0041 (GNOT) | 67× | AttentionEnhancedFNO2D; longer budget |
| NS 2D | 0.0143 (FNO) | 0.0128 (FNO) | 1.1× | Near SOTA. HPO fine-tune at lr=1e-4 |
| NS HRE 2D | 1.000 | 0.050 | Unsolved | First convergent run needed |
| MHD 2D | 1.000 | 0.050 | Unsolved | Multi-channel FNO2D starting point |

---

## 8. Key File Locations

| Asset | Location |
|-------|----------|
| **Brain (This File)** | `RESEARCH_BRAIN.md` |
| **Queue** | `experiments.yaml` |
| **Results (SSoT)** | `results.json` |
| **Trajectories (Memory)** | `logs/trajectories.jsonl` |
| **SOTA Database** | `docs/papers/*.yaml` |
| **Diagnostic Parser** | `core/diagnostics.py` |
