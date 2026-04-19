# Gap Analysis: HypothesisEngine vs. Scientific Debugger (/debug)

## Current State: HypothesisEngine
The `HypothesisEngine` is a **macro-level reasoning engine**. It looks at the global history of experiments to identify long-term trends and structural failures.

- **Data Sources**: `results.json` (aggregated outcomes) and `trajectories.jsonl` (RL-style sequence of actions).
- **Failure Modes**: `spectral_bias`, `capacity_limited`, `step_limited`, `wrong_inductive_bias`.
- **Interventions**: Architectural swaps (e.g., FNO → RFNO), loss function changes (e.g., L2 → H1), or budget adjustments.
- **Limitation**: It treats numerical crashes (NaN/Inf) as a black box (`gradient_collapse`). It doesn't know *why* or *where* the model exploded.

## Target State: Scientific Debugger (/debug)
The `ScientificDebugger` is a **micro-level diagnostic probe**. it is triggered by numerical instability to pinpoint exact failure points.

- **Data Sources**: High-fidelity probe logs containing layer-wise gradients and activation statistics.
- **Failure Modes**: Exploding gradients, vanishing residuals, weight initialization bias.
- **Interventions**: Precise code-level modifications (e.g., "Add LayerNorm to Layer 3", "Reduce initialization scale of the lifting layer").
- **Status**: Core logic implemented in `core/scientific_debugger.py`.

## Identified Gaps

### 1. Telemetry Gap
- **Problem**: The standard `train.py` loop does not log layer-wise statistics because it is computationally expensive (requires `mx.eval` and synchronizations).
- **Solution**: The `ScientificDebugger` uses a `--probe` flag in `train.py` to enable this high-fidelity logging only during a short (5-step) re-run of the failed configuration.

### 2. Integration Gap (T6_LOOP)
- **Problem**: `autorun.py` detects a crash and calls the debugger, but the debugger's output is currently just printed to the console. It is not "ingested" by the autonomous loop to automatically apply the fix.
- **Solution**: `autorun.py` should be upgraded to use the `ScientificDebugger` output to generate a "fix-branch" experiment (e.g., `exp_name_fix`) with the proposed modifications.

### 3. Adversarial Feedback Loop
- **Problem**: `AdversarialReviewer` (T3) critiques new architectures before they run, but it doesn't "know" about common failure modes identified by the `ScientificDebugger`.
- **Solution**: The `AdversarialReviewer` prompt should include recent scientific diagnoses to help it veto architectures that repeat past numerical mistakes.

## Next Steps
1. **T6_LOOP**: Complete the integration in `autorun.py` to ensure `ScientificDebugger` diagnoses are recorded in `trajectories.jsonl` and used for subsequent HPO/suggestions.
2. **T4_SCAF**: Ensure `scaffold.py --reason` is the standard path for any LLM-generated model code.
