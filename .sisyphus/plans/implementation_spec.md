# Implementation Specification for Autoresearch-mlx Enhancements

## Executive Summary
This document outlines a phased implementation plan to enhance the **autoresearch-mlx** codebase, focusing on:  
- **Autonomous hypothesis generation**,  
- **Failure diagnosis**,  
- **Cross-benchmark knowledge transfer**,  
- **Enhanced dashboard analytics**, and  
- **User feedback integration**.  

The goal is to create a self-improving system that identifies novel architectures, learns from failures, and transfers knowledge across benchmarks while keeping experiments reproducible and trackable via a collaborative dashboard.

---

## Implementation Phases

### Phase 1: Prototype Novelty Engine & Cross-Benchmark Graph
**Objective**: Enable the system to propose novel experiments and share insights between benchmarks.  
**Timeline**: 2–4 weeks  

#### Components
1. **Novelty Engine**
   - **Novelty Scoring**: Use `k-NN` similarity metrics on config vectors to identify unexplored configurations.
   - **Hollow-Pipe Discovery**: Generate "skeletal" architecture templates (e.g., `FNO + Transformer` hybrids) using `model_scaffold.py`.
   - **Edge-Case Exploration**: Propose extreme hyperparameters (e.g., `n_modes=64` for 1D benchmarks).

2. **Cross-Benchmark Knowledge Graph**
   - **Graph Structure**: `Benchmark → Model → Accuracy` relationships (Neo4j or in-memory).
   - **Transfer Learning API**: Propose winning hyperparameters from stronger benchmarks (e.g., `UNO` from `Wave 1D` to `Burgers 1D`).

#### Deliverables
- Prototype `novelty_engine.py` with 10 generated experiments.
- Knowledge graph populated with 20 benchmark-model connections.
- First cross-benchmark hypothesis (e.g., "UNO from Wave 1D applied to Burgers 1D").

---

### Phase 2: Failure Diagnosis & Adaptive HPO
**Objective**: Automatically diagnose failures and improve HPO efficiency.  
**Timeline**: 3–5 weeks  

#### Components
1. **Failure Diagnoses Module**
   - **Log Parser**: Extract crash patterns (OOM, NaN, timeout) + hyperparameters.
   - **Classifier**: SVM or ML model trained on labeled failure cases to rank root causes.
   - **Cause-to-Fix Map**: Default fixes (e.g., `batch_size // 2` for OOM + `lr // 10` for NaN).

2. **Adaptive HPO**
   - **Dynamic Kernel Width**: Adjust Bayesian HPO kernel width based on feature dimensionality.
   - **Failure-Aware Search**: Prioritize parameter ranges less likely to fail (e.g., `n_modes > 16` for known stable models).

#### Deliverables
- `diagnostics.py` parses 100+ failure logs.
- HPO module achieves 30% faster convergence on failure-prone configs.

---

### Phase 3: Enhanced Dashboard & Auto-Suggest Integration
**Objective**: Enrich visualization with architectural insights and real-time tuning.  
**Timeline**: 3 weeks  

#### Components
1. **Spectral Bias Visualizations**
   - Add `Plotly.js` heatmaps for spectral error vs. mode/frequency.
   - Example: [Spectral Bias Heatmap Screenshot](mockup/spectral_bias_heatmap.png)

2. **Live Config Tuner**
   - UI sliders for `lr`, `n_modes`, `n_layers` → dynamically requeue experiments.

3. **Interactive Comparison Tool**
   - Compare two experiments (configs, metrics, training curves) side-by-side.

#### Deliverables
- Dashboard UI updated with heatmaps and tuner sliders.
- Comparison tool shows `FNO_burgers_1d` vs. `UNO_burgers_1d` differences.

---

### Phase 4: Post-Run Optimizer & User Feedback
**Objective**: Use successful runs to explore deeper parameter spaces and user input.  
**Timeline**: 2 weeks  

#### Components
1. **Post-Run Optimizer**
   - Graduate successes to new experiments with:  
     - 50% budget for `h/m` sweeps.  
     - Borderline hyperparameters (e.g., `n_modes=16 → 24`).  

2. **User Feedback Loop**
   - UI form for users to flag gaps (e.g., "Add attention mechanisms to 2D models").
   - Prioritize hypotheses matching user goals (e.g., "Energy conservation" for specific benchmarks).

#### Deliverables
- Optimizer graduates 5 experiments from successful runs.
- User suggestions integrated in 3 weeks (e.g., 2 new experiments/month).

---

## Technical Components

### A. Novelty Engine (`novelty_engine.py`)
```python
def generate_novel_hypothesis(existing_configs):
    # Convert existing configs to vector embeddings
    for config in existing_configs:
        embedding = vectorize_config(config)
    # Find gaps in embedding space
    gap_configs = gap_detection(embedding, distance_threshold=0.4)
    # Generate candidates
    candidates = propose_candidates(gap_configs)
    return filter_by_domain_knowledge(candidates)
```

### B. Knowledge Graph (Neo4j Snippet)
```cypher
CREATE (:Benchmark {name: "Burgers 1D", type: "1D", sd: "0.1468"}) 
CREATE (:Model {name: "FNO", h: 128, m: 24}) 
CREATE (:Benchmark)-[:HAS_MODEL]->(:Model) 

CREATE (:Improvement {target:"SWE 2D", adoptee:"UNO_SWE_2D", improvement: 0.005})
FOR A IN improvement WHERE A.source_benchmark ~ "1D" 
FOR B IN improvement WHERE B.target_benchmark ~ "SWE" 
MERGE A -[:WORTH_TRYING]-> B
```

### C. Dashboard Comparison Tool (React Snippet)
```javascript
function updateComparison(configs) {
  const performanceMetrics = configs.map(c => c.metrics).reduce((acc, curr) => 
    Object.entries(curr).reduce((a, [k, v]) => ({...a, [k]: [...(a[k] || []), v]}), {}), 
    {}
  );

  const delta = Object.fromEntries(
    Object.entries(performanceMetrics).map(([k, values]) => [
      k,
      {
        min: Math.min(...values),
        max: Math.max(...values),
        avg: mean(values),
      }
    ])
  );
}
```

---

## Deliverables
1. Code:  
   - `novelty_engine.py`, `knowledge_graph.db`, `dashboard_tuner.js`.  
   - Failure diagnosis logs + classifier model.  
2. Benchmarks:  
   - 50% increase in novel hypothesis quality.  
   - 30% faster HPO convergence on failure-prone configs.  
3. Documentation:  
   - `AUTURES_RESEARCH.md` (implementation plan).  
   - Dashboard user guide.  

---

## Dependencies
- **Tools**: Neo4j, BayesianHPO (from `bayesian_hpo.py`), Plotly.js.  
- **External Systems**: GitHub API for autosuggest, Huidenal analytics.  
- **Infrastructure**: Redis for expiring hypothesis braking.  

---

## Contributors
- **Core Team**: Authors of this document.  
- **Stakeholders**: Machine Learning researchers for hypothesis validation, SWE 2D experts for 2D constraint tuning.  

---

## Risks & Mitigation
| Risk | Mitigation |  
|------|-----------|  
| Overfitting to existing data | Penalize novelty score for proxy metrics (e.g., KL divergence from successful configs). |  
| Dashboard latency | Optimize graph queries with pre-filters (e.g., `n_modes ≤ 24`). |  
| High HPO cost | Share BHO kernels across similar experiments (approx 40% savings). |  

---

## Conclusion
This plan transforms autoresearch-mlx into a **self-improving research platform** capable of:  
1. Generating novel architectures autonomously,  
2. Learning from failures and successes,  
3. Applying cross-benchmark insights, and  
4. Collaborating with users through live tuning and feedback.  

Each phase builds on previous work, ensuring incremental progress while maintaining codebase stability.  

--- 

Would you like to add specific code examples, mockups for the dashboard, or further details on any component?