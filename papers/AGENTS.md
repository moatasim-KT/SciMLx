# PAPERS DIRECTORY

**Purpose:** Paper registry with SOTA results and suggested experiments

## STRUCTURE
```
papers/
├── fno_2020.yaml          # Fourier Neural Operator
├── rfno_2024.yaml        # Residual FNO (our best on KdV)
├── ffno_2023.yaml        # Factorized FNO
├── afno_2022.yaml        # Adaptive FNO (broken here)
├── gnot_2023.yaml       # Graph Neural Operator Transformer
├── deeponet_2021.yaml   # DeepONet
├── pino_2021.yaml       # Physics-Informed Neural Operator (broken)
├── hnn_2019.yaml        # Hamiltonian Neural Networks
├── augmentation_2023.yaml  # Key for Burgers (+aug → 0.1468)
├── h1_loss.yaml         # H1 loss (barely helps)
├── curriculum_2009.yaml  # Curriculum learning
├── ensemble_uq_2023.yaml # Ensemble + uncertainty
├── mppde_2022.yaml       # Multipole PDE
├── inverse_pinn_2023.yaml # Inverse problems
├── neural_ode_ude_2020.yaml # Neural ODEs + UDEs
├── physicsnemo_2024.yaml  # PhysicsNEMO
└── .claude/             # Claude settings
```

## WHERE TO LOOK
| Task | Command |
|------|---------|
| List all papers | `uv run paper_registry.py` |
| SOTA gaps | `uv run paper_registry.py --gaps` |
| Pending ideas | `uv run paper_registry.py --pending` |
| Auto-suggest | `uv run auto_suggest.py` |

## KEY INSIGHTS
- **augmentation_2023.yaml** — single biggest win on Burgers
- **rfno_2024.yaml** — explains why RFNO beats FNO on KdV
- **h1_loss.yaml** — barely helps (0.1559 vs 0.1553)
- 20 papers with SOTA targets + suggested experiments, many pending

## CONVENTIONS
- Status: pending/implemented
- Each has: title, key idea, reported results, our results, suggested experiments
- auto_suggest.py reads results.json to compute dynamic KNOWN_WINS