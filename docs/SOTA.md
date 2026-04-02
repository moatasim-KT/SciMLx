# SOTA Results & Benchmarks

Performance targets are relative L2 error (`val_l2_rel`, lower is better).
All numbers from the FNO / U-NO / WNO papers on their respective test sets.

---

## 1D Burgers Equation (ν = 0.01/π, N=64, T=1)

| Model           | Relative L2 | Notes                                     |
|-----------------|-------------|-------------------------------------------|
| FNO (paper)     | 0.0149      | Li et al. 2020, 16 modes, width 64        |
| UNO             | ~0.012      | Rahman et al. 2022, ~20% over FNO         |
| WNO             | ~0.015      | Tripura et al. 2022                       |
| DeepONet        | 0.05–0.10   | Lu et al. 2021                            |
| PINN            | ~0.10–0.30  | Highly variable; stiff at low ν           |
| **This repo (best)** | **0.2307** | FNO, hidden=64 (as of latest run)    |

**Gap to SOTA:** ~15× on Burgers.  Key levers: more modes, wider model,
PINO loss, UNO architecture.

---

## 2D Darcy Flow (steady-state, N=64×64)

| Model        | Relative L2 | Notes                                |
|--------------|-------------|--------------------------------------|
| FNO (paper)  | 0.0108      | Li et al. 2020, 12 modes, width 32   |
| WNO          | ~0.015      | Better on heterogeneous permeability |
| DeepONet     | ~0.02       | Lu et al. 2021                       |
| U-Net        | ~0.02–0.05  | Standard conv U-Net                  |
| **This repo** | **0.9986** | FNO baseline — solver approximation  |

**Gap to SOTA:** ~90× — the `prepare.py` Darcy solver uses averaged permeability
(not heterogeneous), so the problem is easier in principle but the high baseline
error suggests numerical issues in training.  Focus on Burgers first.

---

## 2D Navier-Stokes (vorticity, ν=1e-3, T=10, N=64×64)

| Model        | Relative L2 | Notes                                |
|--------------|-------------|--------------------------------------|
| FNO (paper)  | 0.0128      | Li et al. 2020                       |
| AFNO         | ~0.008      | Guibas et al. 2022                   |
| U-Net        | ~0.02       |                                      |
| **This repo** | **crash**  | Diverges at step 67 — unstable solver|

**Status:** NS solver in `prepare.py` is numerically unstable for these ICs.
Avoid until the solver is fixed.  Contributed experiments on NS are welcome
(modify the solver strategy in `prepare.py` if permitted by maintainer).

---

## Target Milestones (Burgers 1D)

| Milestone                    | val_l2_rel | Likely approach                  |
|------------------------------|-----------|----------------------------------|
| Current best                 | 0.2307    | FNO hidden=64 (12×64)            |
| Short-term target            | 0.10      | UNO or wider FNO + better LR     |
| Medium-term target           | 0.05      | PINO + architecture search       |
| Paper-quality SOTA           | 0.015     | FNO 16 modes, 64 width, 500 ep   |
| Best published               | ~0.012    | UNO, WNO, or AFNO                |

---

## Notes on Comparison

- Paper results train for 500+ epochs; this repo uses a 5-minute budget
  (~23–50 epochs depending on model size and hardware speed).
- Exact comparison requires the same validation set, grid size, and ν.
  `prepare.py` fixes all of these (seed=42 validation, N=64, ν=0.01/π).
- VRAM: 1D models use ~30–150 MB; 2D models use ~0.5–3 GB.
