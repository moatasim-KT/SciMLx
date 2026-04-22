# 2D Darcy Flow (Variable Coefficient)

**ID**: `darcy_2d`
**Status**: [FIXED (EXT)]
**Task ID**: `T3.2`

## Problem Formulation
The 2D Darcy flow equation is a steady-state elliptic PDE describing the pressure distribution $u(x, y)$ in a porous medium with a heterogeneous permeability field $a(x, y)$ on a periodic domain $[0, 1]^2$:

$$ -\nabla \cdot (a(x, y) \nabla u(x, y)) = f(x, y) $$

where:
- $a(x, y) > 0$ is the spatially varying permeability.
- $f(x, y)$ is a fixed source term.
- $u(x, y)$ is the pressure head.

## Setup
- **Domain**: $[0, 1]^2$ with periodic boundary conditions.
- **Grid Size**: $64 \times 64$.
- **Temporal Horizon**: N/A (Steady state).
- **Numerical Solver**: Preconditioned Conjugate Gradient (PCG) using a spectral Poisson preconditioner ($40$ iterations) to resolve the heterogeneous coupling accurately.

## Initial Conditions
- **Permeability $a$**: Log-normal random field $\exp(z)$, where $z$ is a GRF with zero mean and scale $0.5$ ($k=5$).
- **Source $f$**: Fixed, deterministic source term with zero mean to ensure solvability on a periodic domain.

## Boundary Conditions
Periodic in $x$ and $y$.

## Provenance & Fixes
The legacy version in `prepare.py` was structurally broken: it used only the mean of $a$ (losing all spatial heterogeneity) and the source $f$ was uncoupled from the input. This "EXT" version fixes the variable-coefficient operator application and ensures the source term is properly resolved, making it a valid benchmark for operator learning as described in Li et al. (2020).

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A professional scientific visualization comparing a 2D heterogeneous permeability field with the resulting pressure head in a Darcy flow simulation. The left panel shows the permeability 'a' as a multi-modal log-normal field with sharp boundaries, using a 'copper' color map. The right panel shows the smooth, diffusive pressure response 'u', using a 'viridis' color map. The visualization highlights how the fluid path is governed by the underlying permeability structure. Clean layout, high contrast, academic plotting style."
  viz_type: "dual_field_comparison"
  channels: ["permeability", "pressure"]
  color_map: "copper_viridis"
  layout: "side_by_side"
```
