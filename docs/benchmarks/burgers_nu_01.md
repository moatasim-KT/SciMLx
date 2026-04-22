# 1D Burgers Equation (Moderate Viscosity)

**ID**: `burgers_nu_01`
**Status**: [STABLE]
**Task ID**: `T2.1`

## Problem Formulation
A variant of the 1D viscous Burgers' equation with moderate viscosity, representing a regime where diffusion is stronger than in the standard benchmark:

$$ \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} $$

where:
- $\nu = 0.1$.

## Setup
- **Domain**: $[0, 2\pi)$ with periodic boundary conditions.
- **Grid Size**: $64$.
- **Temporal Horizon**: $T = 1.0$.
- **Time Steps**: $500$.
- **Numerical Solver**: Batch pseudo-spectral IMEX-Euler with 2/3-rule dealiasing.

## Initial Conditions
Smooth random Fourier series with spectral modes truncated at $k=10$.

## Boundary Conditions
Periodic on $[0, 2\pi)$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A scientific 1D heatmap showing the evolution of the Burgers equation with moderate viscosity. The plot displays a smooth wave that gradually decays and broadens over time, illustrating the dominant effect of diffusion. The colors are muted and professional, using a 'Viridis' colormap. Clean lines, mathematical accuracy, high-resolution rendering suitable for a textbook illustration."
  viz_type: "heatmap_1d"
  channels: ["velocity"]
  color_map: "viridis"
  layout: "temporal_waterfall"
```
