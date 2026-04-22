# 1D Viscous Burgers Equation

**ID**: `burgers_1d`
**Status**: [STABLE]
**Task ID**: `T2.1`

## Problem Formulation
The viscous Burgers' equation is a fundamental partial differential equation in fluid mechanics, representing a simplified model for shock wave formation and propagation on a 1D periodic domain $[0, 2\pi)$:

$$ \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} $$

where:
- $u(x, t)$ is the scalar field (e.g., velocity).
- $\nu = \frac{0.01}{\pi} \approx 0.003183$ is the kinematic viscosity.

## Setup
- **Domain**: $[0, 2\pi)$ with periodic boundary conditions.
- **Grid Size**: $64$.
- **Temporal Horizon**: $T = 1.0$.
- **Time Steps**: $500$.
- **Numerical Solver**: Batch pseudo-spectral IMEX-Euler (implicit diffusion, explicit advection) with 2/3-rule dealiasing.

## Initial Conditions
Generated from a smooth random Fourier series on $[0, 2\pi)$ with coefficient amplitude decaying as $k^{-1.5}$ for $C^1$ smoothness. Spectral modes are truncated at $k=10$.

## Boundary Conditions
Periodic boundary conditions on the interval $[0, 2\pi)$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A 1D temporal evolution heatmap of the viscous Burgers equation on a periodic domain. The plot shows a smooth initial wave profile at the top (t=0) that gradually steepens as it travels downwards (t=T), forming a distinct shock front with a sharp gradient. The colormap is high-contrast, representing positive velocity in warm oranges and negative velocity in cool blues. The aesthetic is clean and mathematical, resembling a matplotlib plot from a high-quality physics paper. Sharp gradients, smooth transitions, scientific accuracy."
  viz_type: "heatmap_1d"
  channels: ["velocity"]
  color_map: "RdBu_r"
  layout: "temporal_waterfall"
```
