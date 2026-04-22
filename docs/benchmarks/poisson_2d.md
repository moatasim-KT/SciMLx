# 2D Poisson Equation

**ID**: `poisson_2d`
**Status**: [STABLE]
**Task ID**: `T4.3`

## Problem Formulation
The fundamental elliptic partial differential equation on $[0, 1]^2$, testing a model's ability to learn the mapping from a source field $f(x, y)$ to the potential $u(x, y)$:

$$ -\nabla^2 u(x, y) = f(x, y) $$

where:
- $f(x, y)$ is a zero-mean source term.
- $u(x, y)$ is the resulting potential field.

## Setup
- **Domain**: $[0, 1]^2$ with periodic boundary conditions.
- **Grid Size**: $64 \times 64$.
- **Numerical Solver**: Exact spectral solver in Fourier space (machine precision).

## Initial Conditions
Initialized with a zero-mean random source field $f(x, y)$ generated from a truncated Fourier series ($k=5$, scale=1.0). Zero-mean condition is enforced to ensure a solution exists on the periodic domain.

## Boundary Conditions
Periodic in $x$ and $y$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A professional scientific visualization of the 2D Poisson equation solution. The plot shows the relationship between a multi-modal source field 'f' (with sharp peaks and valleys) and the smooth, diffusive potential field 'u'. The 'inferno' colormap is used to highlight the intensity of the source peaks, while a 'viridis' colormap displays the resulting potential. Mathematical clarity, high precision, clean background, professional publication style."
  viz_type: "source_potential_comparison"
  channels: ["source_f", "potential_u"]
  color_map: "inferno_viridis"
  layout: "side_by_side"
```
