# 2D Advection-Diffusion-Reaction (PDEBench)

**ID**: `pdebench_2d`
**Status**: [STABLE]
**Task ID**: `T5.1`

## Problem Formulation
A comprehensive 2D benchmark mapping to the Advection-Diffusion-Reaction family, capturing multi-scale interactions and non-linear dynamics on $[0, 2\pi)^2$:

$$ \frac{\partial u}{\partial t} = D \nabla^2 u - \mathbf{c} \cdot \nabla u + R u(1 - u) $$

where:
- $D = 0.02$ is the diffusion coefficient.
- $\mathbf{c} = (0.5, 0.5)$ is the advection velocity vector.
- $R = 0.1$ is the reaction rate.

## Setup
- **Domain**: $[0, 2\pi)^2$ with periodic boundary conditions.
- **Grid Size**: $64 \times 64$.
- **Temporal Horizon**: $T = 1.0$.
- **Time Steps**: $500$ ($dt = 0.002$).
- **Numerical Solver**: Fourier pseudo-spectral method with explicit Euler time-stepping.

## Initial Conditions
Smooth random field with low-mode Fourier components ($k=4$) and a small positive offset to initiate the reaction-diffusion process.

## Boundary Conditions
Periodic in $x$ and $y$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A scientific visualization of a 2D reaction-diffusion system. The image shows a field 'u' evolving into organic, spot-like or labyrinthine patterns as the advection-diffusion-reaction process unfolds. A 'YlGnBu' colormap represents the concentration levels. The visualization captures the balance between the smoothing effect of diffusion and the pattern-forming nature of the non-linear reaction term. Professional laboratory aesthetic, high detail, sharp transitions."
  viz_type: "concentration_map_2d"
  channels: ["concentration"]
  color_map: "YlGnBu"
  layout: "evolution_snapshot"
```
