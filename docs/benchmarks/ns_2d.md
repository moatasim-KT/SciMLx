# 2D Navier-Stokes (Vorticity Form)

**ID**: `ns_2d`
**Status**: [FIXED (EXT)]
**Task ID**: `T3.1`

## Problem Formulation
The incompressible Navier-Stokes equations in 2D vorticity-streamfunction formulation on a periodic domain $[0, 2\pi)^2$:

$$ \frac{\partial \omega}{\partial t} + (\mathbf{u} \cdot \nabla) \omega = \nu \nabla^2 \omega $$

where:
- $\omega = \nabla \times \mathbf{u}$ is the scalar vorticity.
- $\nu = 10^{-2}$ is the kinematic viscosity.
- $\mathbf{u} = (\frac{\partial \psi}{\partial y}, -\frac{\partial \psi}{\partial x})$ with $\nabla^2 \psi = \omega$.

## Setup
- **Domain**: $[0, 2\pi)^2$ with periodic boundary conditions.
- **Grid Size**: $64 \times 64$.
- **Temporal Horizon**: $T = 1.0$.
- **Time Steps**: $1000$ ($dt = 0.001$).
- **Numerical Solver**: Semi-implicit Euler spectral solver with 2/3-rule dealiasing.

## Initial Conditions
Smooth random vorticity field with amplitude scale $\alpha = 0.1$. This reduced scale ensures numerical stability ($CFL < 1.0$) for the chosen time step.

## Boundary Conditions
Periodic in $x$ and $y$.

## Provenance & Fixes
The legacy version in `prepare.py` utilized an initial condition scale of $1.0$, which led to velocities near $95$ and a $CFL \approx 61$, causing immediate numerical divergence. This "EXT" version fixes the IC scale to $0.1$, achieving a stable $CFL \approx 0.6$ and enabling convergence for operator learning.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A 2D scientific contour plot of fluid vorticity in a periodic Navier-Stokes simulation at moderate Reynolds number. The image displays a balanced distribution of rotating vortices and shear layers. The 'viridis' colormap illustrates the vorticity magnitude, with smooth transitions and clearly defined vortex cores. Professional scientific aesthetic, high resolution, sharp contours, neutral background, mathematically precise."
  viz_type: "contour_2d"
  channels: ["vorticity"]
  color_map: "viridis"
  layout: "input_target_pair"
```
