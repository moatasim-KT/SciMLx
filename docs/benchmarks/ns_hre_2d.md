# 2D Navier-Stokes (High Reynolds Number)

**ID**: `ns_hre_2d`
**Status**: [FIXED (EXT)]
**Task ID**: `T3.1`

## Problem Formulation
A high-Reynolds number challenge for the 2D Navier-Stokes equations, featuring a 10× lower viscosity and 2× longer temporal horizon compared to the standard `ns_2d` benchmark:

$$ \frac{\partial \omega}{\partial t} + (\mathbf{u} \cdot \nabla) \omega = \nu \nabla^2 \omega $$

where:
- $\nu = 10^{-3}$ ($Re \approx 1000$).

## Setup
- **Domain**: $[0, 2\pi)^2$ with periodic boundary conditions.
- **Grid Size**: $64 \times 64$.
- **Temporal Horizon**: $T = 2.0$.
- **Time Steps**: $1000$ ($dt = 0.002$).
- **Numerical Solver**: 4th-order Exponential Time Differencing Runge-Kutta (ETDRK4) with 2/3-rule dealiasing.

## Initial Conditions
Smooth random vorticity field with amplitude scale $\alpha = 0.05$.

## Boundary Conditions
Periodic in $x$ and $y$.

## Provenance & Fixes
This benchmark is designed to replicate the Re=1000 challenge from Li et al. (2020). It uses the ETDRK4 scheme to handle the increased stiffness of the high-Reynolds regime while maintaining high temporal accuracy.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A high-fidelity scientific visualization of 2D Navier-Stokes turbulence at Re=1000. The image showcases a complex turbulent cascade with a vast array of multi-scale vortical structures, filamentary stretching, and intricate eddy interactions. A vibrant 'magma' color map highlights the intensity of the vorticity field against a dark, minimalist background. Exceptional detail, sharp gradients, cinematic scientific aesthetic, 8k resolution, suitable for a Nature Physics cover."
  viz_type: "vorticity_field_2d"
  channels: ["vorticity"]
  color_map: "magma"
  layout: "target_full_resolution"
```
