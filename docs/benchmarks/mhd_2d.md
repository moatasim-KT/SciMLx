# 2D Magnetohydrodynamics (MHD)

**ID**: `mhd_2d`
**Status**: [STABLE]
**Task ID**: `T3.3`

## Problem Formulation
A 2D incompressible magnetohydrodynamics system in vorticity-potential formulation, modeling the interaction between a conducting fluid and a magnetic field on $[0, 1]^2$:

$$ \frac{\partial \omega}{\partial t} + (\mathbf{u} \cdot \nabla) \omega = \nu \nabla^2 \omega + (\mathbf{B} \cdot \nabla) j $$
$$ \frac{\partial A}{\partial t} + (\mathbf{u} \cdot \nabla) A = \eta \nabla^2 A $$

where:
- $\omega$ is the fluid vorticity.
- $A$ is the magnetic vector potential ($B = \nabla \times A$).
- $j = \nabla^2 A$ is the electric current density.
- $\nu = 10^{-3}, \eta = 10^{-3}$ are the kinematic viscosity and magnetic diffusivity.

## Setup
- **Domain**: $[0, 1]^2$ with periodic boundary conditions.
- **Grid Size**: $64 \times 64$.
- **Temporal Horizon**: $T = 0.5$.
- **Time Steps**: $500$ ($dt = 0.001$).
- **Numerical Solver**: Dual-field semi-implicit spectral solver.

## Initial Conditions
Random fields inspired by the Orszag-Tang vortex benchmark, using smooth low-mode Fourier series ($k=4$, scale=0.1) for both vorticity and magnetic potential.

## Boundary Conditions
Periodic in $x$ and $y$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A complex multi-channel scientific visualization of 2D magnetohydrodynamics (MHD). The image features two overlaid or side-by-side plots: one showing the fluid vorticity with swirling eddies, and the other showing magnetic field lines and current density filaments. High-energy plasma aesthetics with electric blues for the magnetic potential and fiery oranges for the fluid intensity. Intricate coupling between the fields is visible. Professional scientific simulation style, high resolution, cinematic detail."
  viz_type: "multi_field_overlay"
  channels: ["vorticity", "magnetic_potential"]
  color_map: "inferno_ice"
  layout: "coupled_fields"
```
