# 2D Allen-Cahn Phase Field

**ID**: `allen_cahn_2d`
**Status**: [STABLE]
**Task ID**: `T4.1`

## Problem Formulation
The Allen-Cahn equation is a reaction-diffusion PDE that models the process of phase separation (spinodal decomposition) and interface coarsening in multi-component systems on $[0, 1]^2$:

$$ \frac{\partial \phi}{\partial t} = \epsilon^2 \nabla^2 \phi + \phi - \phi^3 $$

where:
- $\phi(x, y, t) \in [-1, 1]$ is the phase field order parameter.
- $\epsilon = 0.05$ is the interface width parameter.
- The cubic term $\phi - \phi^3$ drives the system toward stable phases $\phi \approx \pm 1$.

## Setup
- **Domain**: $[0, 1]^2$ with periodic boundary conditions.
- **Grid Size**: $64 \times 64$.
- **Temporal Horizon**: $T = 1.0$.
- **Time Steps**: $200$ ($dt = 0.005$).
- **Numerical Solver**: 2nd-order Exponential Time Differencing Runge-Kutta (ETDRK2) spectral solver, ensuring stability for the stiff diffusion term.

## Initial Conditions
Generated from a random Gaussian field mapped through a `tanh` function with a scale factor of $1/(\sqrt{2}\epsilon)$. This creates physically realistic initial data with resolved random droplets and domains of the two phases.

## Boundary Conditions
Periodic in $x$ and $y$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A scientific visualization of phase separation in the 2D Allen-Cahn equation. The image displays a marbled pattern of two distinct phases (represented by deep blue and bright white) with sharp but smooth boundaries between them. The coarsening process is evident as small droplets merge into larger domains. High contrast, clean minimalist aesthetic, resembling a high-resolution microscopic image of a polymer blend or metallic alloy. Precise mathematical boundaries, 8k resolution."
  viz_type: "phase_field_2d"
  channels: ["phase"]
  color_map: "RdBu"
  layout: "target_full"
```
