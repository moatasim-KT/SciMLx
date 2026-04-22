# 2D Coupled Multiphysics

**ID**: `multiphysics_2d`
**Status**: [STABLE]
**Task ID**: `T4.3`

## Problem Formulation
A generic multi-physics benchmark simulating coupled cross-field dependencies, such as thermal-mechanical or chemical interactions, using two coupled heat-like equations on $[0, 2\pi)^2$:

$$ \frac{\partial u}{\partial t} = D_1 \nabla^2 u - \alpha v $$
$$ \frac{\partial v}{\partial t} = D_2 \nabla^2 v + \alpha u $$

where:
- $u, v$ are the coupled fields.
- $D_1 = 0.01, D_2 = 0.05$ are diffusion coefficients.
- $\alpha = 2.0$ is the coupling strength.

## Setup
- **Domain**: $[0, 2\pi)^2$ with periodic boundary conditions.
- **Grid Size**: $64 \times 64$.
- **Temporal Horizon**: $T = 1.0$.
- **Numerical Solver**: Analytic Fourier propagator (implemented via 10-step Euler in Fourier space for efficiency).

## Initial Conditions
Two independent smooth random fields generated from truncated Fourier series ($k=4$, scale=1.0).

## Boundary Conditions
Periodic in $x$ and $y$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A sophisticated multi-channel scientific visualization showing the coupling between two physical fields in a 2D domain. The image features a side-by-side or overlaid comparison of field 'u' and field 'v', showing how structures in one field induce responses in the other. High-contrast color scales (e.g., 'hot' for thermal and 'cold' for mechanical) emphasize the cross-field interactions. Clean scientific layout with synchronized axes and detailed annotations. 8k resolution, suitable for a computational physics journal."
  viz_type: "dual_channel_contour"
  channels: ["field_u", "field_v"]
  color_map: "hot_cold"
  layout: "overlaid_isocontours"
```
