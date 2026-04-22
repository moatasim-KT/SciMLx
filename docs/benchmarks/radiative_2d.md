# 2D Radiative Transfer Proxy

**ID**: `radiative_2d`
**Status**: [STABLE]
**Task ID**: `T4.3`

## Problem Formulation
A proxy for radiative transfer modeling highly forward-peaked scattering. The domain represents one spatial dimension $x$ and one angular dimension $\theta$, wrapped into a 2D tensor $u(x, \theta)$ on $[0, 2\pi)^2$:

$$ \frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = \text{Scattering}(\theta) $$

where:
- $u(x, \theta, t)$ is the intensity.
- The scattering term is modeled as a blurring operator in the angular dimension.

## Setup
- **Domain**: $x \in [0, 2\pi), \theta \in [0, 2\pi)$ with periodic boundary conditions.
- **Grid Size**: $64 \times 64$.
- **Temporal Horizon**: $T = 1.0$.
- **Time Steps**: $10$ (Advection-Scattering splitting).
- **Numerical Solver**: Fourier pseudo-spectral method.

## Initial Conditions
Smooth random 2D field generated from a truncated Fourier series ($k=4$, scale=1.0).

## Boundary Conditions
Periodic in both $x$ and $\theta$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A specialized scientific visualization of a radiative transfer simulation. The image shows the intensity distribution $u(x, \theta)$ as a 2D field, where the horizontal axis represents space and the vertical axis represents angle. Sharp advection patterns in the spatial dimension are balanced by smooth scattering 'blurring' in the angular dimension. A 'magma' colormap represents the intensity levels. High-tech scientific aesthetic, clear axes, professional data visualization."
  viz_type: "phase_space_map"
  channels: ["intensity"]
  color_map: "magma"
  layout: "space_angle_tensor"
```
