# 1D Near-Inviscid Burgers Equation

**ID**: `burgers_nu_001`
**Status**: [STABLE]
**Task ID**: `T2.1`

## Problem Formulation
A variant of the Burgers' equation with a significantly lower viscosity, pushing the system into the near-inviscid regime. This results in extremely sharp shock fronts that are nearly discontinuous, challenging the model's ability to handle high-frequency gradients:

$$ \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} $$

where:
- $\nu = 0.001$ (low viscosity, strong shocks).

## Setup
- **Domain**: $[0, 2\pi)$ with periodic boundary conditions.
- **Grid Size**: $64$.
- **Temporal Horizon**: $T = 1.0$.
- **Time Steps**: $500$.
- **Numerical Solver**: Batch pseudo-spectral IMEX-Euler with 2/3-rule dealiasing.

## Initial Conditions
Same as `burgers_1d` (smooth random Fourier series with $k^{-1.5}$ decay).

## Boundary Conditions
Periodic on $[0, 2\pi)$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A high-resolution scientific visualization of the near-inviscid 1D Burgers equation. The image features a series of line plots at different time intervals, showing a smooth sine-like wave collapsing into a near-discontinuous vertical shock front. The gradients are incredibly steep, highlighting the low-viscosity regime. High-contrast colors (e.g., electric blue and deep crimson) define the wave profiles against a clean grid background. Professional scientific plotting style, high detail, sharp vector-like quality."
  viz_type: "line_profile_stack"
  channels: ["velocity"]
  color_map: "magma"
  layout: "multi_temporal_overlay"
```
