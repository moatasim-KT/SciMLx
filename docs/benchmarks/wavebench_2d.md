# 2D Wave Propagation (WaveBench)

**ID**: `wavebench_2d`
**Status**: [STABLE]
**Task ID**: `T4.3`

## Problem Formulation
A 2D wave propagation benchmark designed to test a model's ability to resolve high-frequency harmonic components on $[0, 2\pi)^2$:

$$ \frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u $$

where:
- $c = 2.0$ is the constant wave speed.

## Setup
- **Domain**: $[0, 2\pi)^2$ with periodic boundary conditions.
- **Grid Size**: $64 \times 64$.
- **Temporal Horizon**: $T = 1.0$.
- **Numerical Solver**: Exact analytic Fourier propagator (zero truncation error).

## Initial Conditions
Smooth random surface anomaly generated with an increased number of Fourier modes ($k=12$) to test spectral bias and high-frequency resolution. Released from rest.

## Boundary Conditions
Periodic in $x$ and $y$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A high-contrast scientific visualization of 2D wave interference. The image shows complex, sharp wave fronts and ripples expanding and overlapping across a periodic domain. A 'twilight' or 'shifted-viridis' colormap creates a sense of depth and dynamic motion. The crests and troughs are highly detailed, showcasing the high-frequency components of the simulation. Clean, minimalist presentation, high-resolution scientific data visualization."
  viz_type: "harmonic_waves_2d"
  channels: ["amplitude"]
  color_map: "twilight"
  layout: "target_full"
```
