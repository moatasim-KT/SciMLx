# 2D Linearized Shallow Water Equations

**ID**: `swe_2d`
**Status**: [STABLE]
**Task ID**: `T3.3`

## Problem Formulation
The 2D shallow water equations linearized around a quiescent state ($H_0=1$, $u=v=0$), modeling the propagation of gravity waves in a shallow fluid layer on $[0, 2\pi)^2$:

$$ \frac{\partial^2 h}{\partial t^2} = g H_0 \nabla^2 h $$

where:
- $h(x, y, t)$ is the surface height anomaly.
- $g = 9.81$ is the acceleration due to gravity.
- $H_0 = 1.0$ is the mean depth.
- The wave speed is $c = \sqrt{gH_0} \approx 3.13$.

## Setup
- **Domain**: $[0, 2\pi)^2$ with periodic boundary conditions.
- **Grid Size**: $64 \times 64$.
- **Temporal Horizon**: $T = 1.0$.
- **Numerical Solver**: Exact analytic Fourier propagator (machine precision, zero truncation error).

## Initial Conditions
Smooth random surface height anomaly $h_0(x, y)$ generated from a truncated Fourier series ($k=4$). The system is released from rest ($\frac{\partial h}{\partial t}|_0 = 0$).

## Boundary Conditions
Periodic in $x$ and $y$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A scientific visualization of 2D gravity wave propagation on a shallow water surface. The image shows a complex pattern of intersecting ripple fronts and dispersive waves. A 'ocean' or 'Blues' color map is used to represent height displacement, with highlight glints on the wave crests. The background is a clean, neutral gray. The rendering captures the fluid-like motion and mathematical beauty of wave interference patterns. 8k resolution, professional data viz aesthetic."
  viz_type: "wave_interference_2d"
  channels: ["height"]
  color_map: "ocean"
  layout: "target_field"
```
