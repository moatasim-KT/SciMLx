# 1D Wave Equation

**ID**: `wave_1d`
**Status**: [STABLE]
**Task ID**: `T2.1`

## Problem Formulation
The standard second-order wave equation on a 1D periodic domain, modeling linear wave propagation:

$$ \frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2} $$

where:
- $c = 1.0$ is the constant wave speed.

## Setup
- **Domain**: $[0, 2\pi)$ with periodic boundary conditions.
- **Grid Size**: $64$.
- **Temporal Horizon**: $T = 1.0$.
- **Time Steps**: $400$.
- **Numerical Solver**: Symplectic Störmer-Verlet integrator (energy-conserving).

## Initial Conditions
Smooth random initial displacement $u_0(x)$ from a truncated Fourier series ($k=8$). Initial velocity $\frac{\partial u}{\partial t}|_0$ is set to zero (released from rest).

## Boundary Conditions
Periodic on $[0, 2\pi)$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A scientific visualization of the 1D linear wave equation propagation. The image shows a smooth, undulating wave profile as it splits into two identical pulses traveling in opposite directions across a periodic domain. The visualization is presented as a stack of thin, semi-transparent wave profiles colored with a gradient from soft blue to vibrant purple, creating a 3D-like effect of time progression. Clean scientific aesthetic, grid lines, high-quality vector-like rendering, professional data visualization style."
  viz_type: "stack_profile"
  channels: ["displacement"]
  color_map: "viridis"
  layout: "3d_time_stack"
```
