# 1D Korteweg-de Vries (KdV) Equation

**ID**: `kdv_1d`
**Status**: [STABLE]
**Task ID**: `T2.1`

## Problem Formulation
The Korteweg-de Vries equation is a mathematical model of waves on shallow water surfaces, known for its soliton solutions where non-linear steepening is perfectly balanced by dispersion:

$$ \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + \frac{\partial^3 u}{\partial x^3} = 0 $$

## Setup
- **Domain**: $[0, 2\pi)$ with periodic boundary conditions.
- **Grid Size**: $64$.
- **Temporal Horizon**: $T = 1.0$.
- **Time Steps**: $1000$.
- **Numerical Solver**: Spectral ETDRK4 (4th-order Exponential Time Differencing Runge-Kutta).

## Initial Conditions
Smooth random Fourier series with spectral modes truncated at $k=8$. ICs are designed to trigger soliton interactions.

## Boundary Conditions
Periodic on $[0, 2\pi)$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A professional scientific visualization of the 1D Korteweg-de Vries (KdV) equation. The plot depicts the interaction of several distinct solitons—stable, self-reinforcing wave packets—as they travel through each other without changing shape. A space-time heatmap (time on vertical axis) reveals the characteristic diagonal crossing patterns of the solitons. The colors are vibrant and saturated, emphasizing the wave crests. Scientific simulation aesthetic, high precision, 8k resolution, mathematical elegance."
  viz_type: "heatmap_1d"
  channels: ["wave_height"]
  color_map: "plasma"
  layout: "temporal_waterfall"
```
