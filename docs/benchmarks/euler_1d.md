# 1D Compressible Euler Equations

**ID**: `euler_1d`
**Status**: [STABLE]
**Task ID**: `T2.1`

## Problem Formulation
The 1D compressible Euler equations in conservative form, representing the conservation of mass, momentum, and total energy for an ideal gas on a periodic domain $[0, 2\pi)$:

$$ \frac{\partial \rho}{\partial t} + \frac{\partial}{\partial x}(\rho u) = 0 $$
$$ \frac{\partial (\rho u)}{\partial t} + \frac{\partial}{\partial x}(\rho u^2 + p) = 0 $$
$$ \frac{\partial E}{\partial t} + \frac{\partial}{\partial x}((E + p)u) = 0 $$

where:
- $\rho$ is the density.
- $u$ is the velocity.
- $p$ is the pressure.
- $E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho u^2$ is the total energy, with $\gamma = 1.4$.

## Setup
- **Domain**: $[0, 2\pi)$ with periodic boundary conditions.
- **Grid Size**: $64$.
- **Temporal Horizon**: $T = 1.0$.
- **Time Steps**: $300$.
- **Numerical Solver**: 2nd-order Finite Volume scheme using MUSCL linear reconstruction (minmod limiter), HLL Riemann solver, and SSP-RK2 time integration.

## Initial Conditions
Smooth random Fourier series for $(\rho, u, p)$ ensuring subsonic flow and positivity. Amplitudes are chosen to prevent shock formation within the time horizon $T=1$.

## Boundary Conditions
Periodic on $[0, 2\pi)$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A multi-channel scientific visualization of the 1D compressible Euler equations. The plot features three synchronized subplots showing the evolution of density, velocity, and pressure. Smooth, correlated wave structures transition through the periodic domain. The aesthetics use a sophisticated 'cividis' color palette to distinguish the channels. Subtle grid lines, high-fidelity data points, and a professional layout suitable for an aerospace engineering publication. Sharp gradients without oscillations, emphasizing the MUSCL reconstruction quality."
  viz_type: "multi_channel_line_plot"
  channels: ["density", "velocity", "pressure"]
  color_map: "cividis"
  layout: "vertical_stacked_channels"
```
