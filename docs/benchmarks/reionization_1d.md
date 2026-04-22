# 1D Cosmic Reionization Proxy

**ID**: `reionization_1d`
**Status**: [STABLE]
**Task ID**: `T2.1`

## Problem Formulation
A non-linear advection-reaction model serving as a toy proxy for the propagation of an ionization front during the epoch of cosmic reionization. The model captures the balance between source radiation and recombination:

$$ \frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = S(x) - \alpha u^2 $$

where:
- $u(x, t)$ is the ionization fraction.
- $c = 1.0$ is the propagation speed.
- $S(x)$ is a random spatial source field.
- $\alpha = 0.1$ is the recombination coefficient.

## Setup
- **Domain**: $[0, 1]$ with periodic boundary conditions.
- **Grid Size**: $64$.
- **Temporal Horizon**: $T = 0.5$.
- **Time Steps**: $100$.
- **Numerical Solver**: 1st-order upwind advection scheme combined with an explicit reaction step.

## Initial Conditions
Initialized from the source field $S(x)$, which is generated using a low-mode random Fourier series ($k=3$) shifted to ensure positivity.

## Boundary Conditions
Periodic on $[0, 1]$.

## Visualization Metadata
```yaml
viz_metadata:
  prompt: "A conceptual scientific visualization of an ionization front propagating in a 1D cosmological medium. The image features a glowing, nebulous gradient transitioning from deep void black to brilliant hydrogen-alpha red. A sharp, moving boundary represents the ionization front. Background elements include faint, stylized star-like source points that correspond to the source field S(x). Cinematic lighting, scientific accuracy combined with cosmic aesthetics, high resolution, 8k, ethereal feel."
  viz_type: "glow_heatmap"
  channels: ["ionization_fraction"]
  color_map: "inferno"
  layout: "single_channel_temporal"
```
