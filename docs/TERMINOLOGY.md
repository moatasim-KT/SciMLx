# SciML Terminology

## General Concepts
- **Operator Learning:** Learning the mapping between infinite-dimensional function spaces (e.g., initial condition to solution).
- **Physics-Informed:** Incorporating physical laws (usually as PDE residuals in the loss function) into the neural network training.
- **Resolution Invariance:** The property where a model trained on one grid resolution can be evaluated on another without retraining.

## Model Architectures
- **Spectral Convolution:** Convolution performed in the frequency domain using FFT, often filtering out high-frequency modes.
- **Branch & Trunk Networks:** The two components of a DeepONet; the branch net encodes the input function, and the trunk net encodes the evaluation coordinates.
- **Latent Dynamics:** Modeling the evolution of the system in a reduced-dimensional latent space (e.g., Neural ODEs).

## Metrics
- **Relative L2 Error:** $\frac{||u_{pred} - u_{gt}||_2}{||u_{gt}||_2}$. The standard metric for comparing PDE solver accuracy.
- **Conservation Laws:** Physical quantities (e.g., mass, energy) that should remain constant over time according to the PDE.
