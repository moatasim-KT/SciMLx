# SciML Terminology

## Core Concepts

- **Operator Learning:** Learning a mapping between infinite-dimensional
  function spaces (e.g., initial condition → solution at time T).  Neural
  operators approximate these mappings with finite parameters.

- **Physics-Informed:** Incorporating physical laws (PDE residuals in the loss)
  into neural network training.  Reduces data requirements; improves
  extrapolation.

- **Resolution Invariance:** A model trained on grid resolution N can evaluate
  at a different resolution M without retraining.  FNO and WNO achieve this;
  standard CNNs do not.

- **Relative L2 Error:** `||u_pred − u_gt||_2 / ||u_gt||_2`.  The standard
  metric in this repository.  Lower is better.

- **val_l2_rel:** The specific metric computed by `evaluate_l2_rel()` in
  `prepare.py` on the fixed validation set (256 samples, seed=42).

## Model Architectures

- **Spectral Convolution:** Convolution performed in the frequency domain via
  FFT.  Filters to the N_MODES lowest Fourier coefficients.  O(N log N)
  complexity.

- **FNO Block:** SpectralConv + pointwise Linear + GELU activation.
  The residual connection is built in: output = spectral(x) + linear(x).

- **UNO:** U-shaped Neural Operator.  Encodes the input at progressively coarser
  scales using FNO layers, then decodes back to full resolution with skip
  connections.  Captures both low- and high-frequency features.

- **WNO (Wavelet Neural Operator):** Uses Haar wavelet decomposition instead of
  Fourier modes.  Haar wavelets have compact spatial support → better locality
  and non-periodic boundary handling.

- **Branch & Trunk Networks (DeepONet):** Branch encodes the input function
  (e.g., initial condition); Trunk encodes evaluation coordinates.
  Output = inner product of Branch and Trunk outputs.

- **POD-DeepONet:** Variant where the Trunk is replaced by a fixed set of
  Proper Orthogonal Decomposition (POD) basis functions.  Branch predicts
  coefficients.  Efficient for low-dimensional solution manifolds.

- **AFNO (Adaptive FNO):** Applies block-diagonal MLPs in Fourier space with
  learnable sparsity (softshrink).  More expressive than dense spectral conv.

## Training Concepts

- **AdamW:** Adam optimiser with weight decay applied to parameters directly
  (decoupled regularisation).  Standard for neural operator training.

- **Cosine Warmdown:** Learning rate schedule: linear warmup → flat → cosine
  decay to `FINAL_LR_FRAC × LR`.  Allows aggressive early learning then
  fine convergence.

- **Gradient Clipping:** Scales gradients so their global L2 norm ≤ GRAD_CLIP.
  Prevents training instability from large gradient spikes (common in NS).

- **PINO Loss:** `L = L_data + λ · L_physics` where L_physics is the mean
  squared PDE residual evaluated at predicted solutions.

- **H1 / Sobolev Norm:** `||u||_{H1}² = ||u||_{L2}² + ||∂u/∂x||_{L2}²`.
  Using H1 loss penalises high-frequency prediction errors more strongly.

## PDE Terminology

- **Burgers' Equation:** `∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²`.  1D advection-
  diffusion; develops shocks at low ν.  Standard benchmark for 1D operators.

- **Darcy Flow:** `-∇·(a(x)∇u) = f`.  2D elliptic PDE; models flow in porous
  media.  Key challenge: spatially varying permeability `a(x)`.

- **Navier-Stokes (vorticity form):** `∂ω/∂t + (u·∇)ω = ν∇²ω`.  2D
  incompressible fluid dynamics; chaotic at low ν.

- **KdV (Korteweg-de Vries):** `∂u/∂t + u·∂u/∂x + ∂³u/∂x³ = 0`.  Models
  shallow water waves; admits soliton solutions.

- **IMEX Scheme:** Implicit-Explicit time integration.  Stiff terms (diffusion)
  treated implicitly; non-stiff terms (advection) explicitly.  Used in
  `solve_burgers_batch()`.

- **Dealiasing (2/3 rule):** Zero the top 1/3 of Fourier modes to prevent
  aliasing errors from nonlinear terms.  Standard in pseudo-spectral solvers.

- **Pseudo-Spectral Method:** Compute spatial derivatives in Fourier space,
  nonlinear products in physical space.  Spectrally accurate for periodic BCs.

## Metrics & Benchmarking

- **Conservation Laws:** Physical quantities that should remain constant (mass,
  energy, momentum).  Can be used as additional loss or diagnostic.

- **Relative L2:** Same as val_l2_rel; equals 0 for perfect prediction,
  ~1 for random-noise prediction.

- **H1 Error:** Includes gradient error alongside pointwise error.  More
  sensitive to sharp features and shocks.

- **Step time (dt):** Wall-clock time per gradient step.  On M-series Macs,
  expect 10–100 ms for 1D models, 100–500 ms for 2D.
