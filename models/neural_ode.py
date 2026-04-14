"""Neural ODEs and Universal Differential Equations (UDEs) for 1-D PDEs.

References:
  1. "Neural Ordinary Differential Equations"
     Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud
     NeurIPS 2018 — arXiv:1806.07366

  2. "Universal Differential Equations for Scientific Machine Learning"
     Christopher Rackauckas et al.
     arXiv:2001.04385

Source repo:
  https://github.com/moatasim-KT/SciML-and-Physics-Informed-Machine-Learning-Examples
  (universal-differential-equations example)

Key ideas:
  Neural ODE:
    Replace a discrete residual block with a continuous ODE:
      dh/dt = f_θ(h, t)
    The output is the solution h(T) obtained by numerical integration.
    Gradient computed via the adjoint method (or backprop through solver steps).

  Universal Differential Equation (UDE):
    Augment a KNOWN ODE/PDE with a neural network correction term:
      du/dt = known_physics(u, x) + NN_θ(u, x)
    The NN only needs to learn the UNKNOWN or MISSING terms.
    For Burgers: we know the nonlinear advection u*ux, but can discover nu*uxx.
    For wave:    we know the wave operator, NN corrects for boundary imperfections.

    Key benefit: far fewer parameters needed; NN is a correction, not the full model.

Implementations:
  NeuralODE1d      — Pure neural ODE: learn du/dt = F_θ(u, x), integrate to u(T).
                     n_steps Euler or RK4 steps during forward pass.
                     Best for: wave_1d (smooth, conservative), kdv_1d.

  UniversalDE1d    — UDE: du/dt = advection(u) + NN(u, x).
                     advection(u) = -u * ux computed via FFT (known Burgers term).
                     NN corrects the missing viscosity / higher-order terms.
                     Best for: burgers_1d (we know advection, NN learns diffusion).

  LatentODE1d      — Latent space ODE: encode u0 → z, integrate dz/dt = F_θ(z),
                     decode z(T) → u(T). Good for irregular time series / multiscale.
"""

import mlx.core as mx
import mlx.nn as nn
import math


# ── Derivative Network (shared by NeuralODE and UDE) ─────────────────────────

class DerivNet1d(nn.Module):
    """Neural network approximating the time derivative ∂u/∂t.

    Input:  [u(x), x_grid]  concatenated — [B, N, 2]
    Output: ∂u/∂t(x)        — [B, N, 1]

    A sequence of FNO blocks captures long-range spatial coupling in the
    derivative (important for wave and KdV), while a final pointwise layer
    projects to the scalar tendency.

    Parameters
    ----------
    hidden_dim : int
        Width of the FNO blocks.
    n_modes : int
        Fourier modes in the derivative network.
    n_layers : int
        Number of FNO blocks.
    """

    def __init__(self, hidden_dim: int = 32, n_modes: int = 16, n_layers: int = 3):
        super().__init__()
        from .fno import FNOBlock1d
        self.lift   = nn.Linear(2, hidden_dim)
        self.blocks = [FNOBlock1d(hidden_dim, n_modes) for _ in range(n_layers)]
        self.proj   = nn.Linear(hidden_dim, 1)

    def __call__(self, u: mx.array, grid: mx.array) -> mx.array:
        """u, grid : [B, N] → tendency : [B, N]"""
        x = mx.stack([u, grid], axis=-1)   # [B, N, 2]
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)[:, :, 0]       # [B, N]


# ── NeuralODE1d ───────────────────────────────────────────────────────────────

class NeuralODE1d(nn.Module):
    """Neural ODE for 1-D PDE operator learning.

    Learns ∂u/∂t = F_θ(u, x) and integrates n_steps times from u(0) to u(T).
    Integration uses a simple fixed-step RK4 scheme for accuracy.

    This is an autoregressive operator: the same F_θ is applied repeatedly,
    giving time-translation equivariance.

    Parameters
    ----------
    n_modes : int
        Fourier modes in the derivative network.
    hidden_dim : int
        Width of derivative network layers.
    n_layers : int
        Depth of derivative network.
    n_steps : int
        Number of integration steps from t=0 to t=T_final.
        More steps → more accurate but slower; 20 is a good default.

    Notes
    -----
    - Grid is always the unit interval [0, 1] with N points.
    - dt = 1.0 / n_steps (normalised time horizon = 1).
    - Euler update: u_{k+1} = u_k + dt * F_θ(u_k, x)
    - RK4 update applied for better accuracy.
    """

    def __init__(self, n_modes: int = 16, hidden_dim: int = 64, n_layers: int = 4,
                 n_steps: int = 20):
        super().__init__()
        self.n_steps = n_steps
        self.dt      = 1.0 / n_steps
        self.deriv   = DerivNet1d(hidden_dim=hidden_dim,
                                   n_modes=n_modes,
                                   n_layers=n_layers)

    def _rk4_step(self, u: mx.array, grid: mx.array, dt: float) -> mx.array:
        """One RK4 step: u → u + dt*F(u)."""
        k1 = self.deriv(u,               grid)
        k2 = self.deriv(u + 0.5*dt*k1,  grid)
        k3 = self.deriv(u + 0.5*dt*k2,  grid)
        k4 = self.deriv(u + dt*k3,       grid)
        return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def __call__(self, u0: mx.array) -> mx.array:
        B, N = u0.shape
        grid = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N), (B, N))
        u = u0
        for _ in range(self.n_steps):
            u = self._rk4_step(u, grid, self.dt)
        return u


# ── UniversalDE1d ─────────────────────────────────────────────────────────────

class UniversalDE1d(nn.Module):
    """Universal Differential Equation for 1-D Burgers.

    du/dt = -u * ∂u/∂x + NN_θ(u, x)
             ──────────   ──────────
             known term   unknown term (viscosity, higher-order corrections)

    The known term is the nonlinear Burgers advection computed spectrally.
    The NN only needs to learn the correction (viscous diffusion + model error).

    Key benefit: NN has far fewer parameters than a full neural operator since
    it only models the RESIDUAL of the known physics.

    Parameters
    ----------
    n_modes : int
        Fourier modes in the correction network.
    hidden_dim : int
        Width of correction network.
    n_layers : int
        Depth of correction network.
    n_steps : int
        Integration steps.
    nu : float
        Optional known viscosity; if > 0, also adds explicit diffusion
        (making the NN correction even smaller).
    """

    def __init__(self, n_modes: int = 16, hidden_dim: int = 32, n_layers: int = 3,
                 n_steps: int = 20, nu: float = 0.0):
        super().__init__()
        self.n_steps = n_steps
        self.dt      = 1.0 / n_steps
        self.nu      = nu
        # Small correction network (NN only learns the RESIDUAL)
        self.correction = DerivNet1d(hidden_dim=hidden_dim,
                                      n_modes=n_modes,
                                      n_layers=n_layers)

    def _spectral_advection(self, u: mx.array) -> mx.array:
        """Compute -u * ∂u/∂x via spectral differentiation with 2/3 dealiasing.

        ∂u/∂x = IFFT(ik * FFT(u)).  The 2/3 rule zeros the top third of modes
        before the nonlinear multiply to prevent aliasing-driven instability.
        """
        _, N = u.shape
        u_hat = mx.fft.rfft(u, axis=-1)          # [B, N//2+1] complex
        n_rfft = N // 2 + 1
        k = mx.arange(n_rfft, dtype=mx.float32)  # [n_rfft]

        # 2/3 dealiasing: zero modes above 2/3 * N/2
        k_max = int(n_rfft * 2 / 3)
        mask = (k < k_max).astype(mx.float32)
        u_hat_d = u_hat * mask[None, :]

        # Spectral derivative ∂u/∂x
        ux_hat_r = -u_hat_d.imag * k[None, :]
        ux_hat_i =  u_hat_d.real * k[None, :]
        ux_hat   = ux_hat_r + 1j * ux_hat_i
        ux = mx.fft.irfft(ux_hat, n=N, axis=-1)  # [B, N]
        return -u * ux   # nonlinear advection

    def _spectral_diffusion(self, u: mx.array) -> mx.array:
        """Compute nu * d^2u/dx^2 via spectral differentiation."""
        if self.nu == 0.0:
            return mx.zeros_like(u)
        _, N = u.shape
        u_hat  = mx.fft.rfft(u, axis=-1)
        n_rfft = N // 2 + 1
        k      = mx.arange(n_rfft, dtype=mx.float32)
        uxx_hat = -(k ** 2)[None, :] * u_hat
        uxx    = mx.fft.irfft(uxx_hat.real + 1j * uxx_hat.imag, n=N, axis=-1)
        return self.nu * uxx

    def _tendency(self, u: mx.array, grid: mx.array) -> mx.array:
        """Full tendency: known physics + NN correction."""
        known = self._spectral_advection(u) + self._spectral_diffusion(u)
        nn    = self.correction(u, grid)
        tendency = known + nn
        # Clip tendency magnitude to prevent RK4 stage blow-up during early training
        return mx.clip(tendency, -1e4, 1e4)

    def __call__(self, u0: mx.array) -> mx.array:
        B, N = u0.shape
        grid = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N), (B, N))
        u = u0
        dt = self.dt
        for _ in range(self.n_steps):
            k1 = self._tendency(u,               grid)
            k2 = self._tendency(u + 0.5*dt*k1,  grid)
            k3 = self._tendency(u + 0.5*dt*k2,  grid)
            k4 = self._tendency(u + dt*k3,       grid)
            u  = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return u


# ── LatentODE1d ───────────────────────────────────────────────────────────────

class LatentODE1d(nn.Module):
    """Latent Neural ODE for 1-D operator learning.

    Encode u0 into a latent vector z, integrate dz/dt = G_θ(z), decode z(T) → u(T).

    This is more memory-efficient than NeuralODE1d (the ODE runs in latent space,
    not full-resolution grid) and handles multiscale / irregular features better.

    Architecture:
      encoder: u0[N] → z[latent_dim]   (MLP)
      ode_fn:  z[latent_dim] → dz/dt    (small MLP, n_steps Euler steps)
      decoder: z[latent_dim] + x[N] → u(T)[N]  (DeepONet-style: latent dot trunk)

    Parameters
    ----------
    n_sensors : int
        Grid size N.
    hidden_dim : int
        Encoder/decoder MLP width.
    latent_dim : int
        ODE state dimension. Default = hidden_dim // 2.
    n_layers : int
        Encoder/decoder MLP depth.
    n_steps : int
        Euler integration steps in latent space.
    n_modes : int
        Accepted for API compatibility (unused).
    """

    def __init__(self, n_sensors: int, hidden_dim: int = 64, latent_dim: int = 0,
                 n_layers: int = 4, n_steps: int = 20, n_modes: int = 16):
        super().__init__()
        if latent_dim == 0:
            latent_dim = hidden_dim // 2
        self.latent_dim = latent_dim
        self.n_steps    = n_steps
        self.dt         = 1.0 / n_steps

        def _mlp(in_d, out_d, act=nn.gelu):
            dims = [in_d] + [hidden_dim] * n_layers + [out_d]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(act)
            return nn.Sequential(*layers)

        self.encoder = _mlp(n_sensors, latent_dim)
        # Small ODE function in latent space
        self.ode_fn  = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.Tanh(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        # Trunk for decoding
        self.trunk   = _mlp(1, latent_dim)
        self.bias    = mx.zeros([1])

    def __call__(self, u0: mx.array) -> mx.array:
        _, N = u0.shape

        # Encode IC -> latent state
        z = self.encoder(u0)   # [B, latent_dim]

        # Integrate in latent space (Euler)
        for _ in range(self.n_steps):
            z = z + self.dt * self.ode_fn(z)

        # Decode: inner product with grid trunk
        grid = mx.linspace(0.0, 1.0, N)[:, None]  # [N, 1]
        T    = self.trunk(grid)                     # [N, latent_dim]
        out  = mx.matmul(z, T.T) + self.bias       # [B, N]
        return out
