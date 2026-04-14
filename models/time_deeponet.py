"""Time-Marching DeepONet (TimeDeepONet).

Inspired by:
  "Time-marching neural operator-FE coupling: AI-accelerated physics modeling"
  Wei Wang, Maryam Hakimzadeh, Haihui Ruan, Somdatta Goswami
  Computer Methods in Applied Mechanics and Engineering, vol. 446, 2025
  DOI: https://doi.org/10.1016/j.cma.2025.118319
  arXiv: https://arxiv.org/abs/2504.11383

Original paper key idea:
  FE-NO coupling via domain decomposition.  A physics-informed DeepONet with TWO
  branch networks replaces the fine-mesh FE subdomain:
    • branch1: encodes displacement BC at the CURRENT time step
    • branch2: encodes full displacement + velocity from the PREVIOUS time step
    • trunk:   encodes spatial coordinates
    • output = (branch1 ⊙ branch2) · trunk  (element-wise product then dot with trunk)
  Temporal integration via Newmark-β method; spatial coupling via Schwarz alternating
  method at the overlapping boundary.

Adaptation to our 1-D benchmarks:
  For wave_1d and ns_2d-type problems, we adapt the two-branch structure to
  encode (a) the initial condition and (b) a time-aware context encoding, so the
  model can be tested within our standard (IC → future state) training loop.

  TimeDeepONet1d:
    • branch1 (IC branch):   encodes u₀[N] → b1[p]           (global IC context)
    • branch2 (time branch): encodes t ∈ [0,1] → b2[p]       (temporal gating)
    • trunk:                 encodes x[N] ∈ [0,1]^N → T[N,p] (spatial basis)
    • output[n] = Σ_p b1[p] * b2[p] * T[n,p]  + bias

  This generalises standard DeepONet to be time-aware:
    - b2(t) gates which IC features are active at time t
    - recovers standard DeepONet when b2 is set to all-ones

  DualBranchDeepONet1d:
    For problems where two input fields are available (e.g., initial displacement
    AND initial velocity), encodes both separately and combines:
    • branch1: u0[N] → b1[p]
    • branch2: v0[N] → b2[p]
    • trunk:   x[N]  → T[N,p]
    • output[n] = (b1[p] + b2[p]) · T[n,p]

    Useful for wave_1d (u and du/dt are both known at t=0).
"""

import mlx.core as mx
import mlx.nn as nn


def _mlp(dims: list[int], act=nn.gelu) -> nn.Sequential:
    """Build a simple MLP with GELU activations."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act)
    return nn.Sequential(*layers)


# ── TimeDeepONet1d ────────────────────────────────────────────────────────────

class TimeDeepONet1d(nn.Module):
    """Time-aware DeepONet for 1-D temporal PDEs.

    branch1 encodes the initial condition u0 (sensor values at N grid points).
    branch2 encodes a scalar time t ∈ [0, 1] expanded to a learnable embedding.
    trunk   encodes query coordinates x ∈ [0, 1].

    Parameters
    ----------
    n_sensors : int
        Number of sensor / grid points in the IC (= N for 1-D).
    hidden_dim : int
        Width of all MLP layers.
    p : int
        Latent basis dimension (output of each branch; also the trunk output width).
        Larger p → more expressive decomposition.
    n_layers : int
        Depth of each branch and trunk MLP.
    time_embed_dim : int
        Sinusoidal + learned embedding dim for the scalar time input.

    Forward
    -------
    u0 : [B, N]   — initial condition
    t  : [B]      — query time (normalised to [0, 1])

    Returns
    -------
    pred : [B, N] — predicted solution at time t on the same N-point grid
    """

    def __init__(self, n_sensors: int, hidden_dim: int = 64, p: int = 64,
                 n_layers: int = 4, time_embed_dim: int = 16):
        super().__init__()
        self.p = p
        self.n_sensors    = n_sensors
        self.time_emb_dim = time_embed_dim

        # Branch 1: IC encoder  [B, N] → [B, p]
        b1_dims = [n_sensors] + [hidden_dim] * n_layers + [p]
        self.branch1 = _mlp(b1_dims)

        # Branch 2: time encoder [B, time_embed_dim] → [B, p]
        b2_dims = [time_embed_dim] + [hidden_dim] * n_layers + [p]
        self.branch2 = _mlp(b2_dims)

        # Trunk: coordinate encoder [N, 1] → [N, p]
        t_dims = [1] + [hidden_dim] * n_layers + [p]
        self.trunk   = _mlp(t_dims)

        # Output bias
        self.bias = mx.zeros([1])

    def _time_embedding(self, t: mx.array) -> mx.array:
        """Sinusoidal time embedding, [B] → [B, time_emb_dim]."""
        d = self.time_emb_dim
        half = d // 2
        freqs = mx.exp(-mx.arange(half, dtype=mx.float32) * (math.log(10000) / max(half - 1, 1)))
        # [B, 1] * [half] → [B, half]
        t_col = t[:, None]
        args  = t_col * freqs[None, :]
        emb   = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)  # [B, d]
        return emb

    def __call__(self, u0: mx.array, t: mx.array | None = None) -> mx.array:
        B, N = u0.shape

        # Default t=1 (predict at final time, compatible with existing loaders)
        if t is None:
            t = mx.ones([B])

        # Branch outputs: [B, p]
        b1  = self.branch1(u0)
        b2  = self.branch2(self._time_embedding(t))
        # Element-wise product (learned gating of IC features by time)
        b12 = b1 * b2   # [B, p]

        # Trunk: query at all N grid points — evaluate on 1-D unit grid
        grid = mx.linspace(0.0, 1.0, N)[:, None]   # [N, 1]
        T    = self.trunk(grid)                      # [N, p]

        # Output: [B, p] @ [p, N] → [B, N]
        out = mx.matmul(b12, T.T) + self.bias
        return out


# ── DualBranchDeepONet1d ──────────────────────────────────────────────────────

class DualBranchDeepONet1d(nn.Module):
    """Two-branch DeepONet for problems with two input fields.

    Captures the structural idea of the Time-Marching paper (two branch networks
    encoding separate physical information) without requiring FE solver coupling.

    branch1: encodes displacement IC u0[N]          → b1[p]
    branch2: encodes velocity IC    v0[N] or BC[N]  → b2[p]
    trunk:   encodes coordinates    x[N]            → T[N,p]
    output:  (b1 + b2) · T   (additive combination of two contexts)

    Useful for wave_1d where both u(x,0) and u_t(x,0) are given.

    Parameters
    ----------
    n_sensors : int
        Number of sensor / grid points (= N).
    hidden_dim : int
        MLP width for both branches and trunk.
    p : int
        Latent basis dimension.
    n_layers : int
        MLP depth.
    """

    def __init__(self, n_sensors: int, hidden_dim: int = 64, p: int = 64,
                 n_layers: int = 4):
        super().__init__()
        self.p = p
        dims = [n_sensors] + [hidden_dim] * n_layers + [p]
        self.branch1 = _mlp(dims)
        self.branch2 = _mlp(dims)
        t_dims = [1] + [hidden_dim] * n_layers + [p]
        self.trunk   = _mlp(t_dims)
        self.bias    = mx.zeros([1])

    def __call__(self, u0: mx.array,
                 v0: mx.array | None = None) -> mx.array:
        _, N = u0.shape
        # If no second field provided, use a zero-velocity IC
        if v0 is None:
            v0 = mx.zeros_like(u0)

        b1   = self.branch1(u0)        # [B, p]
        b2   = self.branch2(v0)        # [B, p]
        b12  = b1 + b2                 # additive: both contexts contribute

        grid = mx.linspace(0.0, 1.0, N)[:, None]
        T    = self.trunk(grid)        # [N, p]
        out  = mx.matmul(b12, T.T) + self.bias
        return out


# ── math import (needed for _time_embedding) ──────────────────────────────────
import math
