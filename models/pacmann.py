"""
PACMANN — SciML Neural Operator

Auto-generated stub by model_scaffold.py.
Base architecture: PINN
Notes: fill in architecture details

Edit this file to implement the model, then validate with:
    uv run model_scaffold.py --validate PACMANN models/pacmann.py
"""

import mlx.core as mx
import mlx.nn as nn
from models.fno import SpectralConv1d   # reuse existing building blocks


class PACMANN(nn.Module):
    """
    PACMANN: extend description here.

    Args:
        n_modes   : number of Fourier modes to keep
        hidden_dim: channel width
        n_layers  : number of operator blocks
    """

    def __init__(self, n_modes: int = 16, hidden_dim: int = 64,
                 n_layers: int = 4, **kwargs):
        super().__init__()
        self.n_modes    = n_modes
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.lift = nn.Linear(1, hidden_dim)

        # TODO: replace with your custom operator blocks
        self.blocks = [
            SpectralConv1d(hidden_dim, hidden_dim, n_modes)
            for _ in range(n_layers)
        ]
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, N] input field
        Returns:
            out: [B, N] output field
        """
        # Lift to hidden dim
        h = self.lift(x[..., None])          # [B, N, hidden_dim]

        # Apply operator blocks
        for block in self.blocks:
            h = h + block(h)                 # residual

        # Project back to scalar field
        out = self.proj(h).squeeze(-1)       # [B, N]
        return out
