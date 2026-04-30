"""
TestNet — SciML Neural Operator (MLX)

Auto-generated stub by model_scaffold.py.
Base architecture: FNO
Notes: `hidden_dim=64, n_layers=4`
"""

import mlx.core as mx
import mlx.nn as nn
from .fno import SpectralConv1d

class TestNet(nn.Module):
    def __init__(self, n_modes: int = 16, hidden_dim: int = 64,
                 n_layers: int = 4, **kwargs):
        super().__init__()
        self.n_modes    = n_modes
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.lift = nn.Linear(1, hidden_dim)
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
        h = self.lift(x[..., None])          # [B, N, H]
        for block in self.blocks:
            h = h + block(h)
        return self.proj(h).squeeze(-1)      # [B, N]
