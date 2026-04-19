import mlx.core as mx
import mlx.nn as nn
from models.fno import SpectralConv1d

class ExplosionFNO(nn.Module):
    def __init__(self, n_modes: int = 16, hidden_dim: int = 64,
                 n_layers: int = 4, **kwargs):
        super().__init__()
        self.n_modes    = n_modes
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.lift = nn.Linear(1, hidden_dim)

        # The "bomb" layer to induce NaN
        self.bomb = nn.Linear(hidden_dim, hidden_dim)
        # Initialize with huge values to ensure explosion
        self.bomb.weight = mx.full((hidden_dim, hidden_dim), 1e20)

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
        # Lift to hidden dim
        h = self.lift(x[..., None])          # [B, N, hidden_dim]

        # Apply bomb
        h = self.bomb(h)

        # Apply operator blocks
        for block in self.blocks:
            h = h + block(h)                 # residual

        # Project back to scalar field
        out = self.proj(h).squeeze(-1)       # [B, N]
        return out
