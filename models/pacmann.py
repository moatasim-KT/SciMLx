"""
PACMANN — SciML Neural Operator

Base architecture: PINN/FNO hybrid.
"""

import torch
import torch.nn as nn
from models.fno import SpectralConv1d
from core.device import DEVICE

class PACMANN(nn.Module):
    """
    PACMANN: Physics-Augmented Convolutional Neural Network.

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

        self.blocks = nn.ModuleList([
            SpectralConv1d(hidden_dim, hidden_dim, n_modes)
            for _ in range(n_layers)
        ])
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N] input field
        Returns:
            out: [B, N] output field
        """
        # Lift to hidden dim
        h = self.lift(x.unsqueeze(-1))          # [B, N, hidden_dim]

        # Apply operator blocks
        for block in self.blocks:
            h = h + block(h)                 # residual

        # Project back to scalar field
        out = self.proj(h).squeeze(-1)       # [B, N]
        return out
