"""Deep Operator Network (DeepONet) implementations (PyTorch/CUDA)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepONet(nn.Module):
    """Deep Operator Network (DeepONet)."""

    def __init__(self, branch_dim: int, trunk_dim: int,
                 hidden_dim: int = 128, out_dim: int = 128, n_layers: int = 4):
        super().__init__()

        def _make_net(in_dim: int) -> nn.Module:
            layers = []
            dim = in_dim
            for _ in range(n_layers):
                layers.append(nn.Linear(dim, hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(hidden_dim))
                dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, out_dim))
            return nn.Sequential(*layers)

        self.branch = _make_net(branch_dim)
        self.trunk  = _make_net(trunk_dim)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, u: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            u : [B, branch_dim]
            y : [B, N, trunk_dim] (Optional, if None assumes grid provided in u)
        """
        if y is None:
            # Handle case where u is [B, N] and we need to evaluate at fixed grid
            # This logic depends on how data is packed in this project.
            # Standard DeepONet takes sensors u and query points y.
            return self.forward_fixed(u)

        b_out = self.branch(u)           # [B, out_dim]
        t_out = self.trunk(y)            # [B, N, out_dim]
        return torch.einsum("bo,bno->bn", b_out, t_out) + self.bias

    def forward_fixed(self, u: torch.Tensor):
        # Placeholder for project-specific fixed grid logic if needed
        B, N = u.shape
        grid = torch.linspace(0, 1, N, device=u.device).view(1, N, 1).expand(B, N, 1)
        return self.forward(u, grid)

class PODDeepONet(nn.Module):
    """Proper Orthogonal Decomposition DeepONet."""

    def __init__(self, branch_dim: int, n_basis: int = 64,
                 hidden_dim: int = 128, n_layers: int = 3, grid_size: int = 64):
        super().__init__()
        self.n_basis = n_basis

        layers = [nn.Linear(branch_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
        layers.append(nn.Linear(hidden_dim, n_basis))
        self.branch = nn.Sequential(*layers)

        self.basis = nn.Parameter(torch.randn(n_basis, grid_size) * 0.02)
        self.bias  = nn.Parameter(torch.zeros(1))

    def forward(self, u: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        coeffs = self.branch(u)                         # [B, n_basis]
        return torch.matmul(coeffs, self.basis) + self.bias  # [B, N]
