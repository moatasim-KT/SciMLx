"""Physics-Informed Neural Network (PINN) for SciML (PyTorch/CUDA)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PINN(nn.Module):
    """Standard MLP-based PINN for mapping coordinates to solution."""
    
    def __init__(self, in_dim: int, hidden_dim: int = 128, n_layers: int = 4, out_dim: int = 1):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PINO1d(nn.Module):
    """Physics-Informed Neural Operator (MLP-surrogate)."""
    def __init__(self, sensor_dim: int, hidden_dim: int = 128, n_layers: int = 6):
        super().__init__()
        self.net = PINN(sensor_dim + 1, hidden_dim, n_layers, out_dim=1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N = u0.shape
        grid = torch.linspace(0, 1, N, device=u0.device).view(1, N, 1).expand(B, N, 1)
        
        # Concat u0 sensors with each grid point [B, N, N+1]
        u0_expanded = u0.unsqueeze(1).expand(B, N, N)
        x_in = torch.cat([u0_expanded, grid], dim=-1)
        
        return self.net(x_in).squeeze(-1)


class ModalPINN(nn.Module):
    """PINN with modal decomposition."""
    def __init__(self, n_modes: int, sensor_dim: int, grid_size: int = 64, 
                 hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.n_modes = n_modes
        self.grid_size = grid_size
        self.coeff_net = PINN(sensor_dim, hidden_dim, n_layers, out_dim=n_modes)
        self.modes = nn.Parameter(torch.randn(grid_size, n_modes) * (grid_size * n_modes)**-0.5)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        coeffs = self.coeff_net(u0) # [B, n_modes]
        return torch.matmul(coeffs, self.modes.t())
