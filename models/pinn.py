"""Physics-Informed Neural Network (PINN) for SciML.

Standard PINN implementation: a coordinate-based MLP that maps spatial and 
temporal coordinates (x, t) to the PDE solution u(x, t).

Reference:
  Raissi, Perdikaris, & Karniadakis (2019) "Physics-informed neural networks: 
  A deep learning framework for solving forward and inverse problems involving 
  nonlinear partial differential equations" (Journal of Computational Physics)
"""

import math
import mlx.core as mx
import mlx.nn as nn

class PINN(nn.Module):
    """Standard MLP-based PINN for mapping coordinates to solution.
    
    Can be used in two modes:
      1. Function approximator: (x, t) -> u(x, t)
      2. Operator surrogate: (u0, x, t) -> u(x, t) [DeepONet style]
    
    In this project, we primarily focus on mapping the initial condition u0 
    (represented as sensor samples) to the solution field u(x).
    """
    
    def __init__(self, in_dim: int, hidden_dim: int = 128, n_layers: int = 4, out_dim: int = 1):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(n_layers):
            # Using Tanh activations (classic PINN) or Sine (SIREN style)
            # are common for PINNs to ensure smooth higher-order derivatives.
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU()) # Project default is GELU
            layers.append(nn.LayerNorm(hidden_dim))
            dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, ..., in_dim] - coordinates or (u0, coords) concat
        Returns:
            [...] - solution values
        """
        return self.net(x)

class PINO1d(nn.Module):
    """Physics-Informed Neural Operator (MLP-surrogate).
    
    DeepONet-like structure but implemented as a single MLP taking
    (u0_sensors, x_coord) as input.
    """
    def __init__(self, sensor_dim: int, hidden_dim: int = 128, n_layers: int = 6):
        super().__init__()
        # Input: sensor values u0(x_i) + coordinate x
        self.net = PINN(sensor_dim + 1, hidden_dim, n_layers, out_dim=1)

    def __call__(self, u0: mx.array) -> mx.array:
        """
        Args:
            u0: [B, N] sensor samples of initial condition
        Returns:
            [B, N] predicted field
        """
        B, N = u0.shape
        # Create coordinate grid [B, N, 1]
        grid = mx.linspace(0, 1, N).reshape(1, N, 1)
        grid = mx.broadcast_to(grid, (B, N, 1))
        
        # Concat u0 sensors with each grid point [B, N, N+1]
        # This is expensive for large N! Better for DeepONet architecture.
        # But for N=64, it's 64*65 = 4160 floats per batch item.
        u0_expanded = mx.broadcast_to(u0[:, None, :], (B, N, N))
        x_in = mx.concatenate([u0_expanded, grid], axis=-1)
        
        return self.net(x_in)[:, :, 0]


class ModalPINN(nn.Module):
    """PINN with modal decomposition (e.g. POD, Fourier) prior.
    u(x, t) = sum_i c_i(u0) * phi_i(x, t)
    
    phi_i are pre-computed or fixed basis functions (modes).
    The network predicts coefficients c_i from sensor data u0.
    
    Based on EPFL ML4Science "ModalPINN" for airflow reconstruction (2024).
    """
    def __init__(self, n_modes: int, sensor_dim: int, grid_size: int = 64, 
                 hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.n_modes = n_modes
        self.grid_size = grid_size
        # MLP maps sensor data to coefficients [B, sensor_dim] -> [B, n_modes]
        self.coeff_net = PINN(sensor_dim, hidden_dim, n_layers, out_dim=n_modes)
        # Learnable modes [grid_size, n_modes] - in practice, initialize with POD modes
        self.modes = mx.random.normal([grid_size, n_modes]) * (grid_size * n_modes)**-0.5

    def __call__(self, u0: mx.array) -> mx.array:
        """
        Args:
            u0: [B, N] sensor values of initial condition or boundary
        Returns:
            [B, N] reconstructed field on full grid
        """
        # 1. Predict coefficients from sensors
        coeffs = self.coeff_net(u0) # [B, n_modes]
        
        # 2. Linear combination of basis modes
        # [B, n_modes] @ [n_modes, N] -> [B, N]
        return mx.matmul(coeffs, self.modes.T)
