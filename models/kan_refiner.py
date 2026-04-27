"""Neural-Symbolic Symbiosis: FNO with KAN-based symbolic refinement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fno import FNO1d
from models.kan import KANLinear

class KANRefinedFNO1d(nn.Module):
    """
    A hybrid model that uses FNO to capture the global field and 
    a KAN (Kolmogorov-Arnold Network) as a symbolic refiner for 
    sharp gradients and local details.
    """
    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, kan_grid: int = 5):
        super().__init__()
        # Global backbone
        self.fno = FNO1d(n_modes, hidden_dim, n_layers)
        
        # Symbolic refiner
        # Takes (x, fno_output) and refines it
        self.refiner = nn.Sequential(
            KANLinear(2, hidden_dim // 2, grid_size=kan_grid),
            nn.SiLU(),
            KANLinear(hidden_dim // 2, 1, grid_size=kan_grid)
        )

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        """u0: [B, N]"""
        B, N = u0.shape
        grid = torch.linspace(0.0, 1.0, N, device=u0.device).view(1, N).expand(B, N)
        
        # 1. Global FNO pass
        y_global = self.fno(u0) # [B, N]
        
        # 2. Refinement pass
        # Features: spatial grid and global prediction
        ref_input = torch.stack([grid, y_global], dim=-1) # [B, N, 2]
        y_refined = self.refiner(ref_input).squeeze(-1) # [B, N]
        
        # Final prediction is residual refinement
        return y_global + y_refined

class KANSymbolicBottleneck(nn.Module):
    """
    An FNO variant where the latent space is compressed and then 
    symbolically expanded by a KAN layer.
    """
    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, kan_grid: int = 5):
        super().__init__()
        from models.fno import FNOBlock1d
        
        self.lift = nn.Linear(2, hidden_dim)
        self.blocks = nn.ModuleList([FNOBlock1d(hidden_dim, n_modes) for _ in range(n_layers)])
        
        # Symbolic bottleneck expansion
        self.kan_bottleneck = KANLinear(hidden_dim, hidden_dim, grid_size=kan_grid)
        
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N = u0.shape
        grid = torch.linspace(0.0, 1.0, N, device=u0.device).view(1, N).expand(B, N)
        x = torch.stack([u0, grid], dim=-1)
        
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
            
        # Apply symbolic transformation in latent space
        x = self.kan_bottleneck(x)
        
        return self.proj(x).squeeze(-1)
