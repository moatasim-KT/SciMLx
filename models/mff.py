"""Multi-Fidelity Fusion (MFF) for combining low-fi and high-fi data."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMFF(nn.Module):
    """
    Residual Multi-Fidelity Fusion.
    Learns to correct a low-fidelity prediction to match high-fidelity data.
    """
    def __init__(self, low_fi_model: nn.Module, hidden_dim: int = 64):
        super().__init__()
        self.low_fi_model = low_fi_model
        # Freeze low-fi model if it's already pre-trained
        for param in self.low_fi_model.parameters():
            param.requires_grad = False
            
        self.delta_net = nn.Sequential(
            nn.Linear(1 + 1, hidden_dim), # Input (x) + Low-fi output
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """x: [B, N, C]"""
        # Low-fidelity prediction
        with torch.no_grad():
            y_lo = self.low_fi_model(x) # [B, N]
            
        # Delta correction
        # We concatenate x and y_lo as features for the delta net
        # This is a simplified 1D version
        B, N = y_lo.shape
        x_flat = x[..., 0].reshape(-1, 1) # assuming first channel is spatial coord
        y_lo_flat = y_lo.reshape(-1, 1)
        
        feat = torch.cat([x_flat, y_lo_flat], dim=-1)
        delta = self.delta_net(feat).reshape(B, N)
        
        return y_lo + delta

class MultiFidelityWrapper(nn.Module):
    """
    Learns a mapping f(x, y_lo) -> y_hi.
    """
    def __init__(self, low_fi_model: nn.Module, hi_fi_net: nn.Module):
        super().__init__()
        self.low_fi_model = low_fi_model
        self.hi_fi_net = hi_fi_net
        
    def forward(self, x):
        with torch.no_grad():
            y_lo = self.low_fi_model(x)
        
        # Concatenate x and y_lo as input to the hi-fi net
        # This requires hi_fi_net to accept (in_ch + 1) channels
        y_lo_expanded = y_lo.unsqueeze(-1)
        x_hi = torch.cat([x, y_lo_expanded], dim=-1)
        return self.hi_fi_net(x_hi)
