"""Multi-Fidelity Fusion (MFF) — Torch Implementation."""

import torch
import torch.nn as nn

class MultiFidelityFusion(nn.Module):
    """
    Torch implementation of Multi-Fidelity Fusion.
    Learns to enhance a low-fidelity prediction using a high-fidelity network.
    """
    def __init__(self, low_fi_model: nn.Module, hi_fi_net: nn.Module):
        super().__init__()
        self.low_fi_model = low_fi_model
        # Typically low-fi model is frozen
        for param in self.low_fi_model.parameters():
            param.requires_grad = False
        self.hi_fi_net = hi_fi_net
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C] input features
        """
        with torch.no_grad():
            y_lo = self.low_fi_model(x)
            
        if y_lo.ndim == x.ndim - 1:
            y_lo = y_lo.unsqueeze(-1)
            
        x_hi = torch.cat([x, y_lo], dim=-1)
        return self.hi_fi_net(x_hi)

class ResidualMFF(nn.Module):
    """
    Torch implementation of Residual Multi-Fidelity Fusion.
    Learns a correction delta: y_hi = y_lo + delta(x, y_lo)
    """
    def __init__(self, low_fi_model: nn.Module, delta_net: nn.Module):
        super().__init__()
        self.low_fi_model = low_fi_model
        for param in self.low_fi_model.parameters():
            param.requires_grad = False
        self.delta_net = delta_net
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y_lo = self.low_fi_model(x)
            
        if y_lo.ndim == x.ndim - 1:
            y_lo_input = y_lo.unsqueeze(-1)
        else:
            y_lo_input = y_lo
            
        feat = torch.cat([x, y_lo_input], dim=-1)
        delta = self.delta_net(feat)
        
        if delta.shape == y_lo.shape:
            return y_lo + delta
        return y_lo + delta.squeeze(-1)
