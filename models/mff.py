"""Multi-Fidelity Fusion (MFF) for combining low-fi and high-fi data."""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import mlx.core as mx
    import mlx.nn as mnn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

class MultiFidelityFusionTorch(nn.Module) if HAS_TORCH else object:
    """
    Torch implementation of Multi-Fidelity Fusion.
    Learns to enhance a low-fidelity prediction using a high-fidelity network.
    """
    def __init__(self, low_fi_model: nn.Module, hi_fi_net: nn.Module):
        if not HAS_TORCH:
            raise ImportError("Torch is not installed.")
        super().__init__()
        self.low_fi_model = low_fi_model
        # Typically low-fi model is frozen
        for param in self.low_fi_model.parameters():
            param.requires_grad = False
        self.hi_fi_net = hi_fi_net
        
    def forward(self, x):
        """
        x: [B, N, C] input features
        """
        with torch.no_grad():
            y_lo = self.low_fi_model(x)
            
        # Ensure y_lo has same number of dims as x to concatenate
        if y_lo.ndim == x.ndim - 1:
            y_lo = y_lo.unsqueeze(-1)
            
        # Concatenate original features and low-fidelity prediction
        x_hi = torch.cat([x, y_lo], dim=-1)
        return self.hi_fi_net(x_hi)

class MultiFidelityFusionMLX(mnn.Module) if HAS_MLX else object:
    """
    MLX implementation of Multi-Fidelity Fusion.
    """
    def __init__(self, low_fi_model, hi_fi_net):
        if not HAS_MLX:
            raise ImportError("MLX is not installed.")
        super().__init__()
        self.low_fi_model = low_fi_model
        self.hi_fi_net = hi_fi_net
        
    def __call__(self, x):
        """
        x: [B, N, C]
        """
        # MLX stop_gradient is equivalent to torch.no_grad() for specific variables
        y_lo = mx.stop_gradient(self.low_fi_model(x))
        
        if y_lo.ndim == x.ndim - 1:
            y_lo = mx.expand_dims(y_lo, -1)
            
        x_hi = mx.concatenate([x, y_lo], axis=-1)
        return self.hi_fi_net(x_hi)

class ResidualMFFTorch(nn.Module) if HAS_TORCH else object:
    """
    Torch implementation of Residual Multi-Fidelity Fusion.
    Learns a correction delta: y_hi = y_lo + delta(x, y_lo)
    """
    def __init__(self, low_fi_model: nn.Module, delta_net: nn.Module):
        if not HAS_TORCH:
            raise ImportError("Torch is not installed.")
        super().__init__()
        self.low_fi_model = low_fi_model
        for param in self.low_fi_model.parameters():
            param.requires_grad = False
        self.delta_net = delta_net
        
    def forward(self, x):
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
