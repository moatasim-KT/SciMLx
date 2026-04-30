"""Multi-Fidelity Fusion (MFF) — MLX Implementation."""

import mlx.core as mx
import mlx.nn as nn

class MultiFidelityFusion(nn.Module):
    """
    MLX implementation of Multi-Fidelity Fusion.
    """
    def __init__(self, low_fi_model, hi_fi_net):
        super().__init__()
        self.low_fi_model = low_fi_model
        self.hi_fi_net = hi_fi_net
        
    def __call__(self, x: mx.array) -> mx.array:
        """
        x: [B, N, C]
        """
        y_lo = mx.stop_gradient(self.low_fi_model(x))
        
        if y_lo.ndim == x.ndim - 1:
            y_lo = mx.expand_dims(y_lo, -1)
            
        x_hi = mx.concatenate([x, y_lo], axis=-1)
        return self.hi_fi_net(x_hi)

class ResidualMFF(nn.Module):
    """
    MLX implementation of Residual Multi-Fidelity Fusion.
    """
    def __init__(self, low_fi_model, delta_net):
        super().__init__()
        self.low_fi_model = low_fi_model
        self.delta_net = delta_net
        
    def __call__(self, x: mx.array) -> mx.array:
        y_lo = mx.stop_gradient(self.low_fi_model(x))
        
        if y_lo.ndim == x.ndim - 1:
            y_lo_input = mx.expand_dims(y_lo, -1)
        else:
            y_lo_input = y_lo
            
        feat = mx.concatenate([x, y_lo_input], axis=-1)
        delta = self.delta_net(feat)
        
        if delta.shape == y_lo.shape:
            return y_lo + delta
        return y_lo + delta.squeeze(-1)
