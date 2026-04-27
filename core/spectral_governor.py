"""
Spectral Bias Governor — Framework-agnostic Spectral Bias logic.

Monitors the Fourier spectrum of residuals and suggests loss weight adjustments.
Prevents the 'Spectral Bias' where models fail to learn high-frequency details.
Supports both PyTorch and MLX backends.
"""

import numpy as np
from typing import Any, Optional, Union
from core.device import FRAMEWORK

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

class SpectralBiasGovernor:
    """
    Monitors the Fourier spectrum of residuals and suggests loss weight adjustments.
    Prevents the 'Spectral Bias' where models fail to learn high-frequency details.
    """
    def __init__(self, n_modes: int = 32, update_interval: int = 50):
        self.n_modes = n_modes
        self.update_interval = update_interval
        self.current_weights = None
        self._step_count = 0

    def _update_torch(self, pred: Any, target: Any) -> Any:
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed but requested in SpectralBiasGovernor.")
        
        import torch
        with torch.no_grad():
            residual = pred - target
            if residual.ndim == 2:
                # 1D signal: (Batch, N)
                res_ft = torch.fft.rfft(residual, dim=1).abs().mean(dim=0)
                norm_res = res_ft / (res_ft.mean() + 1e-8)
                self.current_weights = 1.0 + torch.clamp(norm_res - 1.0, min=0.0)
            elif residual.ndim == 3:
                # 2D signal: (Batch, H, W)
                res_ft = torch.fft.rfft2(residual, dim=(1, 2)).abs().mean(dim=0)
                norm_res = res_ft / (res_ft.mean() + 1e-8)
                self.current_weights = 1.0 + torch.clamp(norm_res - 1.0, min=0.0)
        return self.current_weights

    def _update_mlx(self, pred: Any, target: Any) -> Any:
        if not HAS_MLX:
            raise ImportError("MLX not installed but requested in SpectralBiasGovernor.")
        
        import mlx.core as mx
        residual = pred - target
        if residual.ndim == 2:
            # 1D signal: (Batch, N)
            res_ft = mx.abs(mx.fft.rfft(residual, axis=1)).mean(axis=0)
            norm_res = res_ft / (res_ft.mean() + 1e-8)
            self.current_weights = 1.0 + mx.maximum(norm_res - 1.0, 0.0)
        elif residual.ndim == 3:
            # 2D signal: (Batch, H, W)
            res_ft = mx.abs(mx.fft.rfft2(residual, axes=(1, 2))).mean(axis=0)
            norm_res = res_ft / (res_ft.mean() + 1e-8)
            self.current_weights = 1.0 + mx.maximum(norm_res - 1.0, 0.0)
        return self.current_weights

    def update(self, pred: Any, target: Any) -> Optional[Any]:
        """
        Update the spectral weights based on current residuals.
        Returns the computed weights or the cached weights if update interval not reached.
        """
        self._step_count += 1
        if self._step_count % self.update_interval != 0 and self.current_weights is not None:
            return self.current_weights
        
        if FRAMEWORK == "mlx":
            return self._update_mlx(pred, target)
        else:
            return self._update_torch(pred, target)

    def reset(self):
        """Reset the internal step counter and weights."""
        self._step_count = 0
        self.current_weights = None
