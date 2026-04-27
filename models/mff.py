"""
Multi-Fidelity Fusion (MFF) Dispatcher.
Supports Torch and MLX backends.
"""

from typing import Any, Optional

def MultiFidelityFusion(low_fi_model: Any, hi_fi_net: Any, backend: str = 'torch', **kwargs):
    """
    Factory function for MultiFidelityFusion.
    """
    if backend == 'torch':
        from .mff_torch import MultiFidelityFusion as MFFTorch
        return MFFTorch(low_fi_model, hi_fi_net, **kwargs)
    elif backend == 'mlx':
        from .mff_mlx import MultiFidelityFusion as MFFMLX
        return MFFMLX(low_fi_model, hi_fi_net, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

def ResidualMFF(low_fi_model: Any, delta_net: Any, backend: str = 'torch', **kwargs):
    """
    Factory function for ResidualMFF.
    """
    if backend == 'torch':
        from .mff_torch import ResidualMFF as RMFFTorch
        return RMFFTorch(low_fi_model, delta_net, **kwargs)
    elif backend == 'mlx':
        from .mff_mlx import ResidualMFF as RMFFMLX
        return RMFFMLX(low_fi_model, delta_net, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
