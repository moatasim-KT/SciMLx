import pytest
import numpy as np
from core.spectral_governor import SpectralBiasGovernor
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

def test_spectral_bias_governor_init():
    governor = SpectralBiasGovernor(n_modes=16, update_interval=10)
    assert governor.n_modes == 16
    assert governor.update_interval == 10
    assert governor._step_count == 0
    assert governor.current_weights is None

def test_spectral_bias_governor_update():
    governor = SpectralBiasGovernor(n_modes=16, update_interval=1)
    
    if FRAMEWORK == "torch" and HAS_TORCH:
        pred = torch.randn(4, 64)
        target = torch.randn(4, 64)
        weights = governor.update(pred, target)
        assert weights is not None
        assert weights.shape == (33,) # rfft of 64 is 33
        assert (weights >= 1.0).all()
        
    elif FRAMEWORK == "mlx" and HAS_MLX:
        pred = mx.random.normal((4, 64))
        target = mx.random.normal((4, 64))
        weights = governor.update(pred, target)
        assert weights is not None
        assert weights.shape == (33,)
        assert (weights >= 1.0).all()

def test_spectral_bias_governor_interval():
    governor = SpectralBiasGovernor(n_modes=16, update_interval=2)
    
    if FRAMEWORK == "torch" and HAS_TORCH:
        pred = torch.randn(4, 64)
        target = torch.randn(4, 64)
        
        # Step 1: should return None or previous weights (None initially)
        w1 = governor.update(pred, target)
        assert w1 is None
        
        # Step 2: should update
        w2 = governor.update(pred, target)
        assert w2 is not None
        
    elif FRAMEWORK == "mlx" and HAS_MLX:
        pred = mx.random.normal((4, 64))
        target = mx.random.normal((4, 64))
        
        w1 = governor.update(pred, target)
        assert w1 is None
        
        w2 = governor.update(pred, target)
        assert w2 is not None

def test_spectral_bias_governor_reset():
    governor = SpectralBiasGovernor(n_modes=16, update_interval=1)
    governor._step_count = 10
    governor.current_weights = np.ones(10)
    governor.reset()
    assert governor._step_count == 0
    assert governor.current_weights is None
