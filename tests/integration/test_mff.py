import pytest
import numpy as np

torch = pytest.importorskip("torch")
import torch.nn as nn
from models.mff import MultiFidelityFusionTorch, ResidualMFFTorch

def test_mff_torch_fusion():
    """Test MultiFidelityFusionTorch integration."""
    low_fi = nn.Linear(10, 1)
    hi_fi = nn.Linear(11, 5) # in_ch (10) + low_fi_out (1)
    
    mff = MultiFidelityFusionTorch(low_fi, hi_fi)
    
    # Input: [Batch, Nodes, Channels]
    x = torch.randn(2, 100, 10)
    out = mff(x)
    
    assert out.shape == (2, 100, 5)
    # Check that low-fi model is frozen
    for param in low_fi.parameters():
        assert not param.requires_grad

def test_mff_torch_residual():
    """Test ResidualMFFTorch integration."""
    low_fi = nn.Linear(10, 1)
    delta_net = nn.Linear(11, 1)
    
    mff = ResidualMFFTorch(low_fi, delta_net)
    
    x = torch.randn(2, 50, 10)
    out = mff(x)
    
    assert out.shape == (2, 50, 1)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLX tests usually run on Apple Silicon; skipping MLX parity here.")
def test_mff_mlx_stub():
    # MLX tests would go here if environment supported it
    pass
