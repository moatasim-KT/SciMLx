import numpy as np
import torch
from core.dp_federated import FederatedAggregator, DPSGDTrainer

def test_federated_avg_agnostic():
    """Test FedAvg with mixed backend types (represented as numpy/torch)."""
    w1 = {
        "params.0": torch.tensor([1.0, 2.0]),
        "params.1": np.array([10.0, 20.0])
    }
    w2 = {
        "params.0": np.array([3.0, 4.0]),
        "params.1": torch.tensor([30.0, 40.0])
    }
    
    agg = FederatedAggregator()
    weights = [w1, w2]
    
    aggregated = agg.federated_avg(weights)
    
    # Expected: (1+3)/2 = 2, (2+4)/2 = 3
    assert np.allclose(aggregated["params.0"], [2.0, 3.0])
    # Expected: (10+30)/2 = 20, (20+40)/2 = 30
    assert np.allclose(aggregated["params.1"], [20.0, 30.0])

def test_dp_sgd_logic():
    """Test DP-SGD privatizer (clipping and noise)."""
    trainer = DPSGDTrainer(l2_norm_clip=1.0, noise_multiplier=0.0) # No noise for testing clipping
    
    grads = {
        "w": torch.tensor([10.0, 10.0]) # Norm is ~14.14
    }
    
    # Batch size 1
    priv_grads = trainer.private_gradient_update(grads, batch_size=1)
    
    # Norm should be clipped to 1.0
    total_norm = np.linalg.norm(priv_grads["w"])
    assert np.isclose(total_norm, 1.0)
    
    # Test with noise
    trainer_noisy = DPSGDTrainer(l2_norm_clip=1.0, noise_multiplier=1.0)
    priv_grads_noisy = trainer_noisy.private_gradient_update(grads, batch_size=1)
    
    # With noise_multiplier=1.0, it's very unlikely to be exactly the same as clipped only
    assert not np.allclose(priv_grads_noisy["w"], priv_grads["w"])
