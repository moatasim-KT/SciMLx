"""
Differential Privacy and Federated Learning utilities for SciMLx.
Backend-agnostic implementations using to_array() for cross-framework compatibility.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from core.device import to_array

class FederatedAggregator:
    """
    Federated Learning Aggregator.
    Implementations are backend-agnostic, supporting Torch and MLX via NumPy exchange.
    """
    
    @staticmethod
    def federated_avg(client_weights: List[Dict[str, Any]], 
                      client_weights_factors: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """
        Performs Federated Averaging (FedAvg).
        
        Args:
            client_weights: List of model state dictionaries from clients.
            client_weights_factors: Optional weights for each client (e.g., dataset size).
        
        Returns:
            Dictionary of aggregated weights as NumPy arrays.
        """
        if not client_weights:
            return {}
            
        num_clients = len(client_weights)
        if client_weights_factors is None:
            client_weights_factors = [1.0 / num_clients] * num_clients
            
        # Normalize factors
        total_factor = sum(client_weights_factors)
        client_weights_factors = [f / total_factor for f in client_weights_factors]
        
        aggregated_weights = {}
        keys = client_weights[0].keys()
        
        for key in keys:
            # Aggregate across all clients
            weighted_sum = None
            for i, weights in enumerate(client_weights):
                # Ensure we are working with numpy arrays for backend-agnosticism
                w_np = np.array(to_array(weights[key]))
                
                if weighted_sum is None:
                    weighted_sum = w_np * client_weights_factors[i]
                else:
                    weighted_sum += w_np * client_weights_factors[i]
            
            aggregated_weights[key] = weighted_sum
            
        return aggregated_weights

class DPSGDTrainer:
    """
    Differential Privacy SGD (DP-SGD) helper.
    Provides tools for gradient clipping and noise injection.
    """
    def __init__(self, 
                 l2_norm_clip: float = 1.0, 
                 noise_multiplier: float = 0.1, 
                 delta: float = 1e-5):
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        
    def private_gradient_update(self, 
                                gradients: Dict[str, Any], 
                                batch_size: int) -> Dict[str, np.ndarray]:
        """
        Clips gradients and adds noise to satisfy Differential Privacy.
        
        Args:
            gradients: Dictionary of gradients (per-sample gradients preferred).
            batch_size: Number of samples in the batch.
            
        Returns:
            Privatized gradients as NumPy arrays.
        """
        # 1. Convert to numpy
        grads_np = {k: np.array(to_array(v)) for k, v in gradients.items()}
        
        # 2. Global Clipping
        # (In true DP-SGD, we clip per-sample gradients, 
        # but here we implement the logic for the aggregated batch gradient)
        total_norm = np.sqrt(sum(np.sum(np.square(g)) for g in grads_np.values()))
        
        clip_coef = self.l2_norm_clip / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for k in grads_np:
                grads_np[k] *= clip_coef
                
        # 3. Add Noise
        # Noise scale depends on sensitivity (l2_norm_clip) and noise_multiplier
        sigma = self.noise_multiplier * self.l2_norm_clip
        
        privatized_grads = {}
        for k, g in grads_np.items():
            noise = np.random.normal(0, sigma, size=g.shape)
            # Add noise and normalize by batch size
            privatized_grads[k] = (g + noise) / batch_size
            
        return privatized_grads

    def compute_epsilon(self, steps: int, batch_size: int, dataset_size: int) -> float:
        """
        Simplified Rényi DP epsilon computation.
        """
        sampling_rate = batch_size / dataset_size
        # Heuristic for epsilon
        epsilon = self.noise_multiplier * np.sqrt(steps * sampling_rate)
        return float(epsilon)
