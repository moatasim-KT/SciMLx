"""Differential Privacy (DP) for Federated SciML."""

import torch
from typing import List, Optional

class DPSolver:
    """
    Differential Privacy Solver for federated learning.
    Implements gradient clipping and noise addition.
    """
    def __init__(self, 
                 max_grad_norm: float = 1.0, 
                 noise_multiplier: float = 0.1, 
                 batch_size: int = 32):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size

    def clip_gradients(self, parameters: List[torch.Tensor]):
        """
        Clip gradients of parameters to max_grad_norm.
        """
        torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)

    def add_noise(self, parameters: List[torch.Tensor], device: torch.device):
        """
        Add Gaussian noise to gradients.
        """
        sigma = self.noise_multiplier * self.max_grad_norm
        for param in parameters:
            if param.grad is not None:
                noise = torch.randn(param.grad.shape, device=device) * sigma
                param.grad.add_(noise)

    def step(self, optimizer: torch.optim.Optimizer, parameters: List[torch.Tensor]):
        """
        Perform a DP-protected optimization step.
        """
        self.clip_gradients(parameters)
        device = parameters[0].device if parameters else torch.device('cpu')
        self.add_noise(parameters, device)
        optimizer.step()

class GaussianDP:
    """
    Utility for Gaussian noise mechanism.
    """
    @staticmethod
    def compute_noise_scale(epsilon: float, delta: float, sensitivity: float) -> float:
        """
        Compute noise scale sigma for (epsilon, delta)-DP.
        Simplified version.
        """
        import numpy as np
        return sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    @staticmethod
    def add_noise_to_tensor(tensor: torch.Tensor, epsilon: float, delta: float, sensitivity: float) -> torch.Tensor:
        sigma = GaussianDP.compute_noise_scale(epsilon, delta, sensitivity)
        return tensor + torch.randn_like(tensor) * sigma
