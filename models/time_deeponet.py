"""Time-Marching DeepONet (TimeDeepONet) - PyTorch/CUDA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from core.device import DEVICE

def _mlp(dims: list[int], act=F.gelu) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)

class TimeDeepONet1d(nn.Module):
    def __init__(self, n_sensors: int, hidden_dim: int = 64, p: int = 64,
                 n_layers: int = 4, time_embed_dim: int = 16):
        super().__init__()
        self.p = p
        self.n_sensors = n_sensors
        self.time_emb_dim = time_embed_dim

        self.branch1 = _mlp([n_sensors] + [hidden_dim] * n_layers + [p])
        self.branch2 = _mlp([time_embed_dim] + [hidden_dim] * n_layers + [p])
        self.trunk   = _mlp([1] + [hidden_dim] * n_layers + [p])
        self.bias    = nn.Parameter(torch.zeros(1))

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        d = self.time_emb_dim
        half = d // 2
        freqs = torch.exp(-torch.arange(half, device=t.device).float() * (math.log(10000) / max(half - 1, 1)))
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, u0: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        B, N = u0.shape
        if t is None:
            t = torch.ones(B, device=u0.device)
        
        b1 = self.branch1(u0)
        b2 = self.branch2(self._time_embedding(t))
        b12 = b1 * b2

        grid = torch.linspace(0.0, 1.0, N, device=u0.device).unsqueeze(-1)
        T = self.trunk(grid)
        return torch.matmul(b12, T.t()) + self.bias

class DualBranchDeepONet1d(nn.Module):
    def __init__(self, n_sensors: int, hidden_dim: int = 64, p: int = 64, n_layers: int = 4):
        super().__init__()
        self.p = p
        self.branch1 = _mlp([n_sensors] + [hidden_dim] * n_layers + [p])
        self.branch2 = _mlp([n_sensors] + [hidden_dim] * n_layers + [p])
        self.trunk   = _mlp([1] + [hidden_dim] * n_layers + [p])
        self.bias    = nn.Parameter(torch.zeros(1))

    def forward(self, u0: torch.Tensor, v0: torch.Tensor = None) -> torch.Tensor:
        B, N = u0.shape
        if v0 is None:
            v0 = torch.zeros_like(u0)
        
        b1 = self.branch1(u0)
        b2 = self.branch2(v0)
        b12 = b1 + b2

        grid = torch.linspace(0.0, 1.0, N, device=u0.device).unsqueeze(-1)
        T = self.trunk(grid)
        return torch.matmul(b12, T.t()) + self.bias
