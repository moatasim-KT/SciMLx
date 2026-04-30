"""Equivariant Layers for Geometric Deep Learning in SciML."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SO2EquivariantConv2d(nn.Module):
    """
    A simple SO(2) Equivariant Convolutional layer.
    Ensures that rotating the input results in a rotated output.
    This implementation uses a restricted weight space or circular symmetry.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # In a truly SO(2) equivariant conv, weights should be radially symmetric
        # or we use multiple orientations. For now, we'll implement a 
        # Rotational Group Convolution (G-Conv) for 4 rotations (0, 90, 180, 270).
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        """x: [B, C, H, W]"""
        # Apply convolution with 4 rotations and average (or max) to get invariance,
        # or transform weights to get equivariance.
        
        # Simple G-Conv approach:
        out = 0
        for k in range(4):
            # Rotate input
            x_rot = torch.rot90(x, k, dims=(2, 3))
            # Convolve
            res = F.conv2d(x_rot, self.weight, self.bias, padding=self.kernel_size//2)
            # Rotate back
            out += torch.rot90(res, -k, dims=(2, 3))
            
        return out / 4.0

class ENEquivariantLayer(nn.Module):
    """Placeholder for E(n) Equivariant Graph Layer (EGNN style)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1))
        self.node_mlp = nn.Sequential(nn.Linear(in_dim + 1, 16), nn.SiLU(), nn.Linear(16, out_dim))

    def forward(self, h, x):
        """
        h: node features [B, N, D]
        x: node coordinates [B, N, 3]
        """
        # Simple relative distance based message passing
        dist = torch.norm(x.unsqueeze(2) - x.unsqueeze(1), dim=-1, keepdim=True) # [B, N, N, 1]
        edge_feat = self.edge_mlp(dist) # [B, N, N, 1]
        
        m = torch.sum(edge_feat * h.unsqueeze(1), dim=2) # [B, N, D]
        h_new = self.node_mlp(torch.cat([h, m.mean(dim=1, keepdim=True).expand(-1, h.size(1), -1)], dim=-1))
        
        return h_new, x
