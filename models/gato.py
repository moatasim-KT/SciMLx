"""
GATO: Geometry-Aware Transformer Operator.
Uses Heat Kernel Signatures (HKS) and Geometric Attention for operator learning on meshes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from core.heat_kernels import compute_hks, compute_laplacian

class GeometricAttention(nn.Module):
    """
    Attention mechanism biased by geometric heat kernels.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, geo_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]
            geo_bias: [N, N] or [B, N, N] geometric bias (e.g., log of heat kernel)
        """
        B, N, C = x.shape
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if geo_bias is not None:
            if geo_bias.ndim == 2:
                geo_bias = geo_bias.unsqueeze(0).unsqueeze(0) # [1, 1, N, N]
            elif geo_bias.ndim == 3:
                geo_bias = geo_bias.unsqueeze(1) # [B, 1, N, N]
            attn = attn + geo_bias
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)

class GATOLayer(nn.Module):
    """GATO Layer combining GeometricAttention and MLP."""
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = GeometricAttention(hidden_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, geo_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), geo_bias)
        x = x + self.mlp(self.norm2(x))
        return x

class GATO(nn.Module):
    """
    Geometry-Aware Transformer Operator (GATO).
    """
    def __init__(self, 
                 in_ch: int, 
                 hidden_dim: int, 
                 out_ch: int, 
                 n_layers: int = 4, 
                 n_heads: int = 8, 
                 vertices: Optional[np.ndarray] = None, 
                 faces: Optional[np.ndarray] = None,
                 num_eigs: int = 100):
        super().__init__()
        
        hks_dim = 0
        
        if vertices is not None and faces is not None:
            # 1. Compute HKS for positional encoding
            hks = compute_hks(vertices, faces, num_eigenvalues=num_eigs)
            self.register_buffer("hks_pe", torch.from_numpy(hks).float())
            hks_dim = hks.shape[1]
            
            # 2. Compute Heat Kernel for attention bias (simplified as -Distance^2 or similar if full kernel is too big)
            # For this implementation, we use the Laplacian to get a local geometric bias
            L, M = compute_laplacian(vertices, faces)
            # Use a simple proximity bias from Laplacian sparsity pattern as a placeholder for "Geometric Bias"
            # In a real GATO, this would be a precomputed heat kernel matrix H = exp(tL)
            bias = torch.from_numpy(L.toarray()).float()
            self.register_buffer("geo_bias", bias)
        else:
            self.register_buffer("hks_pe", None)
            self.register_buffer("geo_bias", None)
            
        self.lift = nn.Linear(in_ch + hks_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GATOLayer(hidden_dim, n_heads) for _ in range(n_layers)
        ])
        self.proj = nn.Linear(hidden_dim, out_ch)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        if self.hks_pe is not None:
            pe = self.hks_pe.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([x, pe], dim=-1)
            
        x = self.lift(x)
        
        for layer in self.layers:
            x = layer(x, self.geo_bias)
            
        return self.proj(x)
