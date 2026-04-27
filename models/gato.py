"""
GATO: Geometry-Aware Transformer Operator.
Uses Heat Kernel Signatures (HKS) for geometric positional encoding.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from core.heat_kernels import HeatKernel

class GATO(nn.Module):
    """
    Geometry-Aware Transformer Operator (GATO).
    
    This model leverages the spectral properties of the mesh (via HKS) 
    to provide geometry-aware positional encodings to a Transformer.
    """
    def __init__(self, 
                 in_ch: int, 
                 hidden_dim: int, 
                 out_ch: int, 
                 n_layers: int = 4, 
                 n_heads: int = 8, 
                 vertices: Optional[np.ndarray] = None, 
                 faces: Optional[np.ndarray] = None,
                 hks: Optional[np.ndarray] = None,
                 num_eigs: int = 100):
        super().__init__()
        
        # 1. Geometry Encoding (HKS)
        if hks is None and (vertices is not None and faces is not None):
            # Compute HKS using the Mesh-based Heat Kernels implemented in Phase 2
            hks = HeatKernel.compute_hks(vertices, faces, num_eigs=num_eigs)
        
        if hks is not None:
            self.register_buffer("hks_pe", torch.from_numpy(hks).float())
            hks_dim = hks.shape[1]
        else:
            self.hks_pe = None
            hks_dim = 0
            
        # 2. Lifting Layer
        # Combines raw features with HKS positional encodings
        self.lift = nn.Linear(in_ch + hks_dim, hidden_dim)
        
        # 3. Transformer Backbone
        # Uses standard Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation='gelu',
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Projection Layer
        self.proj = nn.Linear(hidden_dim, out_ch)
        
    def forward(self, x: torch.Tensor, hks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, N, in_ch]
            hks: Optional HKS tensor of shape [B, N, K] or [N, K]. 
                 If not provided, uses the HKS stored during initialization.
        
        Returns:
            Output tensor of shape [B, N, out_ch]
        """
        B, N, C = x.shape
        
        # Use provided HKS or the one computed at init
        pe = hks if hks is not None else self.hks_pe
        
        if pe is not None:
            if pe.ndim == 2:
                # [N, K] -> [B, N, K]
                pe = pe.unsqueeze(0).expand(B, -1, -1)
            elif pe.ndim == 3 and pe.shape[0] == 1:
                pe = pe.expand(B, -1, -1)
            
            # Positional encoding injection via concatenation (common in operator learning)
            x = torch.cat([x, pe], dim=-1)
            
        x = self.lift(x)
        x = self.transformer(x)
        return self.proj(x)

class GATOLayer(nn.Module):
    """Single layer of GATO for modularity."""
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x
