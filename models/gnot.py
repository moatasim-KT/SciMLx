"""General Neural Operator Transformer (GNOT) for SciML benchmarks.
# einops used for multi-head reshape ops — clearer than manual reshape+transpose.

Simplified implementation for 1-D and 2-D regular grids using self-attention
over spatial tokens, with coordinate-based positional encoding.

Reference:
  Hao et al. (2023) "GNOT: A General Neural Operator Transformer" (ICML 2023)
  arXiv:2302.14376
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from core.device import DEVICE

class MultiHeadAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dims, dims)
        self.key_proj = nn.Linear(dims, dims)
        self.value_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)
        self.scale = (dims // num_heads) ** -0.5

    def forward(self, queries, keys, values, mask=None):
        B, L, D = queries.shape
        _, S, _ = keys.shape
        H = self.num_heads
        d = D // H

        queries = rearrange(self.query_proj(queries), 'b l (h d) -> b h l d', h=H)
        keys    = rearrange(self.key_proj(keys),      'b s (h d) -> b h s d', h=H)
        values  = rearrange(self.value_proj(values),  'b s (h d) -> b h s d', h=H)

        # PyTorch matmul handles batch/head dimensions correctly
        scores = (queries @ keys.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        out = rearrange(attn @ values, 'b h l d -> b l (h d)')
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dims: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(dims)
        self.attn = MultiHeadAttention(dims, num_heads)
        self.ln2 = nn.LayerNorm(dims)
        self.mlp = nn.Sequential(
            nn.Linear(dims, mlp_ratio * dims),
            nn.GELU(),
            nn.Linear(mlp_ratio * dims, dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GNOT1d(nn.Module):
    """Simplified GNOT for 1-D problems."""
    
    def __init__(self, hidden_dim: int, n_layers: int, n_heads: int = 4, in_channels: int = 1):
        super().__init__()
        self.lift = nn.Linear(in_channels + 1, hidden_dim)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, N] or [B, N, C]
        if x.ndim == 2:
            B, N = x.shape
            x = x[..., None]
        else:
            B, N, _ = x.shape
            
        grid = torch.linspace(0.0, 1.0, N, device=x.device).reshape(1, N, 1).expand(B, N, 1)
        x = torch.cat([x, grid], dim=-1) # [B, N, C+1]
        
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        
        x = F.gelu(self.proj1(self.norm(x)))
        out = self.proj2(x)
        if out.shape[-1] == 1:
            return out[:, :, 0]
        return out

class GNOT2d(nn.Module):
    """GNOT for 2-D problems using axial attention."""

    def __init__(self, hidden_dim: int, n_layers: int, n_heads: int = 4, in_channels: int = 1):
        super().__init__()
        self.lift = nn.Linear(in_channels + 2, hidden_dim)
        self.blocks = nn.ModuleList([AxialTransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, N1, N2] or [B, N1, N2, C]
        if x.ndim == 3:
            B, N1, N2 = x.shape
            x = x[..., None]
        else:
            B, N1, N2, _ = x.shape

        grid1 = torch.linspace(0.0, 1.0, N1, device=x.device).reshape(1, N1, 1, 1).expand(B, N1, N2, 1)
        grid2 = torch.linspace(0.0, 1.0, N2, device=x.device).reshape(1, 1, N2, 1).expand(B, N1, N2, 1)
        x = torch.cat([x, grid1, grid2], dim=-1)  # [B, N1, N2, C+2]

        # Lift to hidden dim
        B2, H, W, Cin = x.shape
        x = self.lift(x.reshape(B2 * H * W, Cin)).reshape(B2, H, W, -1)

        for blk in self.blocks:
            x = blk(x)  # [B, N1, N2, hidden_dim]

        # Project back to output channels
        x = F.gelu(self.proj1(self.norm(x)))
        out = self.proj2(x)  # [B, N1, N2, in_channels]

        if out.shape[-1] == 1:
            return out[:, :, :, 0]
        return out

class GNOT_FFNO_Block(nn.Module):
    """Hybrid Spectral-Attention block for sharp gradient capture."""
    def __init__(self, dims: int, n_modes: int, n_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        from models.afno import DiagSpectralConv1d
        self.ln1 = nn.LayerNorm(dims)
        self.attn = MultiHeadAttention(dims, n_heads)
        self.spec = DiagSpectralConv1d(dims, n_modes)
        
        self.gate = nn.Linear(dims, 1, bias=True)

        self.ln2 = nn.LayerNorm(dims)
        self.mlp = nn.Sequential(
            nn.Linear(dims, mlp_ratio * dims),
            nn.GELU(),
            nn.Linear(mlp_ratio * dims, dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        # Spatial path
        x_attn = self.attn(h, h, h)
        # Spectral path
        x_spec = self.spec(h)

        # Gated fusion
        g = torch.sigmoid(self.gate(h))  
        x = x + g * x_spec + (1 - g) * x_attn
        
        x = x + self.mlp(self.ln2(x))
        return x

class GNOT_FFNO(nn.Module):
    """Burgers Breakthrough Hybrid Model (Phase 8)."""
    def __init__(self, hidden_dim: int, n_layers: int, n_modes: int = 24, n_heads: int = 4, in_channels: int = 1):
        super().__init__()
        self.lift = nn.Linear(in_channels + 1, hidden_dim)
        self.blocks = nn.ModuleList([GNOT_FFNO_Block(hidden_dim, n_modes, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            B, N = x.shape
            x = x[..., None]
        else:
            B, N, _ = x.shape
            
        grid = torch.linspace(0.0, 1.0, N, device=x.device).reshape(1, N, 1).expand(B, N, 1)
        x = torch.cat([x, grid], dim=-1)
        
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        
        x = F.gelu(self.proj1(self.norm(x)))
        out = self.proj2(x)
        if out.shape[-1] == 1:
            return out[:, :, 0]
        return out

class AxialTransformerBlock(nn.Module):
    """Transformer block using Axial Attention to save memory on 2D grids."""
    def __init__(self, dims: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        from models.axial_attention import AxialAttention2d
        self.attn = AxialAttention2d(dims, num_heads)
        self.ln1 = nn.LayerNorm(dims)
        self.ln2 = nn.LayerNorm(dims)
        self.mlp = nn.Sequential(
            nn.Linear(dims, mlp_ratio * dims),
            nn.GELU(),
            nn.Linear(mlp_ratio * dims, dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GNOT_Axial2d(nn.Module):
    """Memory-efficient 2D GNOT using Axial Attention."""
    def __init__(self, hidden_dim: int, n_layers: int, n_heads: int = 4, in_channels: int = 1):
        super().__init__()
        self.lift = nn.Linear(in_channels + 2, hidden_dim)
        self.blocks = nn.ModuleList([AxialTransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, N1, N2] or [B, N1, N2, C]
        if x.ndim == 3:
            B, N1, N2 = x.shape
            x = x[..., None]
        else:
            B, N1, N2, _ = x.shape
            
        grid1 = torch.linspace(0.0, 1.0, N1, device=x.device).reshape(1, N1, 1, 1).expand(B, N1, N2, 1)
        grid2 = torch.linspace(0.0, 1.0, N2, device=x.device).reshape(1, 1, N2, 1).expand(B, N1, N2, 1)
        x     = torch.cat([x, grid1, grid2], dim=-1)  
        
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        
        x = F.gelu(self.proj1(self.norm(x)))
        out = self.proj2(x)
        
        if out.shape[-1] == 1:
            return out[:, :, :, 0]
        return out
