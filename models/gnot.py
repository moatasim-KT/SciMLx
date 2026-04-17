"""General Neural Operator Transformer (GNOT) for SciML benchmarks.

Simplified implementation for 1-D and 2-D regular grids using self-attention
over spatial tokens, with coordinate-based positional encoding.

Reference:
  Hao et al. (2023) "GNOT: A General Neural Operator Transformer" (ICML 2023)
  arXiv:2302.14376
"""

import math
import mlx.core as mx
import mlx.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dims, dims)
        self.key_proj = nn.Linear(dims, dims)
        self.value_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)
        self.scale = (dims // num_heads) ** -0.5

    def __call__(self, queries, keys, values, mask=None):
        B, L, D = queries.shape
        _, S, _ = keys.shape
        H = self.num_heads
        d = D // H

        queries = self.query_proj(queries).reshape(B, L, H, d).transpose(0, 2, 1, 3)
        keys = self.key_proj(keys).reshape(B, S, H, d).transpose(0, 2, 1, 3)
        values = self.value_proj(values).reshape(B, S, H, d).transpose(0, 2, 1, 3)

        # scores: [B, H, L, S]
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ values).transpose(0, 2, 1, 3).reshape(B, L, D)
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

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GNOT1d(nn.Module):
    """Simplified GNOT for 1-D problems."""
    
    def __init__(self, hidden_dim: int, n_layers: int, n_heads: int = 4, in_channels: int = 1):
        super().__init__()
        self.lift = nn.Linear(in_channels + 1, hidden_dim)
        self.blocks = [TransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, in_channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x : [B, N] or [B, N, C]
        if x.ndim == 2:
            B, N = x.shape
            x = x[..., None]
        else:
            B, N, _ = x.shape
            
        grid = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N, 1), (B, N, 1))
        x = mx.concatenate([x, grid], axis=-1) # [B, N, C+1]
        
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        
        x = nn.gelu(self.proj1(self.norm(x)))
        out = self.proj2(x)
        if out.shape[-1] == 1:
            return out[:, :, 0]
        return out

class GNOT2d(nn.Module):
    """GNOT for 2-D problems using axial attention.

    Uses row-then-column attention (O(N³) memory) instead of full quadratic
    attention (O(N⁴)) so it fits in M1 8GB for 64×64 grids.
    Internally identical to GNOT_Axial2d — the registry key GNOT2D/GNOT2d
    maps here for backward compatibility with experiments.yaml entries.
    """

    def __init__(self, hidden_dim: int, n_layers: int, n_heads: int = 4, in_channels: int = 1):
        super().__init__()
        self.lift = nn.Linear(in_channels + 2, hidden_dim)
        self.blocks = [AxialTransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, in_channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x : [B, N1, N2] or [B, N1, N2, C]
        if x.ndim == 3:
            B, N1, N2 = x.shape
            x = x[..., None]
        else:
            B, N1, N2, _ = x.shape

        grid1 = mx.broadcast_to(mx.linspace(0.0, 1.0, N1).reshape(1, N1, 1, 1), (B, N1, N2, 1))
        grid2 = mx.broadcast_to(mx.linspace(0.0, 1.0, N2).reshape(1, 1, N2, 1), (B, N1, N2, 1))
        x = mx.concatenate([x, grid1, grid2], axis=-1)  # [B, N1, N2, C+2]

        # Lift to hidden dim; AxialTransformerBlock expects [B, H, W, C]
        B2, H, W, Cin = x.shape
        x = self.lift(x.reshape(B2 * H * W, Cin)).reshape(B2, H, W, -1)

        for blk in self.blocks:
            x = blk(x)  # [B, N1, N2, hidden_dim]

        # Project back to output channels
        x = nn.gelu(self.proj1(self.norm(x)))
        out = self.proj2(x)  # [B, N1, N2, in_channels]

        if out.shape[-1] == 1:
            return out[:, :, :, 0]
        return out

class GNOT_FFNO_Block(nn.Module):
    """Hybrid Spectral-Attention block for sharp gradient capture.
    
    Combines Transformer spatial attention with Factorized Fourier spectral filtering.
    """
    def __init__(self, dims: int, n_modes: int, n_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        from models.afno import DiagSpectralConv1d
        self.ln1 = nn.LayerNorm(dims)
        self.attn = MultiHeadAttention(dims, n_heads)
        self.spec = DiagSpectralConv1d(dims, n_modes)
        
        # Learnable gate: linear projection from hidden state to scalar blend weight.
        # nn.Linear is tracked by nn.Module so gradients flow through it.
        self.gate = nn.Linear(dims, 1, bias=True)

        self.ln2 = nn.LayerNorm(dims)
        self.mlp = nn.Sequential(
            nn.Linear(dims, mlp_ratio * dims),
            nn.GELU(),
            nn.Linear(mlp_ratio * dims, dims),
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.ln1(x)
        # Spatial path
        x_attn = self.attn(h, h, h)
        # Spectral path
        x_spec = self.spec(h)

        # Gated fusion: gate weight derived from hidden state, broadcast over dim
        g = mx.sigmoid(self.gate(h))  # [B, N, 1] — trainable blend weight
        x = x + g * x_spec + (1 - g) * x_attn
        
        x = x + self.mlp(self.ln2(x))
        return x

class GNOT_FFNO(nn.Module):
    """Burgers Breakthrough Hybrid Model (Phase 8)."""
    def __init__(self, hidden_dim: int, n_layers: int, n_modes: int = 24, n_heads: int = 4, in_channels: int = 1):
        super().__init__()
        self.lift = nn.Linear(in_channels + 1, hidden_dim)
        self.blocks = [GNOT_FFNO_Block(hidden_dim, n_modes, n_heads) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, in_channels)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 2:
            B, N = x.shape
            x = x[..., None]
        else:
            B, N, _ = x.shape
            
        grid = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N, 1), (B, N, 1))
        x = mx.concatenate([x, grid], axis=-1)
        
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        
        x = nn.gelu(self.proj1(self.norm(x)))
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

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, H, W, C]
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GNOT_Axial2d(nn.Module):
    """Memory-efficient 2D GNOT using Axial Attention."""
    def __init__(self, hidden_dim: int, n_layers: int, n_heads: int = 4, in_channels: int = 1):
        super().__init__()
        self.lift = nn.Linear(in_channels + 2, hidden_dim)
        self.blocks = [AxialTransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, in_channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x : [B, N1, N2] or [B, N1, N2, C]
        if x.ndim == 3:
            B, N1, N2 = x.shape
            x = x[..., None]
        else:
            B, N1, N2, _ = x.shape
            
        grid1 = mx.broadcast_to(mx.linspace(0.0, 1.0, N1).reshape(1, N1, 1, 1), (B, N1, N2, 1))
        grid2 = mx.broadcast_to(mx.linspace(0.0, 1.0, N2).reshape(1, 1, N2, 1), (B, N1, N2, 1))
        x     = mx.concatenate([x, grid1, grid2], axis=-1)  
        
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        
        x = nn.gelu(self.proj1(self.norm(x)))
        out = self.proj2(x)
        
        if out.shape[-1] == 1:
            return out[:, :, :, 0]
        return out
