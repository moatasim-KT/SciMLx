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
    
    def __init__(self, hidden_dim: int, n_layers: int, n_heads: int = 4, in_ch: int = 2):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        self.blocks = [TransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, u0: mx.array) -> mx.array:
        B, N = u0.shape
        grid = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N), (B, N))
        x = mx.stack([u0, grid], axis=-1) # [B, N, 2]
        
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        
        x = nn.gelu(self.proj1(self.norm(x)))
        return self.proj2(x)[:, :, 0]

class GNOT2d(nn.Module):
    """Simplified GNOT for 2-D problems."""
    
    def __init__(self, hidden_dim: int, n_layers: int, n_heads: int = 4, in_ch: int = 3):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        self.blocks = [TransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, u0: mx.array) -> mx.array:
        B, N1, N2 = u0.shape
        grid1 = mx.broadcast_to(mx.linspace(0.0, 1.0, N1).reshape(1, N1, 1), (B, N1, N2))
        grid2 = mx.broadcast_to(mx.linspace(0.0, 1.0, N2).reshape(1, 1, N2), (B, N1, N2))
        x = mx.stack([u0, grid1, grid2], axis=-1) # [B, N1, N2, 3]
        
        # Flatten spatial dims to tokens
        x = x.reshape(B, N1 * N2, 3)
        
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        
        x = nn.gelu(self.proj1(self.norm(x)))
        out = self.proj2(x) # [B, N1*N2, 1]
        return out.reshape(B, N1, N2)
