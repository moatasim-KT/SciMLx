import mlx.core as mx
import mlx.nn as nn

class AxialAttention2d(nn.Module):
    """
    Axial Attention for 2D grids. 
    Performs attention along height, then along width.
    Memory complexity: O(B * H * W * (H + W)) instead of O(B * (H*W)^2).
    """
    def __init__(self, dims: int, num_heads: int = 4):
        super().__init__()
        self.attn_h = nn.MultiHeadAttention(dims, num_heads)
        self.attn_w = nn.MultiHeadAttention(dims, num_heads)
        self.ln_h = nn.LayerNorm(dims)
        self.ln_w = nn.LayerNorm(dims)

    def __call__(self, x: mx.array) -> mx.array:
        # x is [B, H, W, C]
        B, H, W, C = x.shape

        # 1. Height-wise attention
        # Reshape: [B, W, H, C] -> [B*W, H, C]
        h = self.ln_h(x)
        h = h.transpose(0, 2, 1, 3).reshape(-1, H, C)
        h = self.attn_h(h, h, h)
        # Reshape back: [B*W, H, C] -> [B, W, H, C] -> [B, H, W, C]
        h = h.reshape(B, W, H, C).transpose(0, 2, 1, 3)
        x = x + h

        # 2. Width-wise attention
        # Reshape: [B, H, W, C] -> [B*H, W, C]
        w = self.ln_w(x)
        w = w.reshape(-1, W, C)
        w = self.attn_w(w, w, w)
        # Reshape back: [B*H, W, C] -> [B, H, W, C]
        w = w.reshape(B, H, W, C)
        x = x + w

        return x
