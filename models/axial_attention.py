import torch
import torch.nn as nn

class AxialAttention2d(nn.Module):
    """
    Axial Attention for 2D grids (PyTorch/CUDA). 
    Performs attention along height, then along width.
    Memory complexity: O(B * H * W * (H + W)) instead of O(B * (H*W)^2).
    """
    def __init__(self, dims: int, num_heads: int = 4):
        super().__init__()
        self.attn_h = nn.MultiheadAttention(dims, num_heads, batch_first=True)
        self.attn_w = nn.MultiheadAttention(dims, num_heads, batch_first=True)
        self.ln_h = nn.LayerNorm(dims)
        self.ln_w = nn.LayerNorm(dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, H, W, C]
        B, H, W, C = x.shape

        # 1. Height-wise attention
        h = self.ln_h(x)
        h = h.permute(0, 2, 1, 3).reshape(-1, H, C)
        h, _ = self.attn_h(h, h, h)
        h = h.reshape(B, W, H, C).permute(0, 2, 1, 3)
        x = x + h

        # 2. Width-wise attention
        w = self.ln_w(x)
        w = w.reshape(-1, W, C)
        w, _ = self.attn_w(w, w, w)
        w = w.reshape(B, H, W, C)
        x = x + w

        return x
