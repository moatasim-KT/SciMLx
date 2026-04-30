"""Wavelet Neural Operator (WNO) for 1-D and 2-D problems - PyTorch/CUDA."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Haar Wavelet Transform ────────────────────────────────────────────────────

def _haar_forward(x: torch.Tensor, levels: int):
    details = []
    approx = x
    sq2 = math.sqrt(2)
    for _ in range(levels):
        B, N, C = approx.shape
        pairs = approx.view(B, N // 2, 2, C)
        lo = (pairs[:, :, 0, :] + pairs[:, :, 1, :]) / sq2
        hi = (pairs[:, :, 0, :] - pairs[:, :, 1, :]) / sq2
        details.append(hi)
        approx = lo
    return details, approx

def _haar_inverse(details: list, approx: torch.Tensor):
    x = approx
    sq2 = math.sqrt(2)
    for hi in reversed(details):
        lo = x
        even = (lo + hi) / sq2
        odd  = (lo - hi) / sq2
        B, N2, C = lo.shape
        x = torch.stack([even, odd], dim=2).view(B, N2 * 2, C)
    return x

class WaveletConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_levels: int = 3):
        super().__init__()
        self.n_levels = n_levels
        self.W = nn.ModuleList([nn.Linear(in_ch, out_ch) for _ in range(n_levels + 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        details, approx = _haar_forward(x, self.n_levels)
        new_details = [self.W[k](d) for k, d in enumerate(details)]
        new_approx  = self.W[-1](approx)
        return _haar_inverse(new_details, new_approx)

class WNOBlock1d(nn.Module):
    def __init__(self, channels: int, n_levels: int = 3):
        super().__init__()
        self.wav = WaveletConv1d(channels, channels, n_levels)
        self.w   = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.wav(x) + self.w(x))

class WNO1d(nn.Module):
    def __init__(self, n_levels: int = 3, hidden_dim: int = 32, n_layers: int = 4, in_ch: int = 2):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([WNOBlock1d(hidden_dim, n_levels) for _ in range(n_layers)])
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N = u0.shape
        grid = torch.linspace(0.0, 1.0, N, device=u0.device).view(1, N).expand(B, N)
        x = torch.stack([u0, grid], dim=-1)
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.proj1(x))
        return self.proj2(x).squeeze(-1)

# ── 2D Wavelet Components ───────────────────────────────────────────────────

def _haar2d_forward(x: torch.Tensor, levels: int):
    details = []
    approx = x
    sq2 = math.sqrt(2)
    for _ in range(levels):
        B, H, W, C = approx.shape
        rows = approx.view(B, H // 2, 2, W, C)
        L = (rows[:, :, 0, :, :] + rows[:, :, 1, :, :]) / sq2
        H_sub = (rows[:, :, 0, :, :] - rows[:, :, 1, :, :]) / sq2
        
        cols_L = L.reshape(B, H // 2, W // 2, 2, C)
        LL = (cols_L[:, :, :, 0, :] + cols_L[:, :, :, 1, :]) / 2.0
        LH = (cols_L[:, :, :, 0, :] - cols_L[:, :, :, 1, :]) / 2.0
        
        cols_H = H_sub.reshape(B, H // 2, W // 2, 2, C)
        HL = (cols_H[:, :, :, 0, :] + cols_H[:, :, :, 1, :]) / 2.0
        HH = (cols_H[:, :, :, 0, :] - cols_H[:, :, :, 1, :]) / 2.0
        
        details.append((LH, HL, HH))
        approx = LL
    return details, approx

def _haar2d_inverse(details, approx):
    x = approx
    for (LH, HL, HH) in reversed(details):
        L_even = (x + LH)
        L_odd  = (x - LH)
        L = torch.stack([L_even, L_odd], dim=3).view(x.shape[0], x.shape[1], -1, x.shape[3])
        
        H_even = (HL + HH)
        H_odd  = (HL - HH)
        H_sub = torch.stack([H_even, H_odd], dim=3).view(x.shape[0], x.shape[1], -1, x.shape[3])
        
        approx_even = (L + H_sub)
        approx_odd  = (L - H_sub)
        x = torch.stack([approx_even, approx_odd], dim=2).view(x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3])
    return x

class WaveletConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_levels: int = 3):
        super().__init__()
        self.n_levels = n_levels
        self.W = nn.ModuleList([nn.Linear(in_ch, out_ch) for _ in range(3 * n_levels + 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        details, approx = _haar2d_forward(x, self.n_levels)
        new_details = []
        idx = 0
        for LH, HL, HH in details:
            new_details.append((self.W[idx](LH), self.W[idx+1](HL), self.W[idx+2](HH)))
            idx += 3
        new_approx = self.W[-1](approx)
        return _haar2d_inverse(new_details, new_approx)

class WNO2d(nn.Module):
    def __init__(self, n_levels: int = 3, hidden_dim: int = 32, n_layers: int = 4, in_channels: int = 1):
        super().__init__()
        self.lift = nn.Linear(in_channels + 2, hidden_dim)
        self.blocks = nn.ModuleList([nn.Sequential(WaveletConv2d(hidden_dim, hidden_dim, n_levels), nn.GELU()) 
                        for _ in range(n_layers)])
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            B, N1, N2 = x.shape
            x = x.unsqueeze(-1)
        else:
            B, N1, N2, _ = x.shape
            
        grid1 = torch.linspace(0.0, 1.0, N1, device=x.device).view(1, N1, 1, 1).expand(B, N1, N2, 1)
        grid2 = torch.linspace(0.0, 1.0, N2, device=x.device).view(1, 1, N2, 1).expand(B, N1, N2, 1)
        x = torch.cat([x, grid1, grid2], dim=-1)
        
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.proj1(x))
        out = self.proj2(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out

class WNO_GNOT(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int, n_levels: int = 3, n_heads: int = 4, in_channels: int = 1):
        super().__init__()
        # WNO_GNOT logic needs MultiHeadAttention from gnot
        from models.gnot import GNOT1d
        # Simplified factory for brevity, should use ported gnot components
        self.lift = nn.Linear(in_channels + 1, hidden_dim)
        # Placeholder for hybrid logic
        self.blocks = nn.ModuleList([WNOBlock1d(hidden_dim, n_levels) for _ in range(n_layers)])
        self.proj = nn.Linear(hidden_dim, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        B, N, C = x.shape
        grid = torch.linspace(0, 1, N, device=x.device).view(1, N, 1).expand(B, N, 1)
        x = torch.cat([x, grid], dim=-1)
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        out = self.proj(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
