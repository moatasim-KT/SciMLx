"""Transolver: Transformer Solver with Physics Attention.
# einops used for multi-head reshape ops — clearer than manual reshape+transpose.

Adapted from PhysicsNeMo (NVIDIA Modulus) Transolver implementation:
  "Transolver: A Fast Transformer Solver for PDEs on General Geometries"
  Haixu Wu, Huakun Luo, Haowen Wang et al. — NeurIPS 2024 / ICML 2024
  arXiv: https://arxiv.org/abs/2402.02366
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from core.device import DEVICE


# ── Physics Attention (1-D structured grid) ───────────────────────────────────

class PhysicsAttn1d(nn.Module):
    """Physics Attention over 1-D structured grids.

    Parameters
    ----------
    dim : int
        Embedding dimension (must be divisible by n_head).
    n_head : int
        Number of attention heads.
    slice_num : int
        Number of physics slices S (analogous to tokens in ViT).
    """

    def __init__(self, dim: int, n_head: int = 4, slice_num: int = 32):
        super().__init__()
        assert dim % n_head == 0, "dim must be divisible by n_head"
        self.dim       = dim
        self.n_head    = n_head
        self.slice_num = slice_num
        self.head_dim  = dim // n_head
        self.scale     = self.head_dim ** -0.5

        # Slice assignment projections: N points → S slice logits (per head)
        self.to_slice  = nn.Linear(dim, n_head * slice_num, bias=False)
        # Standard QKV projections (applied on slice tokens after grouping)
        self.to_q      = nn.Linear(dim, dim, bias=False)
        self.to_k      = nn.Linear(dim, dim, bias=False)
        self.to_v      = nn.Linear(dim, dim, bias=False)
        self.out_proj  = nn.Linear(dim, dim)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, N, D]
        Returns
        -------
        out : [B, N, D]
        """
        B, N, D = x.shape
        H, S, d = self.n_head, self.slice_num, self.head_dim

        # ── Slice assignment ─────────────────────────────────────────────────
        logits = rearrange(self.to_slice(x), 'b n (h s) -> b h n s', h=H)
        A = F.softmax(logits, dim=-1)           # [B, H, N, S]

        # ── Aggregate N grid points → S slice tokens ──────────────────────
        q_grid = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=H)
        k_grid = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=H)
        v_grid = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=H)

        # Weighted average: [B,H,S,d] = A^T [B,H,N,S] @ {q,k,v} [B,H,N,d]
        At     = A.transpose(-2, -1)          # [B, H, S, N]
        q_s    = torch.matmul(At, q_grid)            # [B, H, S, d]
        k_s    = torch.matmul(At, k_grid)
        v_s    = torch.matmul(At, v_grid)

        # ── Self-attention in slice space ─────────────────────────────────
        dots   = torch.matmul(q_s, k_s.transpose(-2, -1)) * self.scale  # [B,H,S,S]
        attn   = F.softmax(dots, dim=-1)
        out_s  = torch.matmul(attn, v_s)             # [B, H, S, d]

        # ── Broadcast back: S slice tokens → N grid points ────────────────
        out_grid = torch.matmul(A, out_s)            # [B, H, N, d]
        out = rearrange(out_grid, 'b h n d -> b n (h d)')
        return self.out_proj(out)


# ── Transolver Block ──────────────────────────────────────────────────────────

class TransolverBlock1d(nn.Module):
    """One Transolver layer: PhysicsAttn + FFN with pre-norm."""

    def __init__(self, dim: int, n_head: int = 4, slice_num: int = 32,
                 mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1   = nn.LayerNorm(dim)
        self.attn    = PhysicsAttn1d(dim, n_head, slice_num)
        self.norm2   = nn.LayerNorm(dim)
        hidden_mlp   = int(dim * mlp_ratio)
        self.ffn     = nn.Sequential(
            nn.Linear(dim, hidden_mlp),
            nn.GELU(),
            nn.Linear(hidden_mlp, dim),
        )
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ── Transolver1d ──────────────────────────────────────────────────────────────

class Transolver1d(nn.Module):
    """Transolver for 1-D structured grids."""

    def __init__(self, n_modes: int = 16, hidden_dim: int = 64,
                 n_layers: int = 4, in_ch: int = 2,
                 n_head: int = 4, slice_num: int = 32,
                 mlp_ratio: float = 2.0):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([
            TransolverBlock1d(hidden_dim, n_head, slice_num, mlp_ratio)
            for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(hidden_dim)
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)
        self.to(DEVICE)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N  = u0.shape
        grid  = torch.linspace(0.0, 1.0, N, device=DEVICE).unsqueeze(0).expand(B, N)
        x     = torch.stack([u0, grid], dim=-1)   # [B, N, 2]
        x     = self.lift(x)                     # [B, N, D]
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(self.norm(x)))
        return self.proj2(x)[:, :, 0]           # [B, N]


# ── Physics Attention (2-D structured grid) ───────────────────────────────────

class PhysicsAttn2d(nn.Module):
    """Physics Attention over 2-D structured grids [B, N1, N2, D]."""

    def __init__(self, dim: int, n_head: int = 4, slice_num: int = 32):
        super().__init__()
        self.inner = PhysicsAttn1d(dim, n_head, slice_num)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N1, N2, D = x.shape
        x_flat = x.reshape(B, N1 * N2, D)
        out    = self.inner(x_flat)
        return out.reshape(B, N1, N2, D)


class TransolverBlock2d(nn.Module):
    """One 2-D Transolver layer."""
    def __init__(self, dim: int, n_head: int = 4, slice_num: int = 32,
                 mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = PhysicsAttn2d(dim, n_head, slice_num)
        self.norm2 = nn.LayerNorm(dim)
        hidden_mlp = int(dim * mlp_ratio)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, hidden_mlp),
            nn.GELU(),
            nn.Linear(hidden_mlp, dim),
        )
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Transolver2d(nn.Module):
    """Transolver for 2-D structured grids (darcy_2d, ns_2d)."""

    def __init__(self, n_modes1: int = 12, n_modes2: int = 12,
                 hidden_dim: int = 64, n_layers: int = 4,
                 in_ch: int = 3, n_head: int = 4, slice_num: int = 64,
                 mlp_ratio: float = 2.0):
        super().__init__()
        self.lift   = nn.Linear(in_ch, hidden_dim)
        self.blocks = nn.ModuleList([
            TransolverBlock2d(hidden_dim, n_head, slice_num, mlp_ratio)
            for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(hidden_dim)
        self.proj1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2  = nn.Linear(hidden_dim // 2, 1)
        self.to(DEVICE)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N1, N2 = u0.shape
        grid1 = torch.linspace(0.0, 1.0, N1, device=DEVICE).view(1, N1, 1).expand(B, N1, N2)
        grid2 = torch.linspace(0.0, 1.0, N2, device=DEVICE).view(1, 1, N2).expand(B, N1, N2)
        x     = torch.stack([u0, grid1, grid2], dim=-1)   # [B, N1, N2, 3]
        x     = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x     = F.gelu(self.proj1(self.norm(x)))
        return self.proj2(x)[:, :, :, 0]               # [B, N1, N2]
