"""Diagonal State Space Model (S4D) for 1-D operator learning.

S4D is a simplified variant of S4 that uses a diagonal state matrix A,
avoiding the complex low-rank corrections of the original S4 while
preserving the O(N log N) FFT-based convolution property.

Reference:
  Gupta, Gu, & Ré (2022) "On the Parameterization and Initialization of 
  Diagonal State Space Models" (arXiv:2206.11893)
"""

import math
import torch
import torch.nn as nn
from core.device import DEVICE

class S4DLayer(nn.Module):
    """Diagonal S4 layer for 1D sequences."""
    
    def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # 1. Learned log-discretisation step (dt)
        self.log_dt = nn.Parameter(torch.empty(d_model).uniform_(math.log(dt_min), math.log(dt_max)))
        
        # 2. Diagonal A matrix (complex)
        # S4D-Lin initialization: real part is -0.5, imag part is 0...N/2
        self.a_real = nn.Parameter(torch.full((d_model, d_state), -0.5))
        self.a_imag = nn.Parameter(torch.arange(d_state, dtype=torch.float32).repeat(d_model, 1))
        
        # 3. Learned B and C (complex)
        self.b_real = nn.Parameter(torch.randn(d_model, d_state) * (d_state ** -0.5))
        self.b_imag = nn.Parameter(torch.randn(d_model, d_state) * (d_state ** -0.5))
        self.c_real = nn.Parameter(torch.randn(d_model, d_state) * (d_state ** -0.5))
        self.c_imag = nn.Parameter(torch.randn(d_model, d_state) * (d_state ** -0.5))
        
        # 4. Learned D (skip connection)
        self.d = nn.Parameter(torch.ones(d_model))
        
        self.to(DEVICE)

    def _get_kernel(self, L: int) -> torch.Tensor:
        """Compute the convolution kernel K of length L."""
        dt = torch.exp(self.log_dt)
        
        # a, b, c: [H, N] complex
        a = torch.complex(self.a_real, self.a_imag)
        b = torch.complex(self.b_real, self.b_imag)
        c = torch.complex(self.c_real, self.c_imag)
        
        t = torch.arange(L, dtype=torch.float32, device=DEVICE)
        
        # exponent: [H, L, N]
        at = a.unsqueeze(1) * dt.unsqueeze(1).unsqueeze(2) * t.unsqueeze(0).unsqueeze(2)
        exp_at = torch.exp(at) # [H, L, N]
        
        # kernel: [H, L]
        k = torch.sum(c.unsqueeze(1) * exp_at * b.unsqueeze(1), dim=-1)
        return k.real

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, H] input sequence
        Returns:
            [B, L, H] output sequence
        """
        _, L, _ = x.shape
        
        # 1. Compute Kernel
        k = self._get_kernel(L) # [H, L]
        
        # 2. FFT Convolution
        x_transpose = x.transpose(1, 2) # [B, H, L]
        x_ft = torch.fft.rfft(x_transpose, n=L, dim=-1)
        k_ft = torch.fft.rfft(k, n=L, dim=-1) # [H, L_ft]
        
        y_ft = x_ft * k_ft.unsqueeze(0)
        y = torch.fft.irfft(y_ft, n=L, dim=-1).transpose(1, 2) # [B, L, H]
        
        # 3. Skip connection
        return y + x * self.d.view(1, 1, -1)

class S4NO1d(nn.Module):
    """State Space Neural Operator (S4NO) for 1-D operator learning."""
    
    def __init__(self, hidden_dim: int, n_layers: int, d_state: int = 64, in_ch: int = 2):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(S4DLayer(hidden_dim, d_state))
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.GELU())
            self.layers.append(nn.LayerNorm(hidden_dim))
            
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)
        
        self.to(DEVICE)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N = u0.shape
        grid = torch.linspace(0.0, 1.0, N, device=DEVICE).unsqueeze(0).repeat(B, 1)
        x = torch.stack([u0, grid], dim=-1)
        
        x = self.lift(x)
        for layer in self.layers:
            x = layer(x)
            
        x = nn.functional.gelu(self.proj1(x))
        return self.proj2(x)[:, :, 0]
