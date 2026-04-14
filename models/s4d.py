"""Diagonal State Space Model (S4D) for 1-D operator learning.

S4D is a simplified variant of S4 that uses a diagonal state matrix A,
avoiding the complex low-rank corrections of the original S4 while
preserving the O(N log N) FFT-based convolution property.

Reference:
  Gupta, Gu, & Ré (2022) "On the Parameterization and Initialization of 
  Diagonal State Space Models" (arXiv:2206.11893)
"""

import math
import mlx.core as mx
import mlx.nn as nn

class S4DLayer(nn.Module):
    """Diagonal S4 layer for 1D sequences."""
    
    def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # 1. Learned log-discretisation step (dt)
        self.log_dt = mx.random.uniform(math.log(dt_min), math.log(dt_max), [d_model])
        
        # 2. Diagonal A matrix (complex)
        # S4D-Lin initialization: real part is -0.5, imag part is 0...N/2
        # This creates a set of oscillators at different frequencies.
        re_a = mx.full([d_model, d_state], -0.5)
        im_a = mx.broadcast_to(mx.arange(d_state, dtype=mx.float32), (d_model, d_state))
        self.a_real = re_a
        self.a_imag = im_a
        
        # 3. Learned B and C (complex)
        self.b_real = mx.random.normal([d_model, d_state]) * (d_state ** -0.5)
        self.b_imag = mx.random.normal([d_model, d_state]) * (d_state ** -0.5)
        self.c_real = mx.random.normal([d_model, d_state]) * (d_state ** -0.5)
        self.c_imag = mx.random.normal([d_model, d_state]) * (d_state ** -0.5)
        
        # 4. Learned D (skip connection)
        self.d = mx.ones([d_model])

    def _get_kernel(self, L: int) -> mx.array:
        """Compute the convolution kernel K of length L."""
        # dt: [H]
        dt = mx.exp(self.log_dt)
        
        # a, b, c: [H, N] complex
        a = self.a_real + 1j * self.a_imag
        b = self.b_real + 1j * self.b_imag
        c = self.c_real + 1j * self.c_imag
        
        # Discretize: A_bar = exp(A * dt), B_bar = (A_bar - I) * A^-1 * B
        # For diagonal A, this is elementwise.
        # Kernel K_k = C_bar * A_bar^k * B_bar
        # Or more simply via the generating function in frequency domain.
        
        # We use the FFT-convolution approach:
        # K = C * exp(A * dt * [0...L-1]) * B
        # [H, N] * [H, L, N] * [H, N] -> [H, L]
        
        t = mx.arange(L, dtype=mx.float32) # [L]
        
        # exponent: [H, L, N]
        at = a[:, None, :] * dt[:, None, None] * t[None, :, None]
        exp_at = mx.exp(at) # [H, L, N]
        
        # kernel: [H, L]
        # sum over state dimension N
        k = mx.sum(c[:, None, :] * exp_at * b[:, None, :], axis=-1)
        return k.real

    def __call__(self, x: mx.array) -> mx.array:
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
        # x_ft: [B, H, L_ft]
        x_ft = mx.fft.rfft(x.transpose(0, 2, 1), axis=-1)
        k_ft = mx.fft.rfft(k, n=L, axis=-1) # [H, L_ft]
        
        y_ft = x_ft * k_ft[None, :, :]
        y = mx.fft.irfft(y_ft, n=L, axis=-1).transpose(0, 2, 1) # [B, L, H]
        
        # 3. Skip connection
        return y + x * self.d[None, None, :]

class S4NO1d(nn.Module):
    """State Space Neural Operator (S4NO) for 1-D operator learning."""
    
    def __init__(self, hidden_dim: int, n_layers: int, d_state: int = 64, in_ch: int = 2):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(S4DLayer(hidden_dim, d_state))
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.GELU())
            self.layers.append(nn.LayerNorm(hidden_dim))
            
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, u0: mx.array) -> mx.array:
        B, N = u0.shape
        grid = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N), (B, N))
        x = mx.stack([u0, grid], axis=-1)
        
        x = self.lift(x)
        for layer in self.layers:
            x = layer(x)
            
        x = nn.gelu(self.proj1(x))
        return self.proj2(x)[:, :, 0]
