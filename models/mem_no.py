"""Memory Neural Operator (MemNO) for SciML.

Integrates structured state-space models (S4) into FNO architectures to 
provide explicit temporal memory for modeling time-dependent PDEs.

Reference: Buitrago Ruiz et al. (2025) "On the Benefits of Memory for 
Modeling Time-Dependent PDEs" (ICLR)
"""

import math
import mlx.core as mx
import mlx.nn as nn
from models.s4d import S4DLayer

class MemNOBlock1d(nn.Module):
    """Hybrid FNO block with integrated S4 state-space memory."""
    def __init__(self, hidden_dim, n_modes, d_state=64):
        super().__init__()
        # 1. Fourier branch
        self.wr = mx.random.normal([n_modes, hidden_dim, hidden_dim]) * (hidden_dim ** -0.5)
        self.wi = mx.random.normal([n_modes, hidden_dim, hidden_dim]) * (hidden_dim ** -0.5)
        self.n_modes = n_modes
        
        # 2. Memory (SSM) branch
        self.memory = S4DLayer(hidden_dim, d_state=d_state)
        
        # 3. Local bypass
        self.w_local = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)

    def _spectral_conv(self, x):
        """Standard 1D spectral convolution."""
        B, N, C = x.shape
        x_ft = mx.fft.rfft(x, axis=1)
        nm = min(self.n_modes, x_ft.shape[1])
        
        # Complex multiply
        out_ft_r = (
            mx.einsum("bnc,nco->bno", x_ft[:, :nm].real, self.wr[:nm])
            - mx.einsum("bnc,nco->bno", x_ft[:, :nm].imag, self.wi[:nm])
        )
        out_ft_i = (
            mx.einsum("bnc,nco->bno", x_ft[:, :nm].real, self.wi[:nm])
            + mx.einsum("bnc,nco->bno", x_ft[:, :nm].imag, self.wr[:nm])
        )
        
        out_ft = out_ft_r + 1j * out_ft_i
        # Pad high frequencies
        padlen = x_ft.shape[1] - nm
        if padlen > 0:
            out_ft = mx.pad(out_ft, [(0, 0), (0, padlen), (0, 0)])
            
        return mx.fft.irfft(out_ft, n=N, axis=1)

    def __call__(self, x):
        # x: [B, N, H]
        x_norm = self.norm(x)
        
        # Parallel branches
        fourier_out = self._spectral_conv(x_norm)
        memory_out = self.memory(x_norm) # Memory sees full sequence
        local_out = self.w_local(x_norm)
        
        # Fuse with activation
        return x + nn.gelu(fourier_out + memory_out + local_out)

class MemNO1d(nn.Module):
    """Memory Neural Operator architecture."""
    def __init__(self, hidden_dim=64, n_layers=4, n_modes=16, d_state=64, in_ch=2):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        
        self.blocks = [
            MemNOBlock1d(hidden_dim, n_modes, d_state) for _ in range(n_layers)
        ]
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def __call__(self, u0):
        # Handle multi-dimensional inputs (e.g., 2D Darcy/NS) by flattening
        B = u0.shape[0]
        if u0.ndim > 2:
            spatial_shape = u0.shape[1:]
            N = 1
            for s in spatial_shape: N *= s
            u0_flat = u0.reshape(B, N)
        else:
            N = u0.shape[1]
            u0_flat = u0

        grid = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N, 1), (B, N, 1))
        x = mx.concatenate([u0_flat[..., None], grid], axis=-1)
        
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
            
        return self.proj(x)[:, :, 0]

if __name__ == "__main__":
    model = MemNO1d()
    u0 = mx.random.normal([2, 64])
    y = model(u0)
    print(f"MemNO output shape: {y.shape}")
