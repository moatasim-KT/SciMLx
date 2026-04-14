"""Alias-Free Mamba Neural Operator (MambaNO) for SciML.

Combines the selective state-space modeling (SSM) of Mamba with 
the spectral filtering of neural operators to maintain alias-free 
approximations of PDE operators.

Reference: Zheng et al. (2024) "Alias-Free Mamba Neural Operator" (NeurIPS)
"""

import math
import mlx.core as mx
import mlx.nn as nn

def selective_scan_mlx(u, delta, A, B, C, D=None):
    """
    Simplified selective scan in MLX.
    u: [B, L, D]
    delta: [B, L, D]
    A: [D, N]
    B: [B, L, N]
    C: [B, L, N]
    D: [D] (optional)

    Recurrence:
    h[t] = A_bar * h[t-1] + B_bar * u[t]
    y[t] = C * h[t]
    """
    B_batch, L, D = u.shape
    N = A.shape[1]

    # Discretization
    # A_bar = exp(delta * A)
    # B_bar = (A_bar - I) * A^-1 * B * delta
    # Simplified version for research: 
    # A_bar = exp(delta[..., None] * A[None, None, ...]) # [B, L, D, N]
    
    # Sequence length loop
    h = mx.zeros((B_batch, D, N))
    outputs = []
    
    u_seq = u.split(L, axis=1) # [L] list of [B, 1, D]
    delta_seq = delta.split(L, axis=1)
    B_seq = B.split(L, axis=1)
    C_seq = C.split(L, axis=1)
    
    for t in range(L):
        u_t = u_seq[t][:, 0, :]      # [B, D]
        d_t = delta_seq[t][:, 0, :]  # [B, D]
        b_t = B_seq[t][:, 0, :]      # [B, N]
        c_t = C_seq[t][:, 0, :]      # [B, N]
        
        # Discretize A and B (Zero-Order Hold)
        a_bar = mx.exp(d_t[:, :, None] * A[None, ...])   # [B, D, N]
        b_bar = d_t[:, :, None] * b_t[:, None, :]        # [B, D, N]
        
        h = a_bar * h + b_bar * u_t[:, :, None]          # [B, D, N]
        y_t = mx.sum(h * c_t[:, None, :], axis=-1)       # [B, D]
        outputs.append(y_t)

    y = mx.stack(outputs, axis=1) # [B, L, D]
    
    if D is not None:
        y = y + u * D
        
    return y

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1
        )
        
        # Selective SSM parameters
        self.x_proj = nn.Linear(self.d_inner, 1 + self.d_state * 2)
        self.dt_proj = nn.Linear(1, self.d_inner)
        
        # Fixed A matrix (initialization)
        A = mx.broadcast_to(mx.arange(1, d_state + 1, dtype=mx.float32), (self.d_inner, d_state))
        self.A_log = mx.log(A)
        self.D = mx.ones([self.d_inner])
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def __call__(self, x):
        """x: [B, L, D]"""
        B, L, D = x.shape
        
        xz = self.in_proj(x) # [B, L, 2*D_inner]
        x, z = mx.split(xz, 2, axis=-1)
        
        # Conv path
        x = self.conv1d(x)[:, :L, :] # [B, L, D_inner]
        x = nn.silu(x)
        
        # SSM path
        x_dbl = self.x_proj(x) # [B, L, 1 + 2*d_state]
        dt, B_ssm, C_ssm = mx.split(x_dbl, [1, 1 + self.d_state], axis=-1)
        
        dt = nn.softplus(self.dt_proj(dt)) # [B, L, D_inner]
        A = -mx.exp(self.A_log) # [D_inner, d_state]
        
        y = selective_scan_mlx(x, dt, A, B_ssm, C_ssm, self.D)
        
        # Output gating
        y = y * nn.silu(z)
        return self.out_proj(y)

class MambaNO1d(nn.Module):
    """Alias-Free Mamba Neural Operator implementation."""
    def __init__(self, hidden_dim=64, n_layers=4, n_modes=16, in_ch=2):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        
        self.layers = []
        for _ in range(n_layers):
            # Hybrid block: Mamba for sequence modeling + Spectral for alias-free filtering
            self.layers.append(MambaBlock(hidden_dim))
            self.layers.append(nn.LayerNorm(hidden_dim))
            # Low-pass spectral filter (alias-free constraint)
            self.layers.append(self._spectral_filter_layer(hidden_dim, n_modes))

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _spectral_filter_layer(self, dim, modes):
        class SpectralFilter(nn.Module):
            def __init__(self, d, m):
                super().__init__()
                self.modes = m
                self.w = mx.random.normal([m, d, d]) * (d ** -0.5)
            def __call__(self, x):
                B, N, C = x.shape
                x_ft = mx.fft.rfft(x, axis=1)
                m = min(self.modes, x_ft.shape[1])
                # Filter high frequencies (alias-free)
                out_ft_m = mx.einsum("bnc,nco->bno", x_ft[:, :m, :], self.w[:m])
                
                # Pad/Concatenate to original length
                pad_len = x_ft.shape[1] - m
                if pad_len > 0:
                    padding = mx.zeros((B, pad_len, C), dtype=x_ft.dtype)
                    out_ft = mx.concatenate([out_ft_m, padding], axis=1)
                else:
                    out_ft = out_ft_m
                
                return mx.fft.irfft(out_ft, n=N, axis=1)
        return SpectralFilter(dim, modes)

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
        for layer in self.layers:
            x = layer(x)
            
        return self.proj(x)[:, :, 0]

if __name__ == "__main__":
    model = MambaNO1d()
    u0 = mx.random.normal([2, 64])
    y = model(u0)
    print(f"MambaNO output shape: {y.shape}")
