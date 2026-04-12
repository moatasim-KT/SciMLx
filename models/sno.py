import mlx.core as mx
import mlx.nn as nn
from models.fno import SpectralConv2d

class SNO2d(nn.Module):
    """Spectral Neural Operator (No continuous alias back-end, direct spectral mapper)."""
    
    def __init__(self, n_modes=8, hidden_dim=32, n_layers=4, in_channels=1, out_channels=1):
        super().__init__()
        # Lift: data channels + 2 spatial grids
        self.p = nn.Linear(in_channels + 2, hidden_dim)
        
        self.spectral_layers = []
        for _ in range(n_layers):
            # purely spectral layers, omitting standard MLP to act purely in frequency
            self.spectral_layers.append(SpectralConv2d(in_ch=hidden_dim, out_ch=hidden_dim, n_modes1=n_modes, n_modes2=n_modes))
            
        self.q = nn.Linear(hidden_dim, out_channels)

    def __call__(self, x):
        # x : [B, N1, N2] or [B, N1, N2, C]
        if x.ndim == 3:
            B, N1, N2 = x.shape
            x = x[..., None]
        else:
            B, N1, N2, _ = x.shape
            
        grid1 = mx.broadcast_to(mx.linspace(0.0, 1.0, N1).reshape(1, N1, 1, 1), (B, N1, N2, 1))
        grid2 = mx.broadcast_to(mx.linspace(0.0, 1.0, N2).reshape(1, 1, N2, 1), (B, N1, N2, 1))
        x     = mx.concatenate([x, grid1, grid2], axis=-1)  # [B, N1, N2, C+2]

        x = self.p(x)
        for spec in self.spectral_layers:
            # strictly spectral operations
            x = nn.gelu(spec(x))
        
        out = self.q(x)
        if out.shape[-1] == 1:
            return out[:, :, :, 0]
        return out
