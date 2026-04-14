import mlx.core as mx
import mlx.nn as nn
import math

class ChebyshevKANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network Layer using Chebyshev Polynomials.
    Reference: Section 3.1.2 of the PIKAN 2025 review.
    """
    def __init__(self, in_features: int, out_features: int, degree: int = 5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        # Base weights for robustness
        scale_base = 1.0 / math.sqrt(in_features)
        self.base_s = mx.ones([out_features, 1])
        self.base_v = mx.random.normal([out_features, in_features]) * scale_base
        
        # Chebyshev coefficients: [out, in, degree + 1]
        scale_cheb = 1.0 / math.sqrt(in_features * (degree + 1))
        self.cheb_weight = mx.random.normal([out_features, in_features, degree + 1]) * scale_cheb

    def chebyshev_basis(self, x: mx.array):
        """
        Compute Chebyshev basis functions recursively up to self.degree.
        x: [..., in_features]. Assumed to be in [-1, 1].
        Returns: [..., in_features, degree + 1]
        """
        # Ensure x is in range [-1, 1] via tanh if not already
        x = mx.tanh(x)
        
        basis = [mx.ones_like(x)]
        if self.degree > 0:
            basis.append(x)
        
        for n in range(2, self.degree + 1):
            basis.append(2.0 * x * basis[-1] - basis[-2])
            
        return mx.stack(basis, axis=-1)

    def __call__(self, x: mx.array) -> mx.array:
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        
        # Base path
        base_output = nn.silu(x) @ (self.base_s * self.base_v).T
        
        # Chebyshev path
        # basis: [batch, in, deg+1]
        basis = self.chebyshev_basis(x)
        
        # Contract over input features and coefficients
        # basis[b, i, k] * cheb_weight[o, i, k] -> [b, o]
        cheb_output = mx.einsum("bik,oik->bo", basis, self.cheb_weight)
        
        out = base_output + cheb_output
        new_shape = [*list(original_shape[:-1]), self.out_features]
        return out.reshape(new_shape)

class ChebyshevKANBlock1d(nn.Module):
    def __init__(self, channels: int, n_modes: int, degree: int = 5):
        super().__init__()
        from models.fno import SpectralConv1d
        self.spec = SpectralConv1d(channels, channels, n_modes)
        self.cheb = ChebyshevKANLinear(channels, channels, degree=degree)

    def __call__(self, x: mx.array) -> mx.array:
        return self.spec(x) + self.cheb(x)

class cPIKAN_FNO(nn.Module):
    """Chebyshev Physics-Informed KAN Spectral Operator."""
    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, in_ch: int = 2, degree: int = 5):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        self.blocks = [ChebyshevKANBlock1d(hidden_dim, n_modes, degree=degree) for _ in range(n_layers)]
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, u0: mx.array) -> mx.array:
        if u0.ndim == 2:
            B, N = u0.shape
            grid = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N), (B, N))
            x = mx.stack([u0, grid], axis=-1)
        else:
            B, N, _C = u0.shape
            grid = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N, 1), (B, N, 1))
            x = mx.concatenate([u0, grid], axis=-1)
            
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = nn.gelu(self.proj1(x))
        return self.proj2(x)[..., 0]
