import mlx.core as mx
import mlx.nn as nn
import math


class KANLinear(nn.Module):
	"""
	Kolmogorov-Arnold Network Layer in MLX.
	Replaces y = Sigma(Wx + b) with y_j = sum_i phi_{i,j}(x_i).

	Each univariate phi_{i,j} is a combination of a base function (SiLU)
	and a learnable spline.

	Reference: Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks"
	"""

	def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3, use_factorization: bool = True):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.grid_size = grid_size
		self.spline_order = spline_order
		self.use_factorization = use_factorization

		# Base weights (Section 3.1.2: Weight Factorization W = diag(s) * V)
		scale_base = 1.0 / math.sqrt(in_features)
		if use_factorization:
			self.base_s = mx.ones([out_features, 1])
			self.base_v = mx.random.normal([out_features, in_features]) * scale_base
		else:
			self.base_weight = mx.random.normal([out_features, in_features]) * scale_base

		# Spline weights: [out, in, grid_size + spline_order]
		n_spline = grid_size + spline_order
		scale_spline = 1.0 / math.sqrt(in_features * grid_size)
		self.spline_weight = mx.random.normal([out_features, in_features, n_spline]) * scale_spline

		# Grid points for B-splines
		self.grid = mx.linspace(-1.0, 1.0, n_spline)[None, None, :]

	def update_grid(self, new_grid_size: int):
		"""Adaptive Grid Extension: Interpolate existing weights to a finer grid."""
		old_n = self.grid_size + self.spline_order
		new_n = new_grid_size + self.spline_order
		
		# Linear interpolation of spline weights from old_n to new_n
		x_old = mx.linspace(0, 1, old_n)
		x_new = mx.linspace(0, 1, new_n)
		
		# Simple linear interpolation logic in MLX
		# indices: find where x_new falls in x_old
		idx = (x_new * (old_n - 1)).astype(mx.int32)
		idx_next = mx.minimum(idx + 1, old_n - 1)
		alpha = (x_new * (old_n - 1)) - idx
		alpha = alpha[None, None, :] # [1, 1, new_n]
		
		w_low = self.spline_weight[:, :, idx]
		w_high = self.spline_weight[:, :, idx_next]
		
		# New interpolated weights
		self.spline_weight = w_low + alpha * (w_high - w_low)
		self.grid_size = new_grid_size
		self.grid = mx.linspace(-1.0, 1.0, new_n)[None, None, :]
		print(f"[KAN] Grid refined: {old_n-self.spline_order} -> {new_grid_size}")

	def b_splines(self, x: mx.array):
		"""
		Compute B-spline basis functions for input x.
		x: [B, N, in_features]
		Returns: [B, N, in_features, n_spline]
		"""
		# Simplified implementation using localized radial-style basis
		# as a proxy for efficiency in MLX without a native spline kernel.
		# This approximates the "Efficient KAN" approach.
		x = x[..., None]  # [B, N, in, 1]
		grid = self.grid  # [1, 1, G]

		# Polynomial-like basis functions localized to grid points
		# Using a Gaussian-like locality kernel for the first version
		dist = mx.abs(x - grid)
		h = 2.0 / self.grid_size
		basis = mx.maximum(1.0 - (dist / h), 0.0)  # Triangular basis
		return basis

	def __call__(self, x: mx.array) -> mx.array:
		# x: [B, ..., in_features]
		original_shape = x.shape
		x = x.reshape(-1, self.in_features)

		# Base path: SiLU(x) * W
		if self.use_factorization:
			# W = s * V
			base_weight = self.base_s * self.base_v
		else:
			base_weight = self.base_weight
		base_output = nn.silu(x) @ base_weight.T

		# Spline path
		# basis: [batch_flat, in_features, grid_points]
		basis = self.b_splines(x)

		# Efficient contraction: sum_{in} basis[b, i, k] * spline_weight[out, i, k]
		# output: [batch_flat, out_features]
		spline_output = mx.einsum("bik,oik->bo", basis, self.spline_weight)

		out = base_output + spline_output

		# Restore shape
		new_shape = [*list(original_shape[:-1]), self.out_features]
		return out.reshape(new_shape)


class KANBlock1d(nn.Module):
	"""1D KAN layer: spectral conv + pointwise KAN."""

	def __init__(self, channels: int, n_modes: int, grid_size: int = 5):
		super().__init__()
		from models.fno import SpectralConv1d

		self.spec = SpectralConv1d(channels, channels, n_modes)
		self.kan = KANLinear(channels, channels, grid_size=grid_size)

	def __call__(self, x: mx.array) -> mx.array:
		# KAN replaces the classic Linear + GELU activation
		return self.spec(x) + self.kan(x)


class KAN_FNO(nn.Module):
	"""Kolmogorov-Arnold Spectral Operator (KAN-FNO) for 1-D PDEs."""

	def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, in_ch: int = 2, grid_size: int = 5):
		super().__init__()
		self.lift = nn.Linear(in_ch, hidden_dim)
		self.blocks = [KANBlock1d(hidden_dim, n_modes, grid_size) for _ in range(n_layers)]
		self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
		self.proj2 = nn.Linear(hidden_dim // 2, 1)

	def __call__(self, u0: mx.array) -> mx.array:
		# u0: [B, N] or [B, N, C]
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

class ModifiedKAN_FNO(nn.Module):
    """
    Modified KAN-FNO incorporating input-projection sub-networks.
    Reference: Section 3.1.2 "Architectures" (Wang et al. modified MLP).
    Uses two encoders to project input into high-dim space, then gates each layer.
    """
    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, in_ch: int = 2, grid_size: int = 5):
        super().__init__()
        self.lift = nn.Linear(in_ch, hidden_dim)
        
        # Encoders for gating
        self.encoder_u = nn.Linear(in_ch, hidden_dim)
        self.encoder_v = nn.Linear(in_ch, hidden_dim)
        
        self.blocks = [KANBlock1d(hidden_dim, n_modes, grid_size) for _ in range(n_layers)]
        
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, u0: mx.array) -> mx.array:
        if u0.ndim == 2:
            B, N = u0.shape
            grid = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N), (B, N))
            raw_x = mx.stack([u0, grid], axis=-1)
        else:
            B, N, _C = u0.shape
            grid = mx.broadcast_to(mx.linspace(0.0, 1.0, N).reshape(1, N, 1), (B, N, 1))
            raw_x = mx.concatenate([u0, grid], axis=-1)
            
        # Global encoders
        u = nn.silu(self.encoder_u(raw_x))
        v = nn.silu(self.encoder_v(raw_x))
        
        x = self.lift(raw_x)
        for blk in self.blocks:
            # Modified MLP style: Gate layer with projection
            h = blk(x)
            x = (1.0 - h) * u + h * v
            
        x = nn.gelu(self.proj1(x))
        return self.proj2(x)[..., 0]
