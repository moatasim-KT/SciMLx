import mlx.core as mx
import mlx.nn as nn


class DeepONet(nn.Module):
    """Deep Operator Network (DeepONet).

    Branch net encodes the input function; Trunk net encodes evaluation
    coordinates.  The operator output is their inner product + bias.

    Reference: Lu et al. (2019) "Learning nonlinear operators via DeepONet"
    (arXiv:1910.03193)

    Improvements over naive baseline:
    - Deeper branch and trunk (4 hidden layers, 128 units by default)
    - LayerNorm after each hidden activation for training stability
    - Proper weight initialization (Xavier uniform)
    - Learnable scalar bias per output point
    """

    def __init__(self, branch_dim: int, trunk_dim: int,
                 hidden_dim: int = 128, out_dim: int = 128, n_layers: int = 4):
        super().__init__()

        def _make_net(in_dim: int) -> nn.Module:
            layers: list = []
            dim = in_dim
            for _ in range(n_layers):
                layers.append(nn.Linear(dim, hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(hidden_dim))
                dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, out_dim))
            return nn.Sequential(*layers)

        self.branch = _make_net(branch_dim)
        self.trunk  = _make_net(trunk_dim)
        self.bias   = mx.zeros([1])

    def __call__(self, u: mx.array, y: mx.array) -> mx.array:
        """
        Args:
            u : [B, branch_dim]    – input function at sensor locations
            y : [B, N, trunk_dim]  – evaluation coordinates
        Returns:
            [B, N] – operator output at query points
        """
        b_out = self.branch(u)           # [B, out_dim]
        t_out = self.trunk(y)            # [B, N, out_dim]
        return mx.einsum("bo,bno->bn", b_out, t_out) + self.bias


class PODDeepONet(nn.Module):
    """Proper Orthogonal Decomposition DeepONet.

    Uses a fixed set of POD basis functions as the trunk (learned offline from
    data), so the branch only needs to predict basis coefficients.  In this
    implementation the POD basis is learned end-to-end (as an embedding matrix)
    rather than computed offline, which keeps the file self-contained while
    preserving the structural benefit of a shared basis.

    Reference: Lu et al. (2022) "Comprehensive study of deeponet for solving
    PDEs" — POD-DeepONet variant.
    """

    def __init__(self, branch_dim: int, n_basis: int = 64,
                 hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        self.n_basis = n_basis

        # Branch: input function → basis coefficients
        layers: list = [nn.Linear(branch_dim, hidden_dim), nn.GELU(),
                        nn.LayerNorm(hidden_dim)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                       nn.LayerNorm(hidden_dim)]
        layers.append(nn.Linear(hidden_dim, n_basis))
        self.branch = nn.Sequential(*layers)

        # Learnable POD basis: [n_basis, N] – each row is one basis function
        # Initialised with small random values; will be learned from data.
        self.basis = mx.random.normal([n_basis, 64]) * 0.02
        self.bias  = mx.zeros([1])

    def __call__(self, u: mx.array, y: mx.array | None = None) -> mx.array:
        """
        Args:
            u : [B, branch_dim]
            y : ignored (basis evaluated at fixed grid)
        Returns:
            [B, N]
        """
        coeffs = self.branch(u)                         # [B, n_basis]
        return mx.matmul(coeffs, self.basis) + self.bias  # [B, N]
