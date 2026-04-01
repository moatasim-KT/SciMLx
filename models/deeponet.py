import mlx.core as mx
import mlx.nn as nn

class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet) baseline.
    
    Consists of a Branch Net (encodes input function) and a 
    Trunk Net (encodes evaluation coordinates).
    """

    def __init__(self, branch_dim: int, trunk_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        # Branch net: u(x) -> G(u)
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # Trunk net: y -> F(y)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.bias = mx.zeros([1])

    def __call__(self, u: mx.array, y: mx.array) -> mx.array:
        """
        Args:
            u: [B, branch_dim] - initial condition/input function sampled at points
            y: [B, N, trunk_dim] - evaluation coordinates
        Returns:
            [B, N] - operator output
        """
        B, N, _ = y.shape
        
        b_out = self.branch(u)  # [B, out_dim]
        t_out = self.trunk(y)   # [B, N, out_dim]
        
        # Inner product between branch and trunk outputs
        out = mx.einsum("bo,bno->bn", b_out, t_out) + self.bias
        return out
