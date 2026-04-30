"""Hamiltonian Neural Network (HNN) and Hamiltonian Neural Operator.

Reference:
  "Hamiltonian Neural Networks"
  Sam Greydanus, Misko Dzamba, Jason Yosinski
  NeurIPS 2019 — arXiv:1906.01563

Source repo:
  https://github.com/moatasim-KT/SciML-and-Physics-Informed-Machine-Learning-Examples
  (matlab-deep-learning/Hamiltonian-Neural-Network)

Key idea:
  Instead of learning u(t) directly, learn the Hamiltonian H(q,p) of the system.
  The time evolution is then governed by Hamilton's equations:
    dq/dt =  ∂H/∂p
    dp/dt = -∂H/∂q

  Training loss:
    L_HNN = ||∂H/∂p - dq/dt||² + ||∂H/∂q + dp/dt||²

  This enforces exact energy conservation: H(q(t), p(t)) = const for all t.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fno import FNOBlock1d, SpectralConv1d
from core.device import DEVICE

# ── Hamiltonian MLP ───────────────────────────────────────────────────────────

class HamiltonianMLP(nn.Module):
    """Small MLP that maps (q, p) → scalar H."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * n_layers + [1]
        layers = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            # Stabilized initialization for Hamiltonian manifold
            with torch.no_grad():
                lin.weight.data *= 0.1
            layers.append(lin)
            if i < len(dims) - 2:
                layers.append(nn.Tanh())  # Tanh: smoother energy landscape
        self.net = nn.Sequential(*layers)

    def forward(self, qp: torch.Tensor) -> torch.Tensor:
        """qp : [..., 2*latent_dim] → H : [..., 1]"""
        return self.net(qp)


# ── Hamiltonian Neural Network (original) ────────────────────────────────────

class HamiltonianNet1d(nn.Module):
    """Hamiltonian Neural Network for 1-D conservative systems.

    Learns H(q, p) and predicts time derivatives via Hamilton's equations.
    Requires training data in (q, p, dq/dt, dp/dt) format.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.H = HamiltonianMLP(2 * state_dim, hidden_dim, n_layers)

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute H(q, p). q,p : [B, D] → H : [B, 1]"""
        qp = torch.cat([q, p], dim=-1)
        return self.H(qp)

    def time_derivatives(self, q: torch.Tensor, p: torch.Tensor):
        """Compute dq/dt = ∂H/∂p and dp/dt = -∂H/∂q via autodiff.

        Returns (dq_dt, dp_dt) each of shape [B, D].
        """
        qp = torch.cat([q, p], dim=-1)
        qp.requires_grad_(True)
        H = self.H(qp).sum()
        grad_H = torch.autograd.grad(H, qp, create_graph=True)[0]
        
        D = self.state_dim
        dq_dt = grad_H[:, D:]  #  ∂H/∂p
        dp_dt = -grad_H[:, :D]  # -∂H/∂q
        return dq_dt, dp_dt

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> tuple:
        """Returns (dq_dt, dp_dt) for use in Hamiltonian loss."""
        return self.time_derivatives(q, p)


# ── Hamiltonian Neural Operator (operator learning variant) ──────────────────

class HamiltonianNO1d(nn.Module):
    """Hamiltonian Neural Operator for 1-D PDE operator learning."""

    def __init__(self, n_sensors: int, hidden_dim: int = 64, latent_dim: int = 0, n_layers: int = 4, n_modes: int = 16):
        super().__init__()
        if latent_dim == 0:
            latent_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_sensors = n_sensors

        def _mlp(in_d, out_d):
            dims = [in_d] + [hidden_dim] * n_layers + [out_d]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.GELU())
            return nn.Sequential(*layers)

        # Position and momentum encoders
        self.q_enc = _mlp(n_sensors, latent_dim)
        self.p_enc = _mlp(n_sensors, latent_dim)
        # Hamiltonian network
        self.H_net = HamiltonianMLP(2 * latent_dim, hidden_dim, n_layers=2)
        # Trunk: maps x coordinate → basis vectors
        self.trunk = _mlp(1, latent_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        self.epsilon = nn.Parameter(torch.tensor(0.1)) # Learnable step size for stabilization

    def _hamiltonian_step(self, q: torch.Tensor, p: torch.Tensor):
        """One step of Hamilton's equations via autodiff."""
        qp = torch.cat([q, p], dim=-1)
        qp.requires_grad_(True)
        H = self.H_net(qp).sum()
        dH = torch.autograd.grad(H, qp, create_graph=True)[0]
        dq = dH[:, self.latent_dim :]  # ∂H/∂p
        dp = -dH[:, : self.latent_dim]  # -∂H/∂q
        return dq, dp

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        B, N = u0.shape

        # Encode IC → (q, p) latent pair
        q = self.q_enc(u0)  # [B, latent_dim]
        p = self.p_enc(u0)  # [B, latent_dim]

        # One Hamiltonian step with learnable epsilon stabilization
        dq, _ = self._hamiltonian_step(q, p)
        q_next = q + self.epsilon * dq  # [B, latent_dim]

        # Trunk: decode from latent back to grid
        grid = torch.linspace(0.0, 1.0, N, device=u0.device).unsqueeze(-1)  # [N, 1]
        T = self.trunk(grid)  # [N, latent_dim]

        # Inner product: [B, latent] @ [latent, N] → [B, N]
        out = torch.matmul(q_next, T.T) + self.bias
        return out


# ── FNO + Energy Conservation Loss ───────────────────────────────────────────

class EnergyConservingFNO1d(nn.Module):
    """FNO with auxiliary energy conservation scoring."""

    def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, in_ch: int = 2):
        super().__init__()
        from .fno import FNO1d

        self.fno = FNO1d(n_modes=n_modes, hidden_dim=hidden_dim, n_layers=n_layers, in_ch=in_ch)
        self.register_buffer('energy_loss', torch.tensor(0.0))

    def energy(self, u: torch.Tensor) -> torch.Tensor:
        """Discrete L2 energy: [B, N] → [B]"""
        return (u**2).mean(dim=-1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        pred = self.fno(u0)
        E0 = self.energy(u0)
        Ep = self.energy(pred)
        # Soft energy conservation: relative energy change
        self.energy_loss = ((Ep - E0).abs() / (E0.abs() + 1e-8)).mean()
        return pred
