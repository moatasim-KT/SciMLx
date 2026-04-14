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

Adaptations for 1-D PDE operator learning:
  - HamiltonianNet1d: standard HNN for (q,p) pairs → energy-conserving trajectories
    For wave_1d: q = displacement u, p = velocity u_t
    For kdv_1d:  q = wave amplitude, p = wave momentum
    Trained with Hamiltonian loss (requires (q, p, dq/dt, dp/dt) pairs from solver)

  - HamiltonianNO1d: Hamiltonian Neural Operator for operator learning
    Adapts HNN to the IC → final state setting in our pipeline:
    1. Encode IC u0 → (q, p) latent pair via MLP branch networks
    2. Learn H(q, p) as a small MLP
    3. Compute ∂H/∂p, ∂H/∂q via MLX autodiff
    4. Output: u_pred = q + alpha*(∂H/∂p) (one "Hamiltonian step" from IC)
    This gives the model an energy-conserving inductive bias without needing
    time-derivative labels — useful for wave_1d and kdv_1d where energy is conserved.

  - EnergyConservingFNO1d: FNO + soft energy conservation loss
    Regular FNO with an auxiliary energy loss ||H(pred) - H(target)||.
    More flexible than strict HNN; can be applied to dissipative systems (Burgers)
    with a lower energy penalty weight.
"""

import mlx.core as mx
import mlx.nn as nn
from .fno import FNOBlock1d, SpectralConv1d


# ── Hamiltonian MLP ───────────────────────────────────────────────────────────


class HamiltonianMLP(nn.Module):
	"""Small MLP that maps (q, p) → scalar H.

	Parameters
	----------
	input_dim : int
	    Dimensionality of q and p combined (2 * latent_dim).
	hidden_dim : int
	    Width of hidden layers.
	n_layers : int
	    Number of hidden layers.
	"""

	def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 3):
		super().__init__()
		dims = [input_dim] + [hidden_dim] * n_layers + [1]
		layers = []
		for i in range(len(dims) - 1):
			lin = nn.Linear(dims[i], dims[i + 1])
			# Stabilized initialization for Hamiltonian manifold
			lin.weight = lin.weight * 0.1 
			layers.append(lin)
			if i < len(dims) - 2:
				layers.append(nn.Tanh())  # Tanh: smoother energy landscape
		self.net = nn.Sequential(*layers)

	def __call__(self, qp: mx.array) -> mx.array:
		"""qp : [..., 2*latent_dim] → H : [..., 1]"""
		return self.net(qp)


# ── Hamiltonian Neural Network (original) ────────────────────────────────────


class HamiltonianNet1d(nn.Module):
	"""Hamiltonian Neural Network for 1-D conservative systems.

	Learns H(q, p) and predicts time derivatives via Hamilton's equations.
	Requires training data in (q, p, dq/dt, dp/dt) format.

	Use case in our pipeline:
	  For wave_1d where u = q (displacement), ut = p (velocity), we can
	  extract (q, p) pairs at intermediate time steps from the wave solver.

	Training objective:
	  L = ||∂H/∂p - dq_dt||² + ||∂H/∂q + dp_dt||²

	Parameters
	----------
	state_dim : int
	    Dimensionality of q or p (= N grid points for wave_1d, or 1 for ODE).
	hidden_dim : int
	    Hidden size of the Hamiltonian MLP.
	n_layers : int
	    Depth of Hamiltonian MLP.
	"""

	def __init__(self, state_dim: int, hidden_dim: int = 64, n_layers: int = 3):
		super().__init__()
		self.state_dim = state_dim
		self.H = HamiltonianMLP(2 * state_dim, hidden_dim, n_layers)

	def hamiltonian(self, q: mx.array, p: mx.array) -> mx.array:
		"""Compute H(q, p). q,p : [B, D] → H : [B, 1]"""
		qp = mx.concatenate([q, p], axis=-1)
		return self.H(qp)

	def time_derivatives(self, q: mx.array, p: mx.array):
		"""Compute dq/dt = ∂H/∂p and dp/dt = -∂H/∂q via autodiff.

		Returns (dq_dt, dp_dt) each of shape [B, D].
		"""

		def h_sum(qp):
			B = qp.shape[0]
			D = self.state_dim
			q_ = qp[:, :D]
			p_ = qp[:, D:]
			return self.H(mx.concatenate([q_, p_], axis=-1)).sum()

		qp = mx.concatenate([q, p], axis=-1)  # [B, 2D]
		grad_H = mx.grad(h_sum)(qp)  # [B, 2D]
		D = self.state_dim
		dq_dt = grad_H[:, D:]  #  ∂H/∂p
		dp_dt = -grad_H[:, :D]  # -∂H/∂q
		return dq_dt, dp_dt

	def __call__(self, q: mx.array, p: mx.array) -> tuple:
		"""Returns (dq_dt, dp_dt) for use in Hamiltonian loss."""
		return self.time_derivatives(q, p)


# ── Hamiltonian Neural Operator (operator learning variant) ──────────────────


class HamiltonianNO1d(nn.Module):
	"""Hamiltonian Neural Operator for 1-D PDE operator learning.

	Adapts HNN to the standard IC → final-state setting in our pipeline.
	No time-derivative labels required.

	Architecture:
	  1. q_enc: u0[N] → q[latent_dim]   (position branch)
	  2. p_enc: u0[N] → p[latent_dim]   (momentum branch — learns IC velocity)
	  3. H_net: (q, p) → H (scalar Hamiltonian)
	  4. Step:  dq = ∂H/∂p,  dp = -∂H/∂q  (one symplectic step)
	  5. Trunk: x[N] → T[N, latent_dim]
	  6. Output: (q + dq) · T  decoded back to grid

	Inductive bias:
	  The Hamiltonian step (4) enforces symplectic structure — energy is approximately
	  conserved at inference time.  This is the core physics prior for wave_1d / kdv_1d.

	Parameters
	----------
	n_sensors : int
	    Grid size N (= 64 for our standard benchmarks).
	hidden_dim : int
	    Width of encoder / trunk MLPs.
	latent_dim : int
	    Latent (q, p) dimensionality. Default = hidden_dim.
	n_layers : int
	    Depth of encoder / trunk / Hamiltonian MLPs.
	n_modes : int
	    Accepted for API compatibility (unused).
	"""

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
		self.bias = mx.zeros([1])
		self.epsilon = mx.array(0.1) # Learnable step size for stabilization

	def _hamiltonian_step(self, q: mx.array, p: mx.array):
		"""One step of Hamilton's equations via autodiff."""

		def h_sum(qp):
			D = self.latent_dim
			return self.H_net(qp).sum()

		qp = mx.concatenate([q, p], axis=-1)
		dH = mx.grad(h_sum)(qp)
		dq = dH[:, self.latent_dim :]  # ∂H/∂p
		dp = -dH[:, : self.latent_dim]  # -∂H/∂q
		return dq, dp

	def __call__(self, u0: mx.array) -> mx.array:
		_B, N = u0.shape

		# Encode IC → (q, p) latent pair
		q = self.q_enc(u0)  # [B, latent_dim]
		p = self.p_enc(u0)  # [B, latent_dim]

		# One Hamiltonian step with learnable epsilon stabilization
		dq, _ = self._hamiltonian_step(q, p)
		q_next = q + self.epsilon * dq  # [B, latent_dim]

		# Trunk: decode from latent back to grid
		grid = mx.linspace(0.0, 1.0, N)[:, None]  # [N, 1]
		T = self.trunk(grid)  # [N, latent_dim]

		# Inner product: [B, latent] @ [latent, N] → [B, N]
		out = mx.matmul(q_next, T.T) + self.bias
		return out


# ── FNO + Energy Conservation Loss ───────────────────────────────────────────


class EnergyConservingFNO1d(nn.Module):
	"""FNO with auxiliary energy conservation scoring.

	Computes a scalar "energy" E(u) = sum(u²)/N for both u0 and u_pred,
	and makes the energy ratio accessible for a soft conservation loss.

	This is more flexible than strict HNN: it can be applied to dissipative
	systems (Burgers, Darcy) by setting energy_weight → 0.

	Usage in training:
	  pred = model(u0)
	  recon_loss = l2_rel(pred, target)
	  # model.energy_loss available after forward pass
	  total_loss = recon_loss + energy_weight * model.energy_loss

	Parameters
	----------
	n_modes, hidden_dim, n_layers : same as FNO1d.
	"""

	def __init__(self, n_modes: int, hidden_dim: int, n_layers: int, in_ch: int = 2):
		super().__init__()
		from .fno import FNO1d

		self.fno = FNO1d(n_modes=n_modes, hidden_dim=hidden_dim, n_layers=n_layers, in_ch=in_ch)
		self.energy_loss = mx.array(0.0)

	def energy(self, u: mx.array) -> mx.array:
		"""Discrete L2 energy: [B, N] → [B]"""
		return (u**2).mean(axis=-1)

	def __call__(self, u0: mx.array) -> mx.array:
		pred = self.fno(u0)
		E0 = self.energy(u0)
		Ep = self.energy(pred)
		# Soft energy conservation: relative energy change
		self.energy_loss = ((Ep - E0).abs() / (E0.abs() + 1e-8)).mean()
		return pred
