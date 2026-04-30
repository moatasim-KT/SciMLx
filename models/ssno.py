"""State-Space Neural Operator (SS-NO) for 1-D operator learning.

Combines diagonal S4D state-space models with spectral convolutions.
SSM captures long-range sequential patterns; spectral conv captures global modes.

Reference: arXiv:2507.23428 — "State-Space Neural Operators"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.device import DEVICE


class AdaptiveS4DLayer(nn.Module):
    """S4D layer with adaptive (learnable) damping coefficient."""

    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Discretisation step (per channel)
        self.log_dt = nn.Parameter(torch.empty(d_model).uniform_(math.log(0.001), math.log(0.1)))

        # Adaptive real part of A: log(|damping|), always negative after negation
        self.log_damping = nn.Parameter(torch.full((d_model,), math.log(0.5)))

        # Imaginary part of A (fixed HiPPO-like initialization)
        self.a_imag = nn.Parameter(torch.arange(d_state, dtype=torch.float32).repeat(d_model, 1))

        # Learnable frequency scale (modulates imaginary frequencies per channel)
        self.freq_scale = nn.Parameter(torch.ones(d_model))

        # B and C matrices (complex)
        scale = d_state ** -0.5
        self.b_real = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.b_imag = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.c_real = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.c_imag = nn.Parameter(torch.randn(d_model, d_state) * scale)

        # D: skip connection weight
        self.d = nn.Parameter(torch.ones(d_model))
        self.to(DEVICE)

    def _get_kernel(self, L: int) -> torch.Tensor:
        """Compute SSM convolution kernel of length L.  Returns [d_model, L]."""
        dt = torch.exp(self.log_dt)                          # [H]
        a_real = -torch.exp(self.log_damping)                # [H]
        a_imag = self.a_imag * torch.abs(self.freq_scale).unsqueeze(-1)  # [H, N]
        
        a = torch.complex(a_real.unsqueeze(-1), a_imag) # [H, N]
        b = torch.complex(self.b_real, self.b_imag)
        c = torch.complex(self.c_real, self.c_imag)

        t = torch.arange(L, dtype=torch.float32, device=DEVICE)                # [L]

        # Exponent: a * dt * t  →  [H, L, N]
        at = a.unsqueeze(1) * dt.view(-1, 1, 1) * t.view(1, -1, 1)
        exp_at = torch.exp(at) # [H, L, N]

        # kernel: [H, L]
        k = torch.sum(c.unsqueeze(1) * exp_at * b.unsqueeze(1), dim=-1)
        return k.real

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, H] → [B, L, H]"""
        _, L, _ = x.shape
        k = self._get_kernel(L)                             # [H, L]

        # FFT convolution
        x_transpose = x.transpose(1, 2) # [B, H, L]
        x_ft = torch.fft.rfft(x_transpose, n=L, dim=-1)
        k_ft = torch.fft.rfft(k, n=L, dim=-1)                 # [H, L_ft]
        
        y_ft = x_ft * k_ft.unsqueeze(0)
        y = torch.fft.irfft(y_ft, n=L, dim=-1).transpose(1, 2)  # [B, L, H]

        return y + x * self.d.view(1, 1, -1)


class SSNOBlock1d(nn.Module):
    """SS-NO block: SSM branch + spectral conv branch, fused via learned gate."""

    def __init__(self, hidden_dim: int, n_modes: int, d_state: int = 32):
        super().__init__()
        # SSM branch
        self.ssm = AdaptiveS4DLayer(hidden_dim, d_state=d_state)
        self.ssm_proj = nn.Linear(hidden_dim, hidden_dim)

        # Spectral conv branch
        self.w = nn.Parameter(torch.randn(n_modes, hidden_dim, hidden_dim, dtype=torch.complex64) * (hidden_dim ** -0.5))
        self.n_modes = n_modes
        self.spec_proj = nn.Linear(hidden_dim, hidden_dim)

        # Shared local linear (bypass path)
        self.W = nn.Linear(hidden_dim, hidden_dim)

        # Gating: combine SSM + spectral outputs
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.to(DEVICE)

    def _spectral_conv(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, H] → [B, N, H]"""
        B, N, H = x.shape
        x_ft = torch.fft.rfft(x, dim=1)
        nm = min(self.n_modes, x_ft.shape[1])

        out_ft = torch.zeros_like(x_ft, dtype=torch.complex64, device=DEVICE)
        out_ft[:, :nm] = torch.einsum("bmc,mco->bmo", x_ft[:, :nm], self.w[:nm])
        
        return torch.fft.irfft(out_ft, n=N, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, H] → [B, N, H]"""
        x_norm = self.norm(x)

        ssm_out = F.gelu(self.ssm_proj(self.ssm(x_norm)))    # [B, N, H]
        spec_out = F.gelu(self.spec_proj(self._spectral_conv(x_norm)))  # [B, N, H]
        bypass = self.W(x_norm)

        fused = F.gelu(self.gate(torch.cat([ssm_out, spec_out], dim=-1)))
        return x + fused + bypass


class SSNO1d(nn.Module):
    """State-Space Neural Operator for 1-D PDE operator learning."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        n_modes: int = 16,
        d_state: int = 32,
        **kw,
    ):
        super().__init__()
        self.lift = nn.Linear(2, hidden_dim)
        self.blocks = nn.ModuleList([
            SSNOBlock1d(hidden_dim, n_modes, d_state) for _ in range(n_layers)
        ])
        self.proj1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj2 = nn.Linear(hidden_dim // 2, 1)
        self.to(DEVICE)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """inp: [B, N, C] → [B, N, 1]"""
        if inp.ndim == 2:
            B, N = inp.shape
            grid = torch.linspace(0.0, 1.0, N, device=DEVICE).unsqueeze(0).expand(B, N)
            x = torch.stack([inp, grid], dim=-1)   # [B, N, 2]
        else:
            x = inp  # [B, N, C]

        # Re-project input channels to hidden_dim via lift (expects C=2)
        if x.shape[-1] != 2:
            if x.shape[-1] > 2:
                x = x[..., :2]
            else:
                x = F.pad(x, (0, 2 - x.shape[-1]))

        x = self.lift(x)
        for block in self.blocks:
            x = block(x)

        x = F.gelu(self.proj1(x))
        return self.proj2(x)[:, :, 0]
