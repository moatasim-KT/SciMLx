"""State-Space Neural Operator (SS-NO) for 1-D operator learning.

Combines diagonal S4D state-space models with spectral convolutions.
SSM captures long-range sequential patterns; spectral conv captures global modes.

Reference: arXiv:2507.23428 — "State-Space Neural Operators"
Key claims: 0.0070 on Burgers (vs SOTA 0.0149), ~200k params, 5x better than SOTA

Key innovations over vanilla S4NO:
  - Adaptive damping: learnable per-channel damping coefficient (not fixed at -0.5)
  - Frequency modulation: SSM imaginary frequencies scaled by learned gate
  - Dual-branch block: SSM output + spectral conv output fused via learned gate
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveS4DLayer(nn.Module):
    """S4D layer with adaptive (learnable) damping coefficient."""

    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Discretisation step (per channel)
        self.log_dt = nn.Parameter(
            torch.empty(d_model).uniform_(math.log(0.001), math.log(0.1))
        )

        # Adaptive real part of A: log(|damping|), always negative after negation
        self.log_damping = nn.Parameter(torch.full((d_model,), math.log(0.5)))

        # Imaginary part of A (fixed HiPPO-like initialization)
        im_a = torch.arange(d_state, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.register_buffer('a_imag', im_a)  # [d_model, d_state]

        # Learnable frequency scale (modulates imaginary frequencies per channel)
        self.freq_scale = nn.Parameter(torch.ones(d_model))

        # B and C matrices (complex, stored as real + imaginary)
        scale = d_state ** -0.5
        self.b_real = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.b_imag = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.c_real = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.c_imag = nn.Parameter(torch.randn(d_model, d_state) * scale)

        # D: skip connection weight
        self.d = nn.Parameter(torch.ones(d_model))

    def _get_kernel(self, L: int) -> torch.Tensor:
        """Compute SSM convolution kernel of length L.  Returns [d_model, L]."""
        dt = torch.exp(self.log_dt)                          # [H]
        a_real = -torch.exp(self.log_damping)                # [H], always negative
        a_imag = self.a_imag * torch.abs(self.freq_scale).unsqueeze(-1)  # [H, N]

        t = torch.arange(L, dtype=torch.float32, device=self.log_dt.device)  # [L]

        # Exponent: a * dt * t  →  [H, L, N]
        at_r = a_real[:, None, None] * dt[:, None, None] * t[None, :, None]
        at_i = a_imag[:, None, :] * dt[:, None, None] * t[None, :, None]

        exp_r = torch.exp(at_r)          # [H, L, N]
        cos_i = torch.cos(at_i)          # [H, L, N]
        sin_i = torch.sin(at_i)          # [H, L, N]

        # Real part of C * exp(A*dt*t) * B
        cb_rr = (self.c_real * self.b_real).unsqueeze(1)   # [H, 1, N]
        cb_ii = (self.c_imag * self.b_imag).unsqueeze(1)
        cb_ri = (self.c_real * self.b_imag).unsqueeze(1)
        cb_ir = (self.c_imag * self.b_real).unsqueeze(1)

        k = torch.sum(
            exp_r * ((cb_rr - cb_ii) * cos_i + (cb_ri + cb_ir) * sin_i),
            dim=-1,
        )  # [H, L]
        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, H] → [B, L, H]"""
        _, L, _ = x.shape
        k = self._get_kernel(L)                             # [H, L]

        # FFT convolution
        x_ft = torch.fft.rfft(x.permute(0, 2, 1), dim=-1)  # [B, H, L//2+1]
        k_ft = torch.fft.rfft(k, n=L, dim=-1)                 # [H, L//2+1]
        y_ft = x_ft * k_ft.unsqueeze(0)
        y = torch.fft.irfft(y_ft, n=L, dim=-1).permute(0, 2, 1)  # [B, L, H]

        return y + x * self.d.unsqueeze(0).unsqueeze(0)


class SSNOBlock1d(nn.Module):
    """SS-NO block: SSM branch + spectral conv branch, fused via learned gate.

    The SSM captures long-range dynamics; spectral conv captures global modes.
    Both branches see the same normalised input; their outputs are gated-merged.
    """

    def __init__(self, hidden_dim: int, n_modes: int, d_state: int = 32):
        super().__init__()
        # SSM branch
        self.ssm = AdaptiveS4DLayer(hidden_dim, d_state=d_state)
        self.ssm_proj = nn.Linear(hidden_dim, hidden_dim)

        # Spectral conv branch (same real/imag split as FNO)
        self.wr = nn.Parameter(torch.randn(n_modes, hidden_dim, hidden_dim) * (hidden_dim ** -0.5))
        self.wi = nn.Parameter(torch.randn(n_modes, hidden_dim, hidden_dim) * (hidden_dim ** -0.5))
        self.n_modes = n_modes
        self.spec_proj = nn.Linear(hidden_dim, hidden_dim)

        # Shared local linear (bypass path)
        self.W = nn.Linear(hidden_dim, hidden_dim)

        # Gating: combine SSM + spectral outputs
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

    def _spectral_conv(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, H] → [B, N, H]  (same API as FNO SpectralConv)"""
        _, N, _ = x.shape
        x_ft = torch.fft.rfft(x, dim=1)                       # [B, N//2+1, H]
        nm = min(self.n_modes, N // 2 + 1)

        out_r = (
            torch.einsum("bmc,mco->bmo", x_ft[:, :nm].real, self.wr[:nm])
            - torch.einsum("bmc,mco->bmo", x_ft[:, :nm].imag, self.wi[:nm])
        )
        out_i = (
            torch.einsum("bmc,mco->bmo", x_ft[:, :nm].real, self.wi[:nm])
            + torch.einsum("bmc,mco->bmo", x_ft[:, :nm].imag, self.wr[:nm])
        )
        padlen = N // 2 + 1 - nm
        if padlen > 0:
            out_r = F.pad(out_r, (0, 0, 0, padlen))
            out_i = F.pad(out_i, (0, 0, 0, padlen))
        y_ft = out_r + 1j * out_i
        return torch.fft.irfft(y_ft, n=N, dim=1)              # [B, N, H]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, H] → [B, N, H]"""
        x_norm = self.norm(x)

        ssm_out = F.gelu(self.ssm_proj(self.ssm(x_norm)))    # [B, N, H]
        spec_out = F.gelu(self.spec_proj(self._spectral_conv(x_norm)))  # [B, N, H]
        bypass = self.W(x_norm)

        fused = F.gelu(self.gate(torch.cat([ssm_out, spec_out], dim=-1)))
        return x + fused + bypass


class SSNO1d(nn.Module):
    """State-Space Neural Operator for 1-D PDE operator learning.

    Args:
        hidden_dim: channel width
        n_layers:   number of SSNOBlock1d layers
        n_modes:    spectral modes per block
        d_state:    SSM state dimension (default 32 for memory efficiency)
    """

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

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """inp: [B, N, C] (standard train.py format) → [B, N, 1]"""
        if inp.ndim == 2:
            # [B, N] fallback: treat as single channel
            B, N = inp.shape
            grid = torch.linspace(0.0, 1.0, N, device=inp.device, dtype=inp.dtype).unsqueeze(0).expand(B, -1)
            x = torch.stack([inp, grid], dim=-1)   # [B, N, 2]
        else:
            x = inp  # [B, N, C] — use as-is after lift projects C→hidden_dim

        # Re-project input channels to hidden_dim via lift (expects C=2)
        if x.shape[-1] != 2:
            # Pad or slice to C=2 for the lift layer
            if x.shape[-1] >= 2:
                x = x[..., :2]
            else:
                x = F.pad(x, (0, 2 - x.shape[-1]))

        x = self.lift(x)                    # [B, N, hidden_dim]
        for block in self.blocks:
            x = block(x)

        x = F.gelu(self.proj1(x))          # [B, N, hidden_dim//2]
        return self.proj2(x)[:, :, 0]       # [B, N]
