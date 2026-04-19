"""Modular loss function library for SciML experiments.

All losses take (pred, y) as [B, N] (1D) or [B, N, N] (2D) tensors
and return a scalar.  They are drop-in replacements for the default
relative-L2 loss in train.py.

Usage in train.py:
    from losses import get_loss_fn
    loss_fn_core = get_loss_fn(LOSS_TYPE, **loss_kwargs)
    # then: loss = loss_fn_core(pred, y)

Available loss types:
    "l2_rel"     – relative L2 (default, current baseline)
    "h1"         – H1 Sobolev: L2 + α·L2(∂u/∂x)  (targets shock fronts); 2D supported
    "h1_strong"  – H1 with α=1.0
    "spectral"   – frequency-weighted L2 (emphasises high-k errors); 2D supported
    "l1_rel"     – relative L1 (robust to outliers)
    "mse"        – plain MSE (no normalisation — for debugging)

References:
    H1 loss: Wen et al. (2022) "U-FNO: An enhanced FNO-based deep learning model
        for multiphase flow" arXiv:2109.03697.  Section 3.2.
    Spectral loss: Guibas et al. (2022) AFNO, Section 4.
"""

import math
import torch


# ── Spectral derivative helpers ───────────────────────────────────────────────

def spectral_grad_1d(u: torch.Tensor) -> torch.Tensor:
    """1D spectral derivative ∂u/∂x via FFT.  u: [B, N] → ∂u/∂x: [B, N]."""
    B, N = u.shape
    k    = torch.arange(N // 2 + 1, dtype=torch.float32, device=u.device)
    u_ft = torch.fft.rfft(u, dim=1)
    # Multiply by ik: real part = -imag*k, imag part = real*k
    du_ft_r = -u_ft.imag * k[None, :]
    du_ft_i =  u_ft.real * k[None, :]
    return torch.fft.irfft(du_ft_r + 1j * du_ft_i, n=N, dim=1)


def spectral_grad2_1d(u: torch.Tensor) -> torch.Tensor:
    """Second spectral derivative ∂²u/∂x².  u: [B, N] → [B, N]."""
    B, N = u.shape
    k    = torch.arange(N // 2 + 1, dtype=torch.float32, device=u.device)
    u_ft = torch.fft.rfft(u, dim=1)
    # Multiply by -k²
    d2u_ft_r = -(k ** 2)[None, :] * u_ft.real
    d2u_ft_i = -(k ** 2)[None, :] * u_ft.imag
    return torch.fft.irfft(d2u_ft_r + 1j * d2u_ft_i, n=N, dim=1)


def _spectral_grad_2d(u: torch.Tensor):
    """2D spectral gradients ∂u/∂x and ∂u/∂y.

    u: [B, N1, N2] — single-channel 2D field.
    Returns (du_dx, du_dy), each [B, N1, N2].

    Uses rfft2 for efficiency.  Wavenumbers correspond to a periodic
    domain [0, 1]² so k = 2π × integer wavenumber.
    """
    B, N1, N2 = u.shape
    u_hat = torch.fft.rfft2(u, dim=(1, 2))  # [B, N1, N2//2+1] complex

    # Integer wavenumbers matching numpy.fft.fftfreq(N)*N semantics
    kx_int = [k if k <= N1 // 2 else k - N1 for k in range(N1)]
    kx = torch.tensor(kx_int, dtype=torch.float32, device=u.device).reshape(1, N1, 1) * (2.0 * math.pi)
    ky = (torch.arange(N2 // 2 + 1, dtype=torch.float32, device=u.device).reshape(1, 1, N2 // 2 + 1)
          * (2.0 * math.pi))

    # Multiply by ik: (a + ib)(ik) = -b*k + ia*k
    du_dx_hat_r = -u_hat.imag * kx
    du_dx_hat_i =  u_hat.real * kx
    du_dy_hat_r = -u_hat.imag * ky
    du_dy_hat_i =  u_hat.real * ky

    du_dx = torch.fft.irfft2(du_dx_hat_r + 1j * du_dx_hat_i, s=(N1, N2), dim=(1, 2))
    du_dy = torch.fft.irfft2(du_dy_hat_r + 1j * du_dy_hat_i, s=(N1, N2), dim=(1, 2))
    return du_dx, du_dy


# ── Core loss implementations ─────────────────────────────────────────────────

def relative_l2(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Relative L2 loss (current default).

    mean_batch( ‖pred − y‖₂ / ‖y‖₂ )
    """
    axes      = tuple(range(1, y.ndim))
    diff      = pred - y
    data_loss = torch.mean(
        torch.sqrt(torch.mean(diff ** 2, dim=axes))
        / (torch.sqrt(torch.mean(y   ** 2, dim=axes)) + 1e-8)
    )
    return data_loss


def h1_loss(pred: torch.Tensor, y: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """H1 Sobolev loss: relative L2 + α · relative L2 of first derivative.

    Penalises high-frequency errors more than L2 alone.  Particularly useful
    when the target has sharp gradients (Burgers shock, KdV soliton).
    Supports both 1D ([B, N]) and 2D ([B, N1, N2]) inputs.

    Args:
        pred:  [B, N] or [B, N1, N2]  model prediction
        y:     [B, N] or [B, N1, N2]  ground truth
        alpha: weight on the derivative term (default 0.1)
    """
    l2 = relative_l2(pred, y)

    if y.ndim == 2:
        # 1D path
        diff   = pred - y
        d_diff = spectral_grad_1d(diff)
        d_y    = spectral_grad_1d(y)
        grad_loss = torch.mean(
            torch.sqrt(torch.mean(d_diff ** 2, dim=(1,)))
            / (torch.sqrt(torch.mean(d_y ** 2, dim=(1,))) + 1e-8)
        )
    elif y.ndim == 3:
        # 2D path: compute ∂/∂x and ∂/∂y gradients
        diff  = pred - y
        dx, dy     = _spectral_grad_2d(diff)
        dx_y, dy_y = _spectral_grad_2d(y)
        diff_norm = torch.sqrt(torch.mean(dx ** 2 + dy ** 2, dim=(1, 2)))
        y_norm    = torch.sqrt(torch.mean(dx_y ** 2 + dy_y ** 2, dim=(1, 2)))
        grad_loss = torch.mean(diff_norm / (y_norm + 1e-8))
    else:
        # Higher-dimensional: fall back to L2 (multi-channel 2D etc.)
        return l2

    return l2 + alpha * grad_loss


def h1_strong_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """H1 loss with α=1.0 — equal weighting of L2 and H1 terms."""
    return h1_loss(pred, y, alpha=1.0)


def h2_loss(pred: torch.Tensor, y: torch.Tensor,
            alpha: float = 0.1, beta: float = 0.01) -> torch.Tensor:
    """H2 Sobolev loss: L2 + α·H1 + β·H2.

    Adds a second-derivative penalty.  Smooths more aggressively.
    1D only (2D second derivatives degrade gracefully to H1).
    """
    if y.ndim != 2:
        # Use H1 for 2D (no 2D second derivative implemented)
        return h1_loss(pred, y, alpha=alpha)

    axes  = (1,)
    diff  = pred - y
    d1    = spectral_grad_1d(diff)
    d2    = spectral_grad2_1d(diff)
    d1_y  = spectral_grad_1d(y)
    d2_y  = spectral_grad2_1d(y)

    l2   = relative_l2(pred, y)
    h1t  = torch.mean(
        torch.sqrt(torch.mean(d1 ** 2, dim=axes))
        / (torch.sqrt(torch.mean(d1_y ** 2, dim=axes)) + 1e-8)
    )
    h2t  = torch.mean(
        torch.sqrt(torch.mean(d2 ** 2, dim=axes))
        / (torch.sqrt(torch.mean(d2_y ** 2, dim=axes)) + 1e-8)
    )
    return l2 + alpha * h1t + beta * h2t


def spectral_loss(pred: torch.Tensor, y: torch.Tensor,
                  high_freq_weight: float = 2.0) -> torch.Tensor:
    """Frequency-weighted L2 loss.

    Weights Fourier coefficients by k^high_freq_weight, emphasising
    high-frequency components.  Useful when the model under-predicts
    fine-scale structure (e.g., soliton tails in KdV).
    Supports both 1D ([B, N]) and 2D ([B, N1, N2]) inputs.

    Args:
        high_freq_weight: exponent for frequency weighting (default 2.0)
    """
    if y.ndim == 2:
        # 1D path
        B, N     = y.shape
        k        = torch.arange(N // 2 + 1, dtype=torch.float32, device=y.device)
        weights  = 1.0 + (k / (N // 2)) ** high_freq_weight  # [N//2+1]

        pred_ft  = torch.fft.rfft(pred, dim=1)
        y_ft     = torch.fft.rfft(y,    dim=1)
        diff_ft  = pred_ft - y_ft

        diff_mag = diff_ft.real ** 2 + diff_ft.imag ** 2      # [B, N//2+1]
        y_mag    = y_ft.real    ** 2 + y_ft.imag    ** 2

        w_err = torch.mean(weights[None, :] * diff_mag, dim=1)   # [B]
        w_nrm = torch.mean(weights[None, :] * y_mag,   dim=1)
        return torch.mean(torch.sqrt(w_err) / (torch.sqrt(w_nrm) + 1e-8))

    elif y.ndim == 3:
        # 2D path: weight by 2D wavenumber magnitude |k|²
        B, N1, N2 = y.shape
        kx_int = [k if k <= N1 // 2 else k - N1 for k in range(N1)]
        kx = torch.tensor(kx_int, dtype=torch.float32, device=y.device).reshape(N1, 1)
        ky = torch.arange(N2 // 2 + 1, dtype=torch.float32, device=y.device).reshape(1, N2 // 2 + 1)
        k_norm = torch.sqrt(kx ** 2 + ky ** 2) / (max(N1, N2) // 2)  # [N1, N2//2+1]
        weights = (1.0 + k_norm ** high_freq_weight).reshape(1, N1, N2 // 2 + 1)

        pred_ft = torch.fft.rfft2(pred, dim=(1, 2))
        y_ft    = torch.fft.rfft2(y,    dim=(1, 2))
        diff_ft = pred_ft - y_ft

        diff_mag = diff_ft.real ** 2 + diff_ft.imag ** 2
        y_mag    = y_ft.real    ** 2 + y_ft.imag    ** 2

        w_err = torch.mean(weights * diff_mag, dim=(1, 2))   # [B]
        w_nrm = torch.mean(weights * y_mag,   dim=(1, 2))
        return torch.mean(torch.sqrt(w_err) / (torch.sqrt(w_nrm) + 1e-8))

    else:
        return relative_l2(pred, y)


def adaptive_h1_loss(pred: torch.Tensor, y: torch.Tensor, base_alpha: float = 0.1, alpha: float = None) -> torch.Tensor:
    """H1 Sobolev loss with auto-scaled alpha.

    Scales the gradient penalty so it contributes ~30% of total loss regardless
    of the current training phase.  Avoids over-regularising late in training
    when the shock/gradient is already well-captured.

    Supports 1D ([B, N]) and 2D ([B, N1, N2]) inputs.
    """
    l2 = relative_l2(pred, y)

    if y.ndim == 2:
        diff   = pred - y
        d_diff = spectral_grad_1d(diff)
        d_y    = spectral_grad_1d(y)
        grad_loss = torch.mean(
            torch.sqrt(torch.mean(d_diff ** 2, dim=(1,)))
            / (torch.sqrt(torch.mean(d_y ** 2, dim=(1,))) + 1e-8)
        )
    elif y.ndim == 3:
        diff  = pred - y
        dx, dy_g   = _spectral_grad_2d(diff)
        dx_y, dy_y = _spectral_grad_2d(y)
        diff_norm = torch.sqrt(torch.mean(dx ** 2 + dy_g ** 2, dim=(1, 2)))
        y_norm    = torch.sqrt(torch.mean(dx_y ** 2 + dy_y ** 2, dim=(1, 2)))
        grad_loss = torch.mean(diff_norm / (y_norm + 1e-8))
    else:
        return l2

    # Auto-scale: target_ratio=0.3 means grad term = 30% of total
    # alpha = (target_ratio / (1 - target_ratio)) * (l2 / grad_loss)
    # Use detach so we don't optimise the scale itself
    # Accept `alpha` as alias for `base_alpha` (for compatibility with train.py --loss h1_adaptive)
    if alpha is not None:
        base_alpha = alpha
    auto_alpha = (0.3 / 0.7 * l2 / (grad_loss + 1e-8)).detach() * base_alpha
    auto_alpha = torch.minimum(auto_alpha, 5.0 * base_alpha)  # cap to avoid instability
    return l2 + auto_alpha * grad_loss


def relative_l1(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Relative L1 loss.  Robust to outlier samples."""
    axes = tuple(range(1, y.ndim))
    return torch.mean(
        torch.mean(torch.abs(pred - y), dim=axes)
        / (torch.mean(torch.abs(y), dim=axes) + 1e-8)
    )


def mse_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Plain MSE — no normalisation.  Use for debugging only."""
    return torch.mean((pred - y) ** 2)


# ── Registry ──────────────────────────────────────────────────────────────────

_LOSS_REGISTRY: dict[str, callable] = {
    "l2_rel":      relative_l2,
    "h1":          h1_loss,
    "h1_strong":   h1_strong_loss,
    "h1_adaptive": adaptive_h1_loss,
    "h2":          h2_loss,
    "spectral":    spectral_loss,
    "l1_rel":      relative_l1,
    "mse":         mse_loss,
}


def get_loss_fn(name: str, **kwargs):
    """Return loss function by name, binding any keyword arguments.

    Example:
        loss_fn = get_loss_fn("h1", alpha=0.2)
        loss_fn = get_loss_fn("l2_rel")
    """
    if name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss: {name!r}.  "
            f"Available: {sorted(_LOSS_REGISTRY.keys())}"
        )
    fn = _LOSS_REGISTRY[name]
    if kwargs:
        import functools
        return functools.partial(fn, **kwargs)
    return fn


def list_losses() -> list[str]:
    return sorted(_LOSS_REGISTRY.keys())
