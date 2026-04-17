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
import mlx.core as mx


# ── Spectral derivative helpers ───────────────────────────────────────────────

def spectral_grad_1d(u: mx.array) -> mx.array:
    """1D spectral derivative ∂u/∂x via FFT.  u: [B, N] → ∂u/∂x: [B, N]."""
    B, N = u.shape
    k    = mx.arange(N // 2 + 1, dtype=mx.float32)
    u_ft = mx.fft.rfft(u, axis=1)
    # Multiply by ik: real part = -imag*k, imag part = real*k
    du_ft_r = -u_ft.imag * k[None, :]
    du_ft_i =  u_ft.real * k[None, :]
    return mx.fft.irfft(du_ft_r + 1j * du_ft_i, n=N, axis=1)


def spectral_grad2_1d(u: mx.array) -> mx.array:
    """Second spectral derivative ∂²u/∂x².  u: [B, N] → [B, N]."""
    B, N = u.shape
    k    = mx.arange(N // 2 + 1, dtype=mx.float32)
    u_ft = mx.fft.rfft(u, axis=1)
    # Multiply by -k²
    d2u_ft_r = -(k ** 2)[None, :] * u_ft.real
    d2u_ft_i = -(k ** 2)[None, :] * u_ft.imag
    return mx.fft.irfft(d2u_ft_r + 1j * d2u_ft_i, n=N, axis=1)


def _spectral_grad_2d(u: mx.array):
    """2D spectral gradients ∂u/∂x and ∂u/∂y.

    u: [B, N1, N2] — single-channel 2D field.
    Returns (du_dx, du_dy), each [B, N1, N2].

    Uses rfft2 for efficiency.  Wavenumbers correspond to a periodic
    domain [0, 1]² so k = 2π × integer wavenumber.
    """
    B, N1, N2 = u.shape
    u_hat = mx.fft.rfft2(u, axes=(1, 2))  # [B, N1, N2//2+1] complex

    # Integer wavenumbers matching numpy.fft.fftfreq(N)*N semantics
    kx_int = [k if k <= N1 // 2 else k - N1 for k in range(N1)]
    kx = mx.array(kx_int, dtype=mx.float32).reshape(1, N1, 1) * (2.0 * math.pi)
    ky = (mx.arange(N2 // 2 + 1, dtype=mx.float32).reshape(1, 1, N2 // 2 + 1)
          * (2.0 * math.pi))

    # Multiply by ik: (a + ib)(ik) = -b*k + ia*k
    du_dx_hat_r = -u_hat.imag * kx
    du_dx_hat_i =  u_hat.real * kx
    du_dy_hat_r = -u_hat.imag * ky
    du_dy_hat_i =  u_hat.real * ky

    du_dx = mx.fft.irfft2(du_dx_hat_r + 1j * du_dx_hat_i, s=(N1, N2), axes=(1, 2))
    du_dy = mx.fft.irfft2(du_dy_hat_r + 1j * du_dy_hat_i, s=(N1, N2), axes=(1, 2))
    return du_dx, du_dy


# ── Core loss implementations ─────────────────────────────────────────────────

def relative_l2(pred: mx.array, y: mx.array) -> mx.array:
    """Relative L2 loss (current default).

    mean_batch( ‖pred − y‖₂ / ‖y‖₂ )
    """
    axes      = tuple(range(1, y.ndim))
    diff      = pred - y
    data_loss = mx.mean(
        mx.sqrt(mx.mean(diff ** 2, axis=axes))
        / (mx.sqrt(mx.mean(y   ** 2, axis=axes)) + 1e-8)
    )
    return data_loss


def h1_loss(pred: mx.array, y: mx.array, alpha: float = 0.1) -> mx.array:
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
        grad_loss = mx.mean(
            mx.sqrt(mx.mean(d_diff ** 2, axis=(1,)))
            / (mx.sqrt(mx.mean(d_y ** 2, axis=(1,))) + 1e-8)
        )
    elif y.ndim == 3:
        # 2D path: compute ∂/∂x and ∂/∂y gradients
        diff  = pred - y
        dx, dy     = _spectral_grad_2d(diff)
        dx_y, dy_y = _spectral_grad_2d(y)
        diff_norm = mx.sqrt(mx.mean(dx ** 2 + dy ** 2, axis=(1, 2)))
        y_norm    = mx.sqrt(mx.mean(dx_y ** 2 + dy_y ** 2, axis=(1, 2)))
        grad_loss = mx.mean(diff_norm / (y_norm + 1e-8))
    else:
        # Higher-dimensional: fall back to L2 (multi-channel 2D etc.)
        return l2

    return l2 + alpha * grad_loss


def h1_strong_loss(pred: mx.array, y: mx.array) -> mx.array:
    """H1 loss with α=1.0 — equal weighting of L2 and H1 terms."""
    return h1_loss(pred, y, alpha=1.0)


def h2_loss(pred: mx.array, y: mx.array,
            alpha: float = 0.1, beta: float = 0.01) -> mx.array:
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
    h1t  = mx.mean(
        mx.sqrt(mx.mean(d1 ** 2, axis=axes))
        / (mx.sqrt(mx.mean(d1_y ** 2, axis=axes)) + 1e-8)
    )
    h2t  = mx.mean(
        mx.sqrt(mx.mean(d2 ** 2, axis=axes))
        / (mx.sqrt(mx.mean(d2_y ** 2, axis=axes)) + 1e-8)
    )
    return l2 + alpha * h1t + beta * h2t


def spectral_loss(pred: mx.array, y: mx.array,
                  high_freq_weight: float = 2.0) -> mx.array:
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
        k        = mx.arange(N // 2 + 1, dtype=mx.float32)
        weights  = 1.0 + (k / (N // 2)) ** high_freq_weight  # [N//2+1]

        pred_ft  = mx.fft.rfft(pred, axis=1)
        y_ft     = mx.fft.rfft(y,    axis=1)
        diff_ft  = pred_ft - y_ft

        diff_mag = diff_ft.real ** 2 + diff_ft.imag ** 2      # [B, N//2+1]
        y_mag    = y_ft.real    ** 2 + y_ft.imag    ** 2

        w_err = mx.mean(weights[None, :] * diff_mag, axis=1)   # [B]
        w_nrm = mx.mean(weights[None, :] * y_mag,   axis=1)
        return mx.mean(mx.sqrt(w_err) / (mx.sqrt(w_nrm) + 1e-8))

    elif y.ndim == 3:
        # 2D path: weight by 2D wavenumber magnitude |k|²
        B, N1, N2 = y.shape
        kx_int = [k if k <= N1 // 2 else k - N1 for k in range(N1)]
        kx = mx.array(kx_int, dtype=mx.float32).reshape(N1, 1)
        ky = mx.arange(N2 // 2 + 1, dtype=mx.float32).reshape(1, N2 // 2 + 1)
        k_norm = mx.sqrt(kx ** 2 + ky ** 2) / (max(N1, N2) // 2)  # [N1, N2//2+1]
        weights = (1.0 + k_norm ** high_freq_weight).reshape(1, N1, N2 // 2 + 1)

        pred_ft = mx.fft.rfft2(pred, axes=(1, 2))
        y_ft    = mx.fft.rfft2(y,    axes=(1, 2))
        diff_ft = pred_ft - y_ft

        diff_mag = diff_ft.real ** 2 + diff_ft.imag ** 2
        y_mag    = y_ft.real    ** 2 + y_ft.imag    ** 2

        w_err = mx.mean(weights * diff_mag, axis=(1, 2))   # [B]
        w_nrm = mx.mean(weights * y_mag,   axis=(1, 2))
        return mx.mean(mx.sqrt(w_err) / (mx.sqrt(w_nrm) + 1e-8))

    else:
        return relative_l2(pred, y)


def adaptive_h1_loss(pred: mx.array, y: mx.array, base_alpha: float = 0.1) -> mx.array:
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
        grad_loss = mx.mean(
            mx.sqrt(mx.mean(d_diff ** 2, axis=(1,)))
            / (mx.sqrt(mx.mean(d_y ** 2, axis=(1,))) + 1e-8)
        )
    elif y.ndim == 3:
        diff  = pred - y
        dx, dy_g   = _spectral_grad_2d(diff)
        dx_y, dy_y = _spectral_grad_2d(y)
        diff_norm = mx.sqrt(mx.mean(dx ** 2 + dy_g ** 2, axis=(1, 2)))
        y_norm    = mx.sqrt(mx.mean(dx_y ** 2 + dy_y ** 2, axis=(1, 2)))
        grad_loss = mx.mean(diff_norm / (y_norm + 1e-8))
    else:
        return l2

    # Auto-scale: target_ratio=0.3 means grad term = 30% of total
    # alpha = (target_ratio / (1 - target_ratio)) * (l2 / grad_loss)
    # Use stop_gradient so we don't optimise the scale itself
    auto_alpha = mx.stop_gradient(0.3 / 0.7 * l2 / (grad_loss + 1e-8)) * base_alpha
    auto_alpha = mx.minimum(auto_alpha, 5.0 * base_alpha)  # cap to avoid instability
    return l2 + auto_alpha * grad_loss


def relative_l1(pred: mx.array, y: mx.array) -> mx.array:
    """Relative L1 loss.  Robust to outlier samples."""
    axes = tuple(range(1, y.ndim))
    return mx.mean(
        mx.mean(mx.abs(pred - y), axis=axes)
        / (mx.mean(mx.abs(y), axis=axes) + 1e-8)
    )


def mse_loss(pred: mx.array, y: mx.array) -> mx.array:
    """Plain MSE — no normalisation.  Use for debugging only."""
    return mx.mean((pred - y) ** 2)


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
