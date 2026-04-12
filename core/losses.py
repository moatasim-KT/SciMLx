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
    "h1"         – H1 Sobolev: L2 + α·L2(∂u/∂x)  (targets shock fronts)
    "h1_strong"  – H1 with α=1.0
    "spectral"   – frequency-weighted L2 (emphasises high-k errors)
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

def _spectral_grad_1d(u: mx.array) -> mx.array:
    """1D spectral derivative ∂u/∂x via FFT.  u: [B, N] → ∂u/∂x: [B, N]."""
    B, N = u.shape
    k    = mx.arange(N // 2 + 1, dtype=mx.float32)
    u_ft = mx.fft.rfft(u, axis=1)
    # Multiply by ik: real part = -imag*k, imag part = real*k
    du_ft_r = -u_ft.imag * k[None, :]
    du_ft_i =  u_ft.real * k[None, :]
    return mx.fft.irfft(du_ft_r + 1j * du_ft_i, n=N, axis=1)


def _spectral_grad2_1d(u: mx.array) -> mx.array:
    """Second spectral derivative ∂²u/∂x².  u: [B, N] → [B, N]."""
    B, N = u.shape
    k    = mx.arange(N // 2 + 1, dtype=mx.float32)
    u_ft = mx.fft.rfft(u, axis=1)
    # Multiply by -k²
    d2u_ft_r = -(k ** 2)[None, :] * u_ft.real
    d2u_ft_i = -(k ** 2)[None, :] * u_ft.imag
    return mx.fft.irfft(d2u_ft_r + 1j * d2u_ft_i, n=N, axis=1)


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

    Args:
        pred:  [B, N]  model prediction
        y:     [B, N]  ground truth
        alpha: weight on the derivative term (default 0.1)
    """
    if y.ndim != 2:
        # Fall back to L2 for 2D — spectral grad is more involved
        return relative_l2(pred, y)

    l2   = relative_l2(pred, y)
    # Derivative of the residual
    diff = pred - y
    d_diff = _spectral_grad_1d(diff)
    d_y    = _spectral_grad_1d(y)
    axes   = (1,)
    grad_loss = mx.mean(
        mx.sqrt(mx.mean(d_diff ** 2, axis=axes))
        / (mx.sqrt(mx.mean(d_y   ** 2, axis=axes)) + 1e-8)
    )
    return l2 + alpha * grad_loss


def h1_strong_loss(pred: mx.array, y: mx.array) -> mx.array:
    """H1 loss with α=1.0 — equal weighting of L2 and H1 terms."""
    return h1_loss(pred, y, alpha=1.0)


def h2_loss(pred: mx.array, y: mx.array,
            alpha: float = 0.1, beta: float = 0.01) -> mx.array:
    """H2 Sobolev loss: L2 + α·H1 + β·H2.

    Adds a second-derivative penalty.  Smooths more aggressively.
    """
    if y.ndim != 2:
        return relative_l2(pred, y)
    axes  = (1,)
    diff  = pred - y
    d1    = _spectral_grad_1d(diff)
    d2    = _spectral_grad2_1d(diff)
    d1_y  = _spectral_grad_1d(y)
    d2_y  = _spectral_grad2_1d(y)

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

    Args:
        high_freq_weight: exponent for frequency weighting (default 2.0)
    """
    if y.ndim != 2:
        return relative_l2(pred, y)

    B, N     = y.shape
    k        = mx.arange(N // 2 + 1, dtype=mx.float32)
    weights  = 1.0 + (k / (N // 2)) ** high_freq_weight  # shape [N//2+1]

    pred_ft  = mx.fft.rfft(pred, axis=1)
    y_ft     = mx.fft.rfft(y,    axis=1)
    diff_ft  = pred_ft - y_ft

    # Weighted L2 in Fourier space (Parseval-like)
    diff_mag = diff_ft.real ** 2 + diff_ft.imag ** 2      # [B, N//2+1]
    y_mag    = y_ft.real    ** 2 + y_ft.imag    ** 2

    w_err  = mx.mean(weights[None, :] * diff_mag, axis=1)   # [B]
    w_nrm  = mx.mean(weights[None, :] * y_mag,   axis=1)
    return mx.mean(mx.sqrt(w_err) / (mx.sqrt(w_nrm) + 1e-8))


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
    "l2_rel":     relative_l2,
    "h1":         h1_loss,
    "h1_strong":  h1_strong_loss,
    "h2":         h2_loss,
    "spectral":   spectral_loss,
    "l1_rel":     relative_l1,
    "mse":        mse_loss,
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
