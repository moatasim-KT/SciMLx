"""Modular loss function library for SciML experiments (PyTorch/CUDA)."""

import math
import torch
import torch.nn.functional as F
from core.device import DEVICE

# ── Spectral derivative helpers ───────────────────────────────────────────────

def spectral_grad_1d(u: torch.Tensor) -> torch.Tensor:
    """1D spectral derivative ∂u/∂x via FFT.  u: [B, N] → ∂u/∂x: [B, N]."""
    B, N = u.shape
    k = torch.arange(N // 2 + 1, device=u.device, dtype=torch.float32)
    u_ft = torch.fft.rfft(u, dim=1)
    
    # Multiply by ik: real part = -imag*k, imag part = real*k
    du_ft = torch.complex(-u_ft.imag * k, u_ft.real * k)
    return torch.fft.irfft(du_ft, n=N, dim=1)


def spectral_grad2_1d(u: torch.Tensor) -> torch.Tensor:
    """Second spectral derivative ∂²u/∂x².  u: [B, N] → [B, N]."""
    B, N = u.shape
    k = torch.arange(N // 2 + 1, device=u.device, dtype=torch.float32)
    u_ft = torch.fft.rfft(u, dim=1)
    
    # Multiply by -k²
    du_ft = -(k ** 2) * u_ft
    return torch.fft.irfft(du_ft, n=N, dim=1)


def _spectral_grad_2d(u: torch.Tensor):
    """2D spectral gradients ∂u/∂x and ∂u/∂y.
    u: [B, N1, N2] — single-channel 2D field.
    """
    B, N1, N2 = u.shape
    u_hat = torch.fft.rfft2(u, dim=(1, 2))

    # Integer wavenumbers matching numpy.fft.fftfreq(N)*N semantics
    kx_int = torch.tensor([k if k <= N1 // 2 else k - N1 for k in range(N1)], device=u.device, dtype=torch.float32)
    kx = kx_int.view(1, N1, 1) * (2.0 * math.pi)
    ky = torch.arange(N2 // 2 + 1, device=u.device, dtype=torch.float32).view(1, 1, N2 // 2 + 1) * (2.0 * math.pi)

    # Multiply by ik
    du_dx_hat = torch.complex(-u_hat.imag * kx, u_hat.real * kx)
    du_dy_hat = torch.complex(-u_hat.imag * ky, u_hat.real * ky)

    du_dx = torch.fft.irfft2(du_dx_hat, s=(N1, N2), dim=(1, 2))
    du_dy = torch.fft.irfft2(du_dy_hat, s=(N1, N2), dim=(1, 2))
    return du_dx, du_dy


# ── Core loss implementations ─────────────────────────────────────────────────

def relative_l2(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Relative L2 loss."""
    dims = tuple(range(1, y.ndim))
    diff_norm = torch.sqrt(torch.mean((pred - y) ** 2, dim=dims))
    y_norm = torch.sqrt(torch.mean(y ** 2, dim=dims))
    return torch.mean(diff_norm / (y_norm + 1e-8))


def h1_loss(pred: torch.Tensor, y: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """H1 Sobolev loss: relative L2 + α · relative L2 of first derivative."""
    l2 = relative_l2(pred, y)

    if y.ndim == 2:
        diff = pred - y
        d_diff = spectral_grad_1d(diff)
        d_y = spectral_grad_1d(y)
        grad_loss = torch.mean(
            torch.sqrt(torch.mean(d_diff ** 2, dim=1))
            / (torch.sqrt(torch.mean(d_y ** 2, dim=1)) + 1e-8)
        )
    elif y.ndim == 3:
        diff = pred - y
        dx, dy = _spectral_grad_2d(diff)
        dx_y, dy_y = _spectral_grad_2d(y)
        diff_norm = torch.sqrt(torch.mean(dx ** 2 + dy ** 2, dim=(1, 2)))
        y_norm = torch.sqrt(torch.mean(dx_y ** 2 + dy_y ** 2, dim=(1, 2)))
        grad_loss = torch.mean(diff_norm / (y_norm + 1e-8))
    else:
        return l2

    return l2 + alpha * grad_loss


def h1_strong_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return h1_loss(pred, y, alpha=1.0)


def h2_loss(pred: torch.Tensor, y: torch.Tensor, alpha: float = 0.1, beta: float = 0.01) -> torch.Tensor:
    if y.ndim != 2:
        return h1_loss(pred, y, alpha=alpha)

    diff = pred - y
    d1 = spectral_grad_1d(diff)
    d2 = spectral_grad2_1d(diff)
    d1_y = spectral_grad_1d(y)
    d2_y = spectral_grad2_1d(y)

    l2 = relative_l2(pred, y)
    h1t = torch.mean(torch.sqrt(torch.mean(d1 ** 2, dim=1)) / (torch.sqrt(torch.mean(d1_y ** 2, dim=1)) + 1e-8))
    h2t = torch.mean(torch.sqrt(torch.mean(d2 ** 2, dim=1)) / (torch.sqrt(torch.mean(d2_y ** 2, dim=1)) + 1e-8))
    return l2 + alpha * h1t + beta * h2t


def spectral_loss(pred: torch.Tensor, y: torch.Tensor, high_freq_weight: float = 2.0) -> torch.Tensor:
    if y.ndim == 2:
        B, N = y.shape
        k = torch.arange(N // 2 + 1, device=y.device, dtype=torch.float32)
        weights = 1.0 + (k / (N // 2)) ** high_freq_weight

        pred_ft = torch.fft.rfft(pred, dim=1)
        y_ft = torch.fft.rfft(y, dim=1)
        diff_ft = pred_ft - y_ft

        diff_mag = diff_ft.abs() ** 2
        y_mag = y_ft.abs() ** 2

        w_err = torch.mean(weights * diff_mag, dim=1)
        w_nrm = torch.mean(weights * y_mag, dim=1)
        return torch.mean(torch.sqrt(w_err) / (torch.sqrt(w_nrm) + 1e-8))
    elif y.ndim == 3:
        B, N1, N2 = y.shape
        kx_int = torch.tensor([k if k <= N1 // 2 else k - N1 for k in range(N1)], device=y.device, dtype=torch.float32)
        kx = kx_int.view(N1, 1)
        ky = torch.arange(N2 // 2 + 1, device=y.device, dtype=torch.float32).view(1, N2 // 2 + 1)
        k_norm = torch.sqrt(kx ** 2 + ky ** 2) / (max(N1, N2) // 2)
        weights = 1.0 + k_norm ** high_freq_weight

        pred_ft = torch.fft.rfft2(pred, dim=(1, 2))
        y_ft = torch.fft.rfft2(y, dim=(1, 2))
        diff_ft = pred_ft - y_ft

        diff_mag = diff_ft.abs() ** 2
        y_mag = y_ft.abs() ** 2

        w_err = torch.mean(weights * diff_mag, dim=(1, 2))
        w_nrm = torch.mean(weights * y_mag, dim=(1, 2))
        return torch.mean(torch.sqrt(w_err) / (torch.sqrt(w_nrm) + 1e-8))
    return relative_l2(pred, y)


def adaptive_h1_loss(pred: torch.Tensor, y: torch.Tensor, base_alpha: float = 0.1) -> torch.Tensor:
    l2 = relative_l2(pred, y)
    
    # Calculate grad_loss similarly to h1_loss but with detached l2 for alpha scaling
    with torch.no_grad():
        if y.ndim == 2:
            diff = pred - y
            d_diff = spectral_grad_1d(diff)
            d_y = spectral_grad_1d(y)
            grad_loss = torch.mean(torch.sqrt(torch.mean(d_diff ** 2, dim=1)) / (torch.sqrt(torch.mean(d_y ** 2, dim=1)) + 1e-8))
        elif y.ndim == 3:
            diff = pred - y
            dx, dy = _spectral_grad_2d(diff)
            dx_y, dy_y = _spectral_grad_2d(y)
            diff_norm = torch.sqrt(torch.mean(dx ** 2 + dy ** 2, dim=(1, 2)))
            y_norm = torch.sqrt(torch.mean(dx_y ** 2 + dy_y ** 2, dim=(1, 2)))
            grad_loss = torch.mean(diff_norm / (y_norm + 1e-8))
        else:
            return l2

    auto_alpha = (0.3 / 0.7 * l2.detach() / (grad_loss + 1e-8)) * base_alpha
    auto_alpha = torch.clamp(auto_alpha, max=5.0 * base_alpha)
    
    # Now re-calculate grad_loss with gradients
    if y.ndim == 2:
        diff = pred - y
        d_diff = spectral_grad_1d(diff)
        d_y = spectral_grad_1d(y)
        grad_loss = torch.mean(torch.sqrt(torch.mean(d_diff ** 2, dim=1)) / (torch.sqrt(torch.mean(d_y ** 2, dim=1)) + 1e-8))
    else:
        diff = pred - y
        dx, dy = _spectral_grad_2d(diff)
        dx_y, dy_y = _spectral_grad_2d(y)
        diff_norm = torch.sqrt(torch.mean(dx ** 2 + dy ** 2, dim=(1, 2)))
        y_norm = torch.sqrt(torch.mean(dx_y ** 2 + dy_y ** 2, dim=(1, 2)))
        grad_loss = torch.mean(diff_norm / (y_norm + 1e-8))

    return l2 + auto_alpha * grad_loss


def relative_l1(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, y.ndim))
    return torch.mean(torch.mean(torch.abs(pred - y), dim=dims) / (torch.mean(torch.abs(y), dim=dims) + 1e-8))


def mse_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, y)


_LOSS_REGISTRY = {
    "l2_rel": relative_l2,
    "h1": h1_loss,
    "h1_strong": h1_strong_loss,
    "h1_adaptive": adaptive_h1_loss,
    "h2": h2_loss,
    "spectral": spectral_loss,
    "l1_rel": relative_l1,
    "mse": mse_loss,
}

def get_loss_fn(name: str, **kwargs):
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(_LOSS_REGISTRY.keys())}")
    fn = _LOSS_REGISTRY[name]
    if kwargs:
        import functools
        return functools.partial(fn, **kwargs)
    return fn

def list_losses():
    return sorted(_LOSS_REGISTRY.keys())
