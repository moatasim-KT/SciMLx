"""2D Non-linear Schrödinger (NLS) — Soliton dynamics.

PDE:
    i ψ_t + ∇²ψ + κ |ψ|² ψ = 0
    
κ: Non-linearity coefficient.

Models dispersive waves and soliton interactions in quantum fluids.
"""

import numpy as np

def solve_nls_2d(batch_size: int = 1, res: int = 64, kappa: float = 1.0, T: float = 0.1):
    """
    Solves 2D NLS using split-step spectral method.
    """
    N = res
    L = 2 * np.pi
    dx = L / N
    dt = 0.001
    n_steps = int(T / dt)
    
    # Mesh
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    
    # Wavenumbers
    kx = np.fft.fftfreq(N, d=dx/L).reshape(1, N) * (2*np.pi/L)
    ky = np.fft.fftfreq(N, d=dx/L).reshape(N, 1) * (2*np.pi/L)
    k2 = kx**2 + ky**2
    
    # IC: Sum of Gaussian wave packets
    psi = np.zeros((batch_size, N, N), dtype=np.complex128)
    for b in range(batch_size):
        psi[b] = np.exp(-((X-L/2)**2 + (Y-L/2)**2)/0.5) * np.exp(1j * (X + Y))
        
    for _ in range(n_steps):
        # 1. Non-linear step (spatial space)
        # dψ/dt = i κ |ψ|² ψ  => ψ = ψ * exp(i κ |ψ|² dt)
        psi = psi * np.exp(1j * kappa * np.abs(psi)**2 * dt)
        
        # 2. Linear step (spectral space)
        # dψ/dt = i ∇²ψ => ψ_hat = ψ_hat * exp(-i |k|² dt)
        psi_hat = np.fft.fft2(psi, axes=(1, 2))
        psi_hat = psi_hat * np.exp(-1j * k2 * dt)
        psi = np.fft.ifft2(psi_hat, axes=(1, 2))
        
    # Return absolute value (magnitude) as real field for FNO
    return np.abs(psi).astype(np.float32)

METADATA = {
    "pde": "Non-linear Schrödinger",
    "kappa": 1.0,
    "solver": "Split-step spectral"
}
