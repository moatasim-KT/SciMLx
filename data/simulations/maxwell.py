"""2D Maxwell's Equations — TM Mode spectral solver.

PDE:
    ∂Ez/∂t = (1/ε) (∂Hy/∂x - ∂Hx/∂y)
    ∂Hx/∂t = -(1/μ) ∂Ez/∂y
    ∂Hy/∂t = (1/μ) ∂Ez/∂x

Models electromagnetic wave propagation in vacuum (ε=1, μ=1).
"""

import numpy as np

def solve_maxwell_2d(batch_size: int = 1, res: int = 64, T: float = 0.5):
    """
    Solves 2D Maxwell (TM mode) via spectral method.
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
    
    # IC: Gaussian pulse in Ez
    Ez = np.zeros((batch_size, N, N), dtype=np.float64)
    Hx = np.zeros((batch_size, N, N), dtype=np.float64)
    Hy = np.zeros((batch_size, N, N), dtype=np.float64)
    
    for b in range(batch_size):
        Ez[b] = np.exp(-((X-L/2)**2 + (Y-L/2)**2)/0.2)
        
    Ez_hat = np.fft.fft2(Ez, axes=(1, 2))
    Hx_hat = np.fft.fft2(Hx, axes=(1, 2))
    Hy_hat = np.fft.fft2(Hy, axes=(1, 2))
    
    for _ in range(n_steps):
        # 1. Update H (explicit)
        # dHx/dt = -dEz/dy
        # dHy/dt = dEz/dx
        Hx_hat = Hx_hat - dt * (1j * ky * Ez_hat)
        Hy_hat = Hy_hat + dt * (1j * kx * Ez_hat)
        
        # 2. Update Ez (explicit)
        # dEz/dt = dHy/dx - dHx/dy
        Ez_hat = Ez_hat + dt * (1j * kx * Hy_hat - 1j * ky * Hx_hat)
        
    return Ez.astype(np.float32), np.real(np.fft.ifft2(Ez_hat)).astype(np.float32)

METADATA = {
    "pde": "Maxwell TM Mode",
    "domain": "Vacuum (ε=1, μ=1)",
    "solver": "Spectral-explicit"
}
