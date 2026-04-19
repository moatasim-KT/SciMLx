"""2D Kolmogorov Flow — Spectral Navier-Stokes solver with sinusoidal forcing.

PDE:
    ω_t + (u·∇)ω = ν∇²ω + f
    f = -n·cos(n y)  (forcing term for vorticity)
    
Kolmogorov flow is a classic benchmark for chaotic dynamics and turbulence transitions.
"""

import numpy as np

def solve_kolmogorov_2d(batch_size: int = 1, res: int = 64, nu: float = 1e-2, n_forcing: int = 4, T: float = 1.0):
    """
    Solves 2D NS with Kolmogorov forcing.
    f_ω = -n_forcing * cos(n_forcing * y)
    """
    N = res
    L = 2 * np.pi
    dx = L / N
    dt = 0.001
    n_steps = int(T / dt)
    
    # Mesh
    y = np.linspace(0, L, N, endpoint=False).reshape(N, 1)
    
    # Wavenumbers
    kx = np.fft.fftfreq(N, d=dx/L).reshape(1, N) * (2*np.pi/L)
    ky = np.fft.fftfreq(N, d=dx/L).reshape(N, 1) * (2*np.pi/L)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0 
    
    # Kolmogorov Forcing in spectral space
    # f = -n * cos(n*y) = -n * (exp(iny) + exp(-iny))/2
    f_spatial = -n_forcing * np.cos(n_forcing * y)
    f_hat = np.fft.fft2(np.broadcast_to(f_spatial, (batch_size, N, N)), axes=(1, 2))
    
    # IC: Random vorticity
    omega = np.random.randn(batch_size, N, N) * 0.01
    omega_hat = np.fft.fft2(omega, axes=(1, 2))
    
    for _ in range(n_steps):
        # 1. Velocity
        psi_hat = -omega_hat / k2
        u = np.real(np.fft.ifft2(1j * ky * psi_hat))
        v = np.real(np.fft.ifft2(-1j * kx * psi_hat))
        
        # 2. Advection
        omega_x = np.real(np.fft.ifft2(1j * kx * omega_hat))
        omega_y = np.real(np.fft.ifft2(1j * ky * omega_hat))
        adv = u * omega_x + v * omega_y
        
        # 3. Step
        omega_hat = (omega_hat + dt * (f_hat - np.fft.fft2(adv))) / (1.0 + dt * nu * k2)
        
    return omega.astype(np.float32), np.real(np.fft.ifft2(omega_hat)).astype(np.float32)

METADATA = {
    "pde": "2D Kolmogorov Flow",
    "nu": 1e-2,
    "forcing": "n=4 sinusoidal",
    "solver": "Semi-implicit spectral"
}
