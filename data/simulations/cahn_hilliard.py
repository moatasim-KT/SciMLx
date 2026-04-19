"""2D Cahn-Hilliard — Phase separation solver.

PDE:
    φ_t = M ∇²(φ³ - φ - ε²∇²φ)
    
    φ: Phase field ∈ [-1, 1]
    M: Mobility
    ε: Interface width

Fourth-order non-linear PDE that tests the model's ability to handle high-order derivatives.
"""

import numpy as np

def solve_cahn_hilliard_2d(batch_size: int = 1, res: int = 64, M: float = 1.0, eps: float = 0.01, T: float = 0.1):
    """
    Solves 2D Cahn-Hilliard via pseudo-spectral semi-implicit scheme.
    """
    N = res
    L = 1.0
    dx = L / N
    dt = 0.0001
    n_steps = int(T / dt)
    
    # Wavenumbers
    k = np.fft.fftfreq(N, d=dx).reshape(1, N) * (2*np.pi)
    kx = np.fft.fftfreq(N, d=dx).reshape(1, N) * (2*np.pi)
    ky = np.fft.fftfreq(N, d=dx).reshape(N, 1) * (2*np.pi)
    k2 = kx**2 + ky**2
    k4 = k2**2
    
    # IC: Random noise near zero
    phi = (np.random.rand(batch_size, N, N) - 0.5) * 0.1
    phi_hat = np.fft.fft2(phi, axes=(1, 2))
    
    for _ in range(n_steps):
        phi = np.real(np.fft.ifft2(phi_hat))
        
        # Chemical potential non-linear term: f(φ) = φ³ - φ
        f_phi = phi**3 - phi
        f_hat = np.fft.fft2(f_phi)
        
        # Semi-implicit step:
        # Linear part (ε²∇⁴φ) is implicit, non-linear part is explicit.
        # dφ/dt = M∇²(f(φ)) - Mε²∇⁴φ
        phi_hat = (phi_hat - dt * M * k2 * f_hat) / (1.0 + dt * M * eps**2 * k4)
        
    return phi.astype(np.float32), np.real(np.fft.ifft2(phi_hat)).astype(np.float32)

METADATA = {
    "pde": "Cahn-Hilliard (4th-order)",
    "mobility": 1.0,
    "epsilon": 0.01,
    "solver": "Semi-implicit spectral"
}
