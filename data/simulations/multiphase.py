"""2D Multiphase Flow — Rising bubble surrogate (NS + Cahn-Hilliard).

PDE:
    ω_t + (u·∇)ω = ν∇²ω + (1/Fr²) φ_x
    φ_t + (u·∇)φ = (1/Pe) ∇²φ

    φ: Phase field (1=liquid, 0=bubble)
    Fr: Froude number (buoyancy)
    Pe: Peclet number (interface thickness)

Captures the rising bubble dynamics through phase-field coupling.
"""

import numpy as np

def solve_bubble_2d(batch_size: int = 1, res: int = 64, nu: float = 1e-2, Fr: float = 0.5, Pe: float = 100.0, T: float = 1.0):
    """
    Solves 2D NS coupled with a phase field for a buoyant bubble.
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
    k2[0, 0] = 1.0 
    
    # IC: Spherical bubble in center
    phi = np.ones((batch_size, N, N))
    for b in range(batch_size):
        cx, cy = L/2, L/3 # bubble starts near bottom
        r = L/10
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        phi[b] = np.where(dist < r, 0.0, 1.0)
        
    omega = np.zeros((batch_size, N, N))
    
    phi_hat = np.fft.fft2(phi, axes=(1, 2))
    omega_hat = np.fft.fft2(omega, axes=(1, 2))
    
    for _ in range(n_steps):
        # 1. Velocity
        psi_hat = -omega_hat / k2
        u = np.real(np.fft.ifft2(1j * ky * psi_hat))
        v = np.real(np.fft.ifft2(-1j * kx * psi_hat))
        
        # 2. Advection
        phi_x = np.real(np.fft.ifft2(1j * kx * phi_hat))
        phi_y = np.real(np.fft.ifft2(1j * ky * phi_hat))
        omega_x = np.real(np.fft.ifft2(1j * kx * omega_hat))
        omega_y = np.real(np.fft.ifft2(1j * ky * omega_hat))
        
        adv_phi = u * phi_x + v * phi_y
        adv_omega = u * omega_x + v * omega_y
        
        # 3. Step
        phi_hat = (phi_hat - dt * np.fft.fft2(adv_phi)) / (1.0 + dt * (1.0/Pe) * k2)
        
        # Buoyancy force from density gradient (simplified)
        buoyancy = (1.0 / Fr**2) * (1j * kx * phi_hat)
        omega_hat = (omega_hat + dt * (buoyancy - np.fft.fft2(adv_omega))) / (1.0 + dt * nu * k2)
        
    return phi.astype(np.float32), np.real(np.fft.ifft2(phi_hat)).astype(np.float32)

METADATA = {
    "pde": "Navier-Stokes + Phase-Field",
    "problem": "Rising Bubble 2D",
    "solver": "Spectral"
}
