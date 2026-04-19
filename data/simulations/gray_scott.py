"""2D Gray-Scott — Reaction-diffusion pattern formation.

PDE:
    u_t = Du ∇²u - uv² + f(1-u)
    v_t = Dv ∇²v + uv² - (f+k)v

Models chemical reactions that produce complex Turing patterns.
"""

import numpy as np

def solve_gray_scott_2d(batch_size: int = 1, res: int = 64, Du: float = 0.16, Dv: float = 0.08, f: float = 0.035, k: float = 0.06, T: float = 100.0):
    """
    Solves 2D Gray-Scott.
    """
    N = res
    dx = 1.0
    dt = 1.0
    n_steps = int(T / dt)
    
    # Mesh
    u = np.ones((batch_size, N, N))
    v = np.zeros((batch_size, N, N))
    
    # IC: Central square seed
    for b in range(batch_size):
        u[b, N//2-5:N//2+5, N//2-5:N//2+5] = 0.5
        v[b, N//2-5:N//2+5, N//2-5:N//2+5] = 0.25
        # Add some noise
        u[b] += np.random.rand(N, N) * 0.05
        v[b] += np.random.rand(N, N) * 0.05
        
    # Spectral Laplacians
    kx = np.fft.fftfreq(N).reshape(1, N) * (2*np.pi)
    ky = np.fft.fftfreq(N).reshape(N, 1) * (2*np.pi)
    k2 = -(kx**2 + ky**2)
    
    for _ in range(n_steps):
        u_hat = np.fft.fft2(u, axes=(1, 2))
        v_hat = np.fft.fft2(v, axes=(1, 2))
        
        # Diffusion in spectral space
        diff_u = np.real(np.fft.ifft2(Du * k2 * u_hat))
        diff_v = np.real(np.fft.ifft2(Dv * k2 * v_hat))
        
        # Reaction
        uv2 = u * v**2
        du = diff_u - uv2 + f * (1.0 - u)
        dv = diff_v + uv2 - (f + k) * v
        
        u = u + dt * du
        v = v + dt * dv
        
    return u.astype(np.float32), v.astype(np.float32)

METADATA = {
    "pde": "Gray-Scott (Reaction-Diffusion)",
    "Du": 0.16,
    "Dv": 0.08,
    "solver": "Spectral-explicit"
}
