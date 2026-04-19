"""Classic Fluid Mechanics — Analytical solvers for Couette and Poiseuille flow.

This module provides exact solutions for fundamental viscous flows:
1. Couette Flow: Shear-driven flow between parallel plates.
2. Poiseuille Flow: Pressure-driven flow between parallel plates.

ML interface: Boundary/Pressure parameters → velocity profile u(y).
"""

import numpy as np

def generate_couette_data(batch_size: int = 100, n_points: int = 64):
    """
    u(y) = U * y / L
    Inputs: U (top plate velocity) ~ [0.1, 1.0]
    """
    y = np.linspace(0, 1, n_points)
    U = np.random.uniform(0.1, 1.0, (batch_size, 1))
    
    # [B, N]
    u = U * y[None, :]
    
    # Return (params, profile)
    return U.astype(np.float32), u.astype(np.float32)

def generate_poiseuille_data(batch_size: int = 100, n_points: int = 64):
    """
    u(y) = (G / 2*mu) * (y * (L - y))
    Inputs: G (pressure gradient) ~ [1.0, 10.0]
    """
    y = np.linspace(0, 1, n_points)
    G = np.random.uniform(1.0, 10.0, (batch_size, 1))
    mu = 0.1 # fixed viscosity
    L = 1.0
    
    # [B, N]
    u = (G / (2 * mu)) * (y[None, :] * (L - y[None, :]))
    
    return G.astype(np.float32), u.astype(np.float32)

METADATA = {
    "problems": ["couette_flow_1d", "poiseuille_flow_1d"],
    "domain": "y ∈ [0, 1]",
    "solver": "Analytical exact solutions"
}
