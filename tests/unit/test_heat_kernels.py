import pytest
import numpy as np
from core.heat_kernels import compute_hks

def test_hks_calculation():
    # Simple tetrahedron mesh
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=float)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 2, 3]
    ])
    
    # We use small number of eigenvalues because it's a small mesh
    hks = compute_hks(vertices, faces, num_eigenvalues=3)
    
    assert hks.shape[0] == 4
    assert hks.shape[1] == 100
    # HKS should be normalized (sum across time = 1 in my implementation)
    assert np.allclose(hks.sum(axis=1), 1.0)

def test_hks_with_times():
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ], dtype=float)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]
    ])
    
    times = np.array([0.1, 0.5, 1.0])
    hks = compute_hks(vertices, faces, num_eigenvalues=3, times=times)
    
    assert hks.shape == (4, 3)
