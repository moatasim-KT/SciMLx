import numpy as np
import pytest

# Skip the entire module if torch is not installed
torch = pytest.importorskip("torch")

from models.gato import GATO

def test_gato_with_hks():
    """Test GATO with pre-computed HKS."""
    n_nodes = 50
    hks_dim = 20
    hks = np.random.rand(n_nodes, hks_dim)
    
    model = GATO(
        in_ch=3, 
        hidden_dim=64, 
        out_ch=1, 
        n_layers=2, 
        n_heads=4, 
        hks=hks
    )
    
    # Input: [Batch, Nodes, Channels]
    x = torch.randn(4, n_nodes, 3)
    out = model(x)
    
    assert out.shape == (4, n_nodes, 1)

def test_gato_geometric_init():
    """Test GATO initialization with mesh (smoke test)."""
    # Create a small sphere-like mesh or just a random one that compute_laplacian can handle
    # Using a simple tetrahedron for speed and stability in tests
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=float)
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ])
    
    # Use few eigenvalues for a tiny mesh
    model = GATO(
        in_ch=1, 
        hidden_dim=16, 
        out_ch=1, 
        vertices=vertices, 
        faces=faces,
        num_eigs=3 # Max eigs for 4 vertices is 4
    )
    
    x = torch.randn(1, 4, 1)
    out = model(x)
    assert out.shape == (1, 4, 1)
    assert model.hks_pe is not None
