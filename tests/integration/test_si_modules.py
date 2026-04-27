import numpy as np
import pytest

# Skip the entire module if torch is not installed
torch = pytest.importorskip("torch")

from models.gato import GATO
from models.mff import MultiFidelityFusion, ResidualMFF
from core.dp_federated import DPSolver

def test_gato_shape():
    B, N, C = 2, 128, 4
    # Create a simple grid-like mesh to ensure non-singular laplacian
    vertices = np.random.rand(N, 3)
    # Ensure all vertices are referenced at least once
    faces = []
    for i in range(N - 2):
        faces.append([i, i+1, i+2])
    faces = np.array(faces)
    
    model = GATO(in_ch=C, hidden_dim=32, out_ch=1, n_layers=2, n_heads=4, vertices=vertices, faces=faces, num_eigs=10)
    x = torch.randn(B, N, C)
    out = model(x)
    print(f"GATO out shape: {out.shape}")
    assert out.shape == (B, N, 1)

def test_mff_shape():
    B, N, C = 2, 64, 4
    low_fi = torch.nn.Linear(C, 1)
    hi_fi = torch.nn.Linear(C + 1, 1)
    
    mff = MultiFidelityFusion(low_fi, hi_fi, backend='torch')
    x = torch.randn(B, N, C)
    out = mff(x)
    print(f"MFF out shape: {out.shape}")
    assert out.shape == (B, N, 1)
    
    delta_net = torch.nn.Linear(C + 1, 1)
    rmff = ResidualMFF(low_fi, delta_net, backend='torch')
    out_res = rmff(x)
    print(f"ResidualMFF out shape: {out_res.shape}")
    assert out_res.shape == (B, N, 1)

def test_dp_solver():
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    solver = DPSolver(max_grad_norm=1.0, noise_multiplier=0.1)
    
    x = torch.randn(1, 10)
    y = model(x)
    y.backward()
    
    # Check gradients before DP step
    orig_grads = [p.grad.clone() for p in model.parameters()]
    
    solver.step(optimizer, list(model.parameters()))
    
    print("DP Solver step completed successfully.")

if __name__ == "__main__":
    test_gato_shape()
    test_mff_shape()
    test_dp_solver()
    print("All shape validations passed!")
