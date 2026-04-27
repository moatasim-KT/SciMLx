import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Optional

def compute_laplacian(vertices: np.ndarray, faces: np.ndarray):
    """
    Compute the cotangent Laplace-Beltrami operator for a triangle mesh.
    Returns:
        L: Stiffness matrix (negative semi-definite)
        M: Mass matrix (diagonal)
    """
    n = vertices.shape[0]
    m = faces.shape[0]
    
    # Vertices of each face
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Edge vectors
    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0
    
    # Edge lengths squared
    l0 = np.sum(e0**2, axis=1)
    l1 = np.sum(e1**2, axis=1)
    l2 = np.sum(e2**2, axis=1)
    
    # Areas (using Heron's formula or cross product)
    # Area = 0.5 * |e1 x e2|
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    
    # Cotangents
    # cot(alpha) = (l1 + l2 - l0) / (4 * Area)
    cot0 = (l1 + l2 - l0) / (4 * areas + 1e-12)
    cot1 = (l0 + l2 - l1) / (4 * areas + 1e-12)
    cot2 = (l0 + l1 - l2) / (4 * areas + 1e-12)
    
    # Stiffness matrix
    I = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1]])
    J = np.concatenate([faces[:, 2], faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 1], faces[:, 0]])
    V = np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2]) * 0.5
    
    L = csr_matrix((V, (I, J)), shape=(n, n))
    L = L - csr_matrix((np.array(L.sum(axis=1)).flatten(), (np.arange(n), np.arange(n))), shape=(n, n))
    
    # Mass matrix (barycentric)
    M_val = np.concatenate([areas/3, areas/3, areas/3])
    M_idx = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    M = csr_matrix((M_val, (M_idx, M_idx)), shape=(n, n))
    
    return L, M

def compute_hks(vertices: np.ndarray, faces: np.ndarray, num_eigenvalues: int = 100, times: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Heat Kernel Signature (HKS) for a mesh.
    
    Args:
        vertices: (N, 3) vertex coordinates.
        faces: (M, 3) triangle indices.
        num_eigenvalues: Number of eigenvalues to use.
        times: (T,) time scales. If None, automatically determined.
    """
    L, M = compute_laplacian(vertices, faces)
    
    # Solve generalized eigenvalue problem: L phi = lambda M phi
    # Since L is negative semi-definite, we use -L to get positive eigenvalues
    eigenvalues, eigenvectors = eigsh(-L, k=num_eigenvalues, M=M, which='SM')
    
    # Ensure eigenvalues are positive and sorted
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    if times is None:
        # Default time scales
        times = np.logspace(np.log10(1.0/eigenvalues[-1]), np.log10(100.0/eigenvalues[1]), 100)
    
    # HKS(x, t) = sum_i exp(-lambda_i * t) * phi_i(x)^2
    phi2 = eigenvectors**2
    hks = np.zeros((vertices.shape[0], len(times)))
    
    for i, t in enumerate(times):
        hks[:, i] = np.sum(np.exp(-eigenvalues * t) * phi2, axis=1)
        
    # Normalize
    hks /= np.sum(hks, axis=1, keepdims=True)
    
    return hks
