import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Optional, List

class HeatKernel:
    """
    Heat Kernel tools for spectral geometry on meshes.
    """
    
    @staticmethod
    def compute_laplacian(vertices: np.ndarray, faces: np.ndarray) -> csr_matrix:
        """
        Compute the cotangent Laplace-Beltrami operator.
        """
        n = vertices.shape[0]
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        
        # Edges
        e0 = v2 - v1
        e1 = v0 - v2
        e2 = v1 - v0
        
        # Cotangents
        def cotan(a, b):
            return np.sum(a * b, axis=1) / np.sqrt(np.sum(np.cross(a, b)**2, axis=1))
            
        cot0 = cotan(-e1, e2)
        cot1 = cotan(-e2, e0)
        cot2 = cotan(-e0, e1)
        
        I = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1]])
        J = np.concatenate([faces[:, 2], faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 1], faces[:, 0]])
        W = np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2]) * 0.5
        
        W_sparse = csr_matrix((W, (I, J)), shape=(n, n))
        diag = np.array(W_sparse.sum(axis=1)).flatten()
        L = csr_matrix((diag, (range(n), range(n))), shape=(n, n)) - W_sparse
        return L

    @staticmethod
    def compute_hks(vertices: np.ndarray, faces: np.ndarray, 
                    num_eigs: int = 100, 
                    times: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Heat Kernel Signature (HKS).
        """
        L = HeatKernel.compute_laplacian(vertices, faces)
        
        # We need the mass matrix for the generalized eigenvalue problem
        # Simple lumped mass matrix
        n = vertices.shape[0]
        areas = np.zeros(n)
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        face_areas = 0.5 * np.sqrt(np.sum(np.cross(v1 - v0, v2 - v0)**2, axis=1))
        for i in range(3):
            np.add.at(areas, faces[:, i], face_areas / 3.0)
        M = csr_matrix((areas, (range(n), range(n))), shape=(n, n))
        
        # Solve generalized eigenvalue problem: L phi = lambda M phi
        evals, evecs = eigsh(L, k=num_eigs, M=M, which='SM')
        
        if times is None:
            # Automatic time scale selection based on eigenvalues
            times = np.logspace(np.log10(4*np.log(10)/evals[-1]), 
                                np.log10(4*np.log(10)/evals[1]), 20)
            
        # HKS(x, t) = sum_i exp(-lambda_i * t) * phi_i(x)^2
        # evecs is (n, num_eigs), evals is (num_eigs,)
        evecs_sq = evecs**2
        
        hks = []
        for t in times:
            weights = np.exp(-evals * t)
            hks.append(np.dot(evecs_sq, weights))
            
        return np.stack(hks, axis=1)
