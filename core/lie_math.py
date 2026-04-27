import numpy as np
from scipy.linalg import expm, logm
from typing import Union, Optional

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Map a 3D vector to a skew-symmetric matrix in so(3).
    """
    if v.shape != (3,):
        raise ValueError("Input vector must be 3D.")
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def vee_operator(phi: np.ndarray) -> np.ndarray:
    """
    Inverse of skew_symmetric (the vee operator).
    """
    return np.array([phi[2, 1], phi[0, 2], phi[1, 0]])

class LieLatentSpace:
    """
    A latent space that operates on a Lie Group (e.g., SO(3)).
    Useful for equivariant neural networks and physics-informed models.
    """
    def __init__(self, group_type: str = "SO3"):
        self.group_type = group_type

    def exp(self, v: np.ndarray) -> np.ndarray:
        """
        Exponential map from Lie Algebra to Lie Group.
        """
        if self.group_type == "SO3":
            if v.ndim == 1:
                return expm(skew_symmetric(v))
            else:
                # Batch processing
                return np.stack([expm(skew_symmetric(vi)) for vi in v])
        raise NotImplementedError(f"Group {self.group_type} not implemented.")

    def log(self, R: np.ndarray) -> np.ndarray:
        """
        Logarithmic map from Lie Group to Lie Algebra.
        """
        if self.group_type == "SO3":
            if R.ndim == 2:
                return vee_operator(logm(R))
            else:
                return np.stack([vee_operator(logm(Ri)) for Ri in R])
        raise NotImplementedError(f"Group {self.group_type} not implemented.")

    def project_latent(self, z: np.ndarray) -> np.ndarray:
        """
        Project a generic latent vector onto the Lie Group.
        """
        # For SO(3), we might assume z is a 3D vector in the tangent space at identity.
        return self.exp(z)
