import numpy as np
from scipy.linalg import expm, logm
from typing import Optional, Union

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Map from vector [ω1, ω2, ω3] to skew-symmetric matrix in so(3)."""
    if v.ndim == 1:
        if v.shape[0] != 3:
            raise ValueError("skew_symmetric expects a 3-element vector")
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    elif v.ndim == 2:
        # Batch processing
        batch_size = v.shape[0]
        m = np.zeros((batch_size, 3, 3))
        m[:, 0, 1] = -v[:, 2]
        m[:, 0, 2] = v[:, 1]
        m[:, 1, 0] = v[:, 2]
        m[:, 1, 2] = -v[:, 0]
        m[:, 2, 0] = -v[:, 1]
        m[:, 2, 1] = v[:, 0]
        return m
    raise ValueError("Input must be 1D or 2D array")

def vee_operator(m: np.ndarray) -> np.ndarray:
    """Map from skew-symmetric matrix to vector in R^3."""
    if m.ndim == 2:
        return np.array([m[2, 1], m[0, 2], m[1, 0]])
    elif m.ndim == 3:
        return np.stack([m[:, 2, 1], m[:, 0, 2], m[:, 1, 0]], axis=1)
    raise ValueError("Input must be 2D or 3D array")

class LieLatentSpace:
    """
    High-level Lie Algebra operations in a latent space.
    Supports SO(3) and SE(3).
    """
    def __init__(self, group_type: str = "SO3"):
        self.group_type = group_type.upper()
        if self.group_type not in ["SO3", "SE3"]:
            raise ValueError(f"Unsupported group type: {group_type}")

    def hat(self, v: np.ndarray) -> np.ndarray:
        if self.group_type == "SO3":
            return skew_symmetric(v)
        elif self.group_type == "SE3":
            if v.ndim == 1:
                omega = v[:3]
                translation = v[3:]
                m = np.zeros((4, 4))
                m[:3, :3] = skew_symmetric(omega)
                m[:3, 3] = translation
                return m
            elif v.ndim == 2:
                batch_size = v.shape[0]
                m = np.zeros((batch_size, 4, 4))
                m[:, :3, :3] = skew_symmetric(v[:, :3])
                m[:, :3, 3] = v[:, 3:]
                m[:, 3, 3] = 1.0
                return m
        return NotImplemented

    def vee(self, m: np.ndarray) -> np.ndarray:
        if self.group_type == "SO3":
            return vee_operator(m)
        elif self.group_type == "SE3":
            if m.ndim == 2:
                omega = vee_operator(m[:3, :3])
                translation = m[:3, 3]
                return np.concatenate([omega, translation])
            elif m.ndim == 3:
                omega = vee_operator(m[:, :3, :3])
                translation = m[:, :3, 3]
                return np.concatenate([omega, translation], axis=1)
        return NotImplemented

    def exp(self, v: np.ndarray) -> np.ndarray:
        """Exponential map from Lie Algebra to Lie Group."""
        if v.ndim == 1:
            return expm(self.hat(v))
        elif v.ndim == 2:
            # Batch expm
            m = self.hat(v)
            res = np.stack([expm(mi) for mi in m])
            return res
        raise ValueError("Input must be 1D or 2D")

    def log(self, m: np.ndarray) -> np.ndarray:
        """Logarithmic map from Lie Group to Lie Algebra."""
        if m.ndim == 2:
            return self.vee(logm(m))
        elif m.ndim == 3:
            # Batch logm
            res = np.stack([self.vee(logm(mi)) for mi in m])
            return res
        raise ValueError("Input must be 2D or 3D")
