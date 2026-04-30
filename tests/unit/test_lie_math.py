import pytest
import numpy as np
from core.lie_math import LieLatentSpace, skew_symmetric, vee_operator

def test_skew_symmetric():
    v = np.array([1.0, 2.0, 3.0])
    omega = skew_symmetric(v)
    assert omega[0, 1] == -3.0
    assert omega[0, 2] == 2.0
    assert omega[1, 0] == 3.0
    assert np.allclose(omega, -omega.T)

def test_vee_operator():
    v = np.array([1.0, 2.0, 3.0])
    omega = skew_symmetric(v)
    v_rec = vee_operator(omega)
    assert np.allclose(v, v_rec)

def test_lie_latent_space_so3():
    space = LieLatentSpace(group_type="SO3")
    v = np.array([0.1, 0.0, 0.0])
    R = space.exp(v)
    
    # Check if it's a rotation matrix (orthogonal, det=1)
    assert np.allclose(np.dot(R, R.T), np.eye(3))
    assert np.allclose(np.linalg.det(R), 1.0)
    
    v_rec = space.log(R)
    assert np.allclose(v, v_rec)

def test_batch_lie():
    space = LieLatentSpace(group_type="SO3")
    v = np.random.randn(5, 3) * 0.1
    R = space.exp(v)
    assert R.shape == (5, 3, 3)
    
    v_rec = space.log(R)
    assert np.allclose(v, v_rec)
