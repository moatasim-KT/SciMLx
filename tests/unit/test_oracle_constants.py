import pytest
import numpy as np
from core.oracle_constants import OracleOfConstants, mutual_information_score

def test_buckingham_pi_simple():
    # Variables: velocity (v), length (L), density (rho), viscosity (mu)
    # Units: [M, L, T]
    # v: [0, 1, -1]
    # L: [0, 1, 0]
    # rho: [1, -3, 0]
    # mu: [1, -1, -1]
    
    names = ['v', 'L', 'rho', 'mu']
    dims = np.array([
        [0, 1, -1],  # v
        [0, 1, 0],   # L
        [1, -3, 0],  # rho
        [1, -1, -1]  # mu
    ])
    
    oracle = OracleOfConstants(names, dims)
    groups = oracle.find_pi_groups()
    
    # We expect Reynolds number: rho * v * L / mu
    # [1, -3, 0] + [0, 1, -1] + [0, 1, 0] - [1, -1, -1] = [0, 0, 0]
    # So exponents for [v, L, rho, mu] should be [1, 1, 1, -1] (or a multiple)
    
    assert len(groups) > 0
    # Check if any group represents Reynolds number
    found_re = False
    for g in groups:
        # Normalize by mu's exponent if it exists
        if 'mu' in g:
            scale = -1.0 / g['mu']
            norm_g = {k: v * scale for k, v in g.items()}
            if norm_g.get('v') == 1.0 and norm_g.get('L') == 1.0 and norm_g.get('rho') == 1.0:
                found_re = True
                break
    assert found_re

def test_mutual_information_score():
    x = np.linspace(0, 10, 100)
    y = 2 * x + np.random.normal(0, 0.1, 100)
    
    mi_high = mutual_information_score(x, y)
    
    y_random = np.random.normal(0, 1, 100)
    mi_low = mutual_information_score(x, y_random)
    
    assert mi_high > mi_low
    assert mi_high > 0
    assert mi_low >= 0

def test_oracle_init_error():
    with pytest.raises(ValueError):
        OracleOfConstants(['a', 'b'], np.array([[1, 0]]))
