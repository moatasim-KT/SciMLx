"""
Oracle of Constants — Foundation for Dimensional Analysis and Feature Discovery.

Implements the Buckingham Pi Theorem for identifying dimensionless π-groups
and provides Mutual Information scoring for variable importance.
"""

import numpy as np
from scipy.linalg import null_space
from scipy.stats import entropy
from typing import List, Dict, Optional, Tuple

class OracleOfConstants:
    """
    Oracle of Constants — Dimensional analysis and π-group identification.
    Uses Buckingham Pi Theorem to find dimensionless groups from a set of variables.
    """
    def __init__(self, variable_names: List[str], dimensions: np.ndarray):
        """
        Initializes the Oracle with variables and their physical dimensions.
        
        Args:
            variable_names: List of strings representing variable names (e.g., ['L', 'v', 'rho', 'mu']).
            dimensions: (n_variables, n_fundamental_units) matrix where each row 
                       represents the exponents of fundamental units for that variable.
                       Example: if units are [M, L, T], velocity (L/T) is [0, 1, -1].
        """
        if len(variable_names) != dimensions.shape[0]:
            raise ValueError("Number of variable names must match the number of rows in dimensions matrix.")
            
        self.variable_names = variable_names
        self.dimensions = np.array(dimensions) # (n, k)
        
    def find_pi_groups(self) -> List[Dict[str, float]]:
        """
        Identify the null space of the dimensions matrix to find dimensionless groups.
        
        Returns:
            List of dictionaries, each mapping variable names to their exponent in the π-group.
        """
        # We want x such that dimensions.T @ x = 0
        # dimensions.T is (k, n) where k is units and n is variables.
        ns = null_space(self.dimensions.T)
        
        pi_groups = []
        for i in range(ns.shape[1]):
            vec = ns[:, i]
            
            # Normalize to make coefficients more readable (heuristically)
            abs_vec = np.abs(vec)
            nonzero = abs_vec[abs_vec > 1e-10]
            if len(nonzero) > 0:
                scale = np.min(nonzero)
                vec = vec / scale
            
            # Round to avoid floating point noise from null_space calculation
            vec = np.round(vec, 6)
            
            group = {}
            for j, name in enumerate(self.variable_names):
                if abs(vec[j]) > 1e-8:
                    group[name] = float(vec[j])
            
            if group:
                pi_groups.append(group)
            
        return pi_groups

def mutual_information_score(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """
    Calculate Mutual Information score between two variables using binning.
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    
    Args:
        x: Input array X.
        y: Input array Y.
        bins: Number of bins for histogram estimation.
        
    Returns:
        Estimated Mutual Information score (in nats).
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Compute marginal histograms
    c_x = np.histogram(x, bins=bins)[0]
    c_y = np.histogram(y, bins=bins)[0]
    
    # Compute joint histogram
    c_xy = np.histogram2d(x, y, bins=bins)[0]
    
    # Compute entropies using scipy.stats.entropy (base e by default)
    h_x = entropy(c_x)
    h_y = entropy(c_y)
    h_xy = entropy(c_xy.flatten())
    
    # I(X;Y) = H(X) + H(Y) - H(X,Y)
    mi = h_x + h_y - h_xy
    return max(0.0, float(mi))

class MutualInformationScore:
    """
    Mutual Information Score estimator.
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    def __init__(self, bins: int = 20):
        self.bins = bins
        
    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate the MI score between x and y."""
        return mutual_information_score(x, y, bins=self.bins)
        
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Call method for easy usage as a scoring function."""
        return self.score(x, y)
