"""SciML Diagnostics — Advanced metrics and visualizations for scientific discovery.

Extracts spectral bias, spatial error distribution, and generates comparison PNGs.
"""

import numpy as np
import mlx.core as mx
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

from utils import FIGS_DIR

def calculate_spectral_bias(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Calculate error magnitude across Fourier modes."""
    if pred.ndim == 1:
        pred = pred[None, :]
        truth = truth[None, :]
    
    # RFFT magnitudes
    p_ft = np.abs(np.fft.rfft(pred, axis=-1))
    t_ft = np.abs(np.fft.rfft(truth, axis=-1))
    
    # Mean error per mode
    err_ft = np.abs(p_ft - t_ft).mean(axis=0)
    total_energy = t_ft.mean(axis=0).sum() + 1e-8
    
    # Cumulative error in low vs high frequencies
    n_modes = len(err_ft)
    low_cutoff = n_modes // 4
    high_cutoff = n_modes // 2
    
    low_err = err_ft[:low_cutoff].sum() / total_energy
    mid_err = err_ft[low_cutoff:high_cutoff].sum() / total_energy
    high_err = err_ft[high_cutoff:].sum() / total_energy
    
    return {
        "low_freq_error": float(low_err),
        "mid_freq_error": float(mid_err),
        "high_freq_error": float(high_err),
        "spectral_gap": float(np.max(err_ft))
    }

def generate_experiment_comparison(exp_id: str, 
                                   inputs: np.ndarray, 
                                   truth: np.ndarray, 
                                   pred: np.ndarray,
                                   benchmark: str):
    """Generate side-by-side PNG for the Experiment Inspector."""
    is_1d = truth.ndim == 2 # [B, N]
    
    fig_name = f"inspect_{exp_id}"
    fig_path = FIGS_DIR / f"{fig_name}.png"
    
    if is_1d:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        x = np.linspace(0, 1, truth.shape[1])
        
        # Sample 0
        axes[0].plot(x, truth[0], 'k-', label='Truth')
        axes[0].plot(x, pred[0], 'r--', label='Pred')
        axes[0].set_title("Solution Comparison")
        axes[0].legend()
        
        # Spatial Error
        axes[1].plot(x, np.abs(truth[0] - pred[0]), 'crimson')
        axes[1].set_title("Spatial Error |y - ŷ|")
        
        # Spectral Error
        p_ft = np.abs(np.fft.rfft(pred[0]))
        t_ft = np.abs(np.fft.rfft(truth[0]))
        axes[2].bar(range(len(p_ft)), np.abs(p_ft - t_ft), color='indigo')
        axes[2].set_title("Spectral Error Magnitude")
        axes[2].set_yscale('log')
    else:
        # 2D case
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        im0 = axes[0].imshow(truth[0], cmap='viridis')
        axes[0].set_title("Ground Truth")
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(pred[0], cmap='viridis')
        axes[1].set_title("Prediction")
        plt.colorbar(im1, ax=axes[1])
        
        err = np.abs(truth[0] - pred[0])
        im2 = axes[2].imshow(err, cmap='inferno')
        axes[2].set_title("Absolute Error")
        plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    fig.savefig(fig_path, dpi=100)
    plt.close(fig)
    return str(fig_path)

if __name__ == "__main__":
    # Smoke test
    t = np.random.randn(1, 64)
    p = t + 0.1 * np.random.randn(1, 64)
    print("Spectral Bias Test:", calculate_spectral_bias(p, t))
