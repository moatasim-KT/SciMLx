"""SciML Diagnostics — Advanced metrics and visualizations for scientific discovery.

Extracts spectral bias, spatial error distribution, and generates comparison PNGs.
"""

import numpy as np
import mlx.core as mx
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, List

from core.utils import FIGS_DIR

def calculate_spectral_bias(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Calculate error magnitude across Fourier modes."""
    # Handle optional channel dimension [B, N, 1]
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if truth.ndim == 3 and truth.shape[-1] == 1:
        truth = truth.squeeze(-1)
        
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

def parse_log_file(log_path: Path) -> Dict[str, Any]:
    """Extract metrics, diagnostics, and crash type from a train.py log file."""
    results = {
        "val": None,
        "mem_mb": 0.0,
        "diag": {},
        "inspect_id": None,
        "crash_type": None,
    }
    if not log_path.exists():
        results["crash_type"] = "FileNotFound"
        return results

    try:
        import re as _re
        content = log_path.read_text()
        grad_norms: list = []
        for line in content.splitlines():
            if line.startswith("val_l2_rel:"):
                results["val"] = float(line.split(":")[1].strip())
            elif line.startswith("peak_vram_mb:"):
                results["mem_mb"] = float(line.split(":")[1].strip())
            elif line.startswith("diag_"):
                key = line.split(":")[0].strip()
                try:
                    val_str = line.split(":")[1].strip()
                    if "=" in val_str:
                        # Parse multi-value signature: "key: v1=X v2=Y"
                        parts = val_str.split()
                        for p in parts:
                            if "=" in p:
                                pk, pv = p.split("=", 1)
                                results["diag"][f"{key}_{pk}"] = float(pv)
                    else:
                        results["diag"][key] = float(val_str)
                except (ValueError, IndexError):
                    pass
            elif line.startswith("inspect_id:"):
                results["inspect_id"] = line.split(":", 1)[1].strip()
            else:
                # Extract per-step grad norm logged by trainer every 20 steps:
                # "step XXXXX (XX.X%%) | loss: ... | gnorm: X.XXX | ..."
                m = _re.search(r"gnorm:\s*([\d.]+)", line)
                if m:
                    try:
                        grad_norms.append(float(m.group(1)))
                    except ValueError:
                        pass

        if grad_norms:
            results["diag"]["diag_grad_norm_max"]  = max(grad_norms)
            results["diag"]["diag_grad_norm_mean"] = sum(grad_norms) / len(grad_norms)

        # Classify crash type if no val_l2_rel found
        if results["val"] is None:
            results["crash_type"] = classify_failure(content)
    except Exception:
        results["crash_type"] = "ParseError"
        
    return results

def classify_failure(content: str) -> str:
    """Detailed classification of log content into failure types."""
    lower = content.lower()
    
    # Incompatibility: 1D model on 2D benchmark
    if "too many values to unpack" in lower and ("expected 2" in lower or "expected 3" in lower):
        return "IncompatibleDimensions"
    
    # OOM / VRAM issues
    if "vram limit exceeded" in lower:
        return "VRAMLimit"
    if "out of memory" in lower or "[metal::malloc]" in lower or "alloc" in lower:
        return "OOM"
    
    # Divergence / Numerical
    if "nan" in lower or "inf" in lower or "diverged" in lower:
        return "NaN/Inf"
    
    # Broadcasting / Grid issues
    if "broadcast_shapes" in lower or "cannot be broadcast" in lower:
        return "BroadcastingError"
    
    # Timeouts
    if "timeout" in lower or "timed out" in lower:
        return "Timeout"
    
    # Code issues
    if "importerror" in lower or "modulenotfounderror" in lower:
        return "ImportError"
    if "valueerror" in lower:
        return "ValueError"
    if "assertionerror" in lower:
        return "AssertionError"
    if "runtimeerror" in lower:
        return "RuntimeError"
    
    # Catch-alls
    if "traceback" in lower or "error" in lower:
        return "UnknownError"
    
    return "NoOutput"

def check_early_stop_condition(log_path: Path, baseline: float, multiplier: float = 50.0) -> Optional[float]:
    """Scan log for early-stop conditions (val >> baseline)."""
    if baseline >= float("inf") or baseline <= 0:
        return None
    threshold = baseline * multiplier
    try:
        content = log_path.read_text()
    except Exception:
        return None
    for line in reversed(content.splitlines()):
        if not line.startswith("val@"):
            continue
        try:
            pct_part, val_part = line.split(":", 1)
            pct = int(pct_part[4:].rstrip("%"))
            mid_val = float(val_part.split()[0])
            if pct >= 30 and mid_val > threshold:
                return mid_val
        except (ValueError, IndexError):
            continue
        break
    return None

def get_fix_strategies(crash_type: str, current_config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Map crash types to a list of potential configuration fixes (Cause-to-Fix Map).
    
    Returns a list of (description, field_overrides) tuples.
    """
    fixes = []
    
    if crash_type == "IncompatibleDimensions":
        # Usually unfixable without changing the model/benchmark pairing
        return []

    if crash_type == "VRAMLimit":
        fixes.append((
            "Reduce hidden_dim, layers, and batch_size (VRAM limit)",
            {
                "hidden_dim": max(32, current_config.get("hidden_dim", 64) // 2),
                "n_layers": max(2, current_config.get("n_layers", 4) // 2),
                "batch_size": max(8, current_config.get("batch_size", 32) // 2),
            }
        ))

    if crash_type == "OOM":
        fixes.append((
            "Halve batch_size (OOM)",
            {"batch_size": max(8, current_config.get("batch_size", 32) // 2)}
        ))

    if crash_type == "NaN/Inf":
        fixes.append((
            "Reduce learning rate and add grad clipping (NaN/Inf)",
            {
                "lr": round(current_config.get("lr", 1e-3) / 10, 8),
                "grad_clip": 5.0
            }
        ))

    if crash_type == "BroadcastingError":
        fixes.append((
            "Halve n_modes (Broadcasting Error)",
            {"n_modes": max(4, current_config.get("n_modes", 16) // 2)}
        ))

    if crash_type == "Timeout":
        fixes.append((
            "Reduce model depth and width (Timeout)",
            {
                "hidden_dim": max(32, current_config.get("hidden_dim", 64) // 2),
                "n_layers": max(2, current_config.get("n_layers", 4) // 2),
            }
        ))

    if crash_type in ["ValueError", "RuntimeError", "AssertionError"]:
        # Generic fallback for common errors: reduce modes
        fixes.append((
            "Reduce n_modes (Generic Error Fallback)",
            {"n_modes": max(4, current_config.get("n_modes", 16) // 2)}
        ))

    if not fixes and crash_type not in ["NoOutput", "Killed", "FileNotFound"]:
        # Catch-all unknown error fix
        fixes.append((
            "General model size reduction (Unknown Error)",
            {
                "hidden_dim": max(32, current_config.get("hidden_dim", 64) // 2),
                "n_layers": max(2, current_config.get("n_layers", 4) // 2),
            }
        ))

    return fixes

class OracleOfConstants:
    """
    Agentic sub-system that analyzes data to identify missing dimensionless 
    physical numbers (Reynolds, Peclet, Nusselt) not explicitly in features.
    """
    def __init__(self, benchmark: str):
        self.benchmark = benchmark
        # Known constants for benchmarks
        self.known_constants = {
            "burgers_1d": ["viscosity"],
            "ns_2d": ["reynolds_number", "viscosity"],
            "darcy_2d": ["permeability"],
            "euler_1d": ["gamma", "mach_number"]
        }

    def identify_missing_constants(self, data_sample: Dict[str, Any]) -> List[str]:
        """Analyze sample data to guess which constants might be missing."""
        provided_keys = set(data_sample.keys())
        expected = self.known_constants.get(self.benchmark, [])
        missing = [c for c in expected if c not in provided_keys]
        return missing

    def estimate_dimensionless_numbers(self, u: np.ndarray, dx: float, nu: float) -> Dict[str, float]:
        """Calculate physics-based dimensionless numbers from field data."""
        results = {}
        if "ns" in self.benchmark or "burgers" in self.benchmark:
            # Re = U * L / nu
            u_max = np.max(np.abs(u))
            L = u.shape[-1] * dx
            results["reynolds_number"] = float(u_max * L / (nu + 1e-8))
            
        return results

class PhysicalAdversary:
    """
    Generates 'Physical Adversaries'—extreme edge cases (e.g., shock waves at 
    Mach 10) to test if the model's 'Scientific Intuition' holds up.
    """
    @staticmethod
    def generate_shock_wave(n: int, mach: float = 10.0) -> np.ndarray:
        """Generate a sharp discontinuity (shock) at a random location."""
        x = np.linspace(0, 1, n)
        shock_loc = np.random.uniform(0.3, 0.7)
        u = np.where(x < shock_loc, mach, 1.0)
        # Add some Gibbs-like oscillations near shock to make it harder
        u += 0.1 * np.exp(-((x - shock_loc) / 0.05)**2) * np.sin(50 * x)
        return u

    @staticmethod
    def generate_high_frequency_forcing(n: int, frequency: float = 100.0) -> np.ndarray:
        """Generate a highly oscillatory field to test spectral bias."""
        x = np.linspace(0, 1, n)
        return np.sin(frequency * np.pi * x)

    def stress_test_model(self, model_fn, n: int = 128) -> Dict[str, float]:
        """Run stress tests and return failure metrics."""
        shock = self.generate_shock_wave(n)
        osc = self.generate_high_frequency_forcing(n)
        
        # Placeholder for actual model inference and error calculation
        return {
            "shock_stability": 0.0,
            "spectral_resolution": 0.0
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
