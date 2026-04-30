"""Aesthetic Interpretability: Sonification of Residuals for SciMLx."""

import numpy as np
import wavio
from pathlib import Path

def sonify_residual(residual: np.ndarray, output_path: str, duration: float = 2.0, sample_rate: int = 44100):
    """
    Convert PDE residual norm into a multi-sensory diagnostic audio file.
    - Low residual: Pure sine wave (hum).
    - High residual: White noise (static).
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Calculate global residual norm as a proxy for "failure"
    res_norm = np.linalg.norm(residual)
    # Map residual to noise level [0, 1]
    # Assuming residual norm > 1.0 is "failing"
    noise_level = np.clip(res_norm, 0.0, 1.0)
    sine_level = 1.0 - noise_level
    
    # Generate components
    sine_wave = np.sin(2 * np.pi * 440 * t) # A4 note
    white_noise = np.random.uniform(-1, 1, len(t))
    
    # Mix
    audio = (sine_level * sine_wave) + (noise_level * white_noise)
    
    # Normalize to 16-bit PCM range
    audio = audio / np.max(np.abs(audio))
    
    print(f"[Audio] Sonifying residual (norm={res_norm:.4f}) to {output_path}...")
    wavio.write(output_path, audio, sample_rate, sampwidth=2)
    return output_path

class ResidualSonifier:
    """Helper class to track and sonify residuals during training."""
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def sonify_step(self, step: int, residual: np.ndarray):
        path = self.log_dir / f"residual_step_{step}.wav"
        return sonify_residual(residual, str(path))
