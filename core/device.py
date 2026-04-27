import torch
import numpy as np
import platform

try:
    import mlx.core as mx
    # Verify it can run
    _ = mx.array([1.0])
    _HAS_MLX = True
except (ImportError, RuntimeError, AttributeError):
    _HAS_MLX = False

def get_framework():
    """Return the active framework: 'mlx' or 'torch'."""
    # MLX is preferred on Apple Silicon if available
    if _HAS_MLX and platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "torch"

FRAMEWORK = get_framework()

def get_device():
    """Return the best available device string or torch.device."""
    if FRAMEWORK == "mlx":
        return "mlx"
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

def to_framework_device(data):
    """Move tensors or models to the active device for Torch, or pass-through for MLX."""
    if FRAMEWORK == "mlx":
        return data
    
    if isinstance(data, (torch.Tensor, torch.nn.Module)):
        return data.to(DEVICE)
    if isinstance(data, dict):
        return {k: to_framework_device(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_framework_device(v) for v in data)
    return data

def to_device(data):
    """Alias for to_framework_device for backward compatibility."""
    return to_framework_device(data)

def to_array(data):
    """Convert data to framework-native array/tensor."""
    if FRAMEWORK == "mlx":
        if isinstance(data, mx.array):
            return data
        return mx.array(data)
    else:
        if isinstance(data, torch.Tensor):
            return data.to(DEVICE)
        return torch.as_tensor(data, device=DEVICE)
