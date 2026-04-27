import os
import platform
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import mlx.core as mx
    # Verify it can run
    _ = mx.array([1.0])
    _HAS_MLX = True
except (ImportError, RuntimeError, AttributeError):
    _HAS_MLX = False

def get_framework():
    """
    Detect the compute framework.
    Priority:
    1. SCIMLX_BACKEND environment variable ('torch' or 'mlx').
    2. 'mlx' if on Apple Silicon and mlx is installed.
    3. 'torch' as default.
    """
    env_backend = os.environ.get("SCIMLX_BACKEND", "").lower()
    if env_backend == "mlx":
        if _HAS_MLX:
            return "mlx"
        # If MLX requested but not available, we don't return "mlx" 
        # because it would break later calls.
        return "torch"
    if env_backend == "torch":
        return "torch"

    # MLX is preferred on Apple Silicon if available
    if _HAS_MLX and platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "torch"

FRAMEWORK = get_framework()

def get_device():
    """Return the best available device string or torch.device."""
    if FRAMEWORK == "mlx":
        return "mlx"
    
    if _HAS_TORCH:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    return "cpu"

DEVICE = get_device()

def to_array(data, dtype=None):
    """
    Convert data to framework-native array/tensor.
    Ensures data is on the correct device.
    """
    if FRAMEWORK == "mlx" and _HAS_MLX:
        if isinstance(data, mx.array):
            return data if dtype is None else data.astype(dtype)
        # Handle torch tensors if passed to mlx to_array
        if _HAS_TORCH and torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        return mx.array(data, dtype=dtype)
    
    if _HAS_TORCH:
        if torch.is_tensor(data):
            res = data.to(DEVICE)
        else:
            # Handle mlx arrays if passed to torch to_array
            if _HAS_MLX and isinstance(data, mx.array):
                data = np.array(data)
            res = torch.as_tensor(data, device=DEVICE)
        
        if dtype is not None:
            # Handle common dtype conversions
            if isinstance(dtype, str):
                if dtype == "float32": dtype = torch.float32
                elif dtype == "float64": dtype = torch.float64
            res = res.to(dtype)
        return res
    
    # Fallback to numpy if no backend framework is available
    res = np.array(data)
    if dtype is not None:
        res = res.astype(dtype)
    return res

def to_device(data):
    """
    Move tensors or models to the active device.
    For MLX, this is mostly a no-op as it manages its own memory.
    """
    if FRAMEWORK == "mlx":
        return data
    
    if _HAS_TORCH:
        if isinstance(data, (torch.Tensor, torch.nn.Module)):
            return data.to(DEVICE)
        if isinstance(data, dict):
            return {k: to_device(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(to_device(v) for v in data)
    
    return data

def to_framework_device(data):
    """Alias for to_device for backward compatibility."""
    return to_device(data)
