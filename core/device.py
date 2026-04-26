import torch

def get_device():
    """Return the best available device (CUDA, then CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

def to_device(data):
    """Move tensors or models to the active device."""
    if isinstance(data, (torch.Tensor, torch.nn.Module)):
        return data.to(DEVICE)
    if isinstance(data, dict):
        return {k: to_device(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(v) for v in data)
    return data
