# Contributing: Adding New Models (CUDA Optimized)

This guide covers the development setup and the workflow for adding new PyTorch-based neural operator architectures.

---

## Development Setup

```bash
# Requires NVIDIA GPU + CUDA 12.1+ + uv
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo-url> scimlx
cd scimlx
uv sync
```

Verify your environment:

```bash
uv run python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0)})')"
# Expected: CUDA Available: True (...)
```

---

## Model Interface Contract

Every model must inherit from `torch.nn.Module` and implement the `forward` method.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyOperator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        n_modes: int,
        **kwargs,   # Mandatory to absorb extra configuration args
    ):
        super().__init__()
        self.lift = nn.Linear(1, hidden_dim)
        # Build layers here...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1D benchmarks: x is (B, N) or (B, N, C)
        # 2D benchmarks: x is (B, H, W) or (B, H, W, C)
        # Return tensor of the same spatial shape
        ...
```

### Shape Conventions

| Benchmark type | Input shape | Output shape |
|---|---|---|
| 1D (burgers, kdv, wave, …) | `(B, N)` | `(B, N)` |
| 2D (darcy, ns, …) | `(B, H, W)` | `(B, H, W)` |

### Critical Rules

- **Device Agnostic**: Never hardcode `.cuda()` or `.to('cuda')`. Models are moved to the correct device automatically by the harness via `core/device.py`.
- **Complex Weights**: For spectral operators, use `torch.complex64` for parameters.
- **`**kwargs`**: Always include `**kwargs` in your `__init__` method.

---

## The Scaffold Pipeline

Use the scaffold to ensure your model is correctly integrated into the registry.

### 1. Generate Stub

```bash
uv run -m core.scaffold --stub MyOperator --base FNO
```

### 2. Validate

```bash
uv run -m core.scaffold --validate MyOperator models/my_operator.py
```

This runs a smoke test with random tensors on your GPU to verify output shapes and import safety.

### 3. Register

```bash
uv run -m core.scaffold --register MyOperator models/my_operator.py
```

Automatically updates `models/__init__.py`, `core/research_plugins.py`, and appends a baseline experiment to `experiments.yaml`.

---

## PR Checklist

- [ ] `uv run -m core.scaffold --validate MyOperator models/my_operator.py` passes.
- [ ] Model uses `torch.compile` safely (no dynamic python control flow in forward).
- [ ] Memory usage is efficient (tested on 2D benchmarks if applicable).
- [ ] A short training run completes:
  ```bash
  uv run train.py --benchmark burgers_1d --model MyOperator --budget 60
  ```
