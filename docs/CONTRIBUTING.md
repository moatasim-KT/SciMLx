# Contributing: Adding New Models

This guide covers the development setup and the end-to-end workflow for adding a new
neural operator architecture to the research loop.

---

## Development Setup

```bash
# Requires Apple Silicon + Python 3.10–3.13 + uv
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo-url> autoresearch-mlx
cd autoresearch-mlx
uv sync
```

Verify your environment:

```bash
uv run python -c "import mlx.core as mx; print(mx.default_device())"
# Expected: Device(gpu, 0)
```

---

## Model Interface Contract

Every model must satisfy this exact interface. `train.py` dispatches through
`ModelRegistry.build()` and calls only `__init__` and `__call__`.

```python
import mlx.core as mx
import mlx.nn as nn

class MyOperator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        n_modes: int,
        **kwargs,   # absorb extra args (n_levels, n_head, etc.) without error
    ):
        super().__init__()
        # Build your layers here

    def __call__(self, x: mx.array) -> mx.array:
        # 1D benchmarks: x is (batch, grid, channels)
        # 2D benchmarks: x is (batch, h, w, channels)
        # Return tensor of the same spatial shape as x
        ...
```

### Shape Conventions

| Benchmark type | Input shape | Output shape |
|---|---|---|
| 1D (burgers, kdv, wave, …) | `(B, N, C)` | `(B, N, C)` |
| 2D (darcy, ns, …) | `(B, H, W, C)` | `(B, H, W, C)` |

`N`, `H`, `W` are always `GRID_SIZE = 64`. `C` is 1 for most benchmarks; multi-channel
benchmarks (`euler_1d`, `mhd_2d`) pass `C > 1` via `kwargs`.

### Critical Rules

- Never import `torch` or `numpy` inside model forward passes — only `mlx.core`.
- Do not call `mx.eval()` inside `__call__`. The trainer handles lazy evaluation.
- `**kwargs` in `__init__` is mandatory — the harness always passes extra keyword
  arguments that your model may not need.

---

## The Three-Gate Scaffold Pipeline

All new models must pass three validation gates before entering the experiment queue.
This prevents broken imports or shape errors from corrupting overnight runs.

### Gate 1: Generate Starter File

```bash
uv run -m core.scaffold --stub MyOperator --base FNO
# Creates models/my_operator.py
```

The generated stub contains:
- A `lift` layer: `nn.Linear(1, hidden_dim)`
- A list of `SpectralConv1d` operator blocks
- A `proj` head: `nn.Sequential(Linear, GELU, Linear)`
- Forward pass with residual connections

Modify the stub to replace the placeholder blocks with your actual architecture.
The `TODO: replace with your custom operator blocks` comment marks where to work.

### Gate 2: Validate

```bash
uv run -m core.scaffold --validate MyOperator models/my_operator.py
```

The validator runs three checks in order:

| Check | Tool | Pass Condition |
|---|---|---|
| **Syntax** | `ast.parse()` | No `SyntaxError` |
| **Import** | `importlib.util.spec_from_file_location` | Module loads, class name found |
| **Smoke test** | Forward pass | Output shape matches input shape |

The smoke test instantiates with `(n_modes=8, hidden_dim=16, n_layers=2)` and runs
a forward pass with both shapes:
- 1D: `mx.zeros((2, 64, 1))` → output must be `(2, 64, 1)`
- 2D: `mx.zeros((2, 16, 16, 1))` → output must be `(2, 16, 16, 1)`

All three gates must pass. Fix any failures before proceeding.

### Gate 3: Register

```bash
uv run -m core.scaffold --register MyOperator
```

`--register` re-runs all three gates, then modifies three files:

1. **`models/__init__.py`** — adds `from .my_operator import MyOperator` and
   `"MyOperator"` to `__all__`
2. **`core/research_plugins.py`** — adds
   `MODEL_REGISTRY.register_class("MyOperator", MyOperator)` before the
   `BENCHMARK_REGISTRY` line
3. **`experiments.yaml`** — appends a starter `ExperimentConfig` entry with
   `budget_s: 480` (2D) or `300` (1D) and `priority: 3`

---

## Registry Mechanics

`core/research_plugins.py` maintains two registries:

```python
MODEL_REGISTRY = ModelRegistry()
BENCHMARK_REGISTRY = BenchmarkRegistry()
```

**`ModelRegistry.register_lazy(key, module_name, cls_name)`** — deferred import (class loaded
on first `build()` call). This is the standard pattern used throughout `research_plugins.py`.  
**`ModelRegistry.register_class(key, cls)`** — eager registration (class imported at
module load time). Used internally by `register_lazy` after first resolution.

The scaffold's `--register` step adds a `register_lazy` call to `research_plugins.py`.

When `train.py` calls `MODEL_REGISTRY.build("MyOperator", benchmark="burgers_1d", hidden_dim=64, ...)`:
1. The registry resolves the lazy import if needed
2. For all 2D benchmarks: enforces `hidden_dim < 64`, `n_layers < 8` (raises `ValueError`)
3. Auto-maps single `n_modes` → `(n_modes1, n_modes2)` for 2D models that need two counts
4. Calls `cls(hidden_dim=..., n_layers=..., n_modes=..., **extra_kwargs)`

---

## Code Style

- No comments unless the **why** is non-obvious (a hidden constraint, a workaround for
  a specific MLX bug, a subtle mathematical invariant)
- No docstrings — well-named parameters communicate enough
- Do not add error handling for scenarios that cannot happen in the harness
- Prefer editing existing files over creating new ones

---

## PR Checklist

Before submitting a pull request adding a new model:

- [ ] `uv run -m core.scaffold --validate MyOperator models/my_operator.py` → all 3 gates pass
- [ ] `uv run -m core.scaffold --register MyOperator` completes without error
- [ ] Model appears in `experiments.yaml` with a sensible `rationale`
- [ ] `from models import MyOperator` succeeds in a fresh Python session
- [ ] A short training run completes without crash:
  ```bash
  uv run train.py --benchmark burgers_1d --model_type MyOperator \
    --hidden_dim 16 --n_layers 2 --n_modes 8 --budget 60
  ```
- [ ] For 2D models: same smoke test on `darcy_2d` with `hidden_dim 32`
