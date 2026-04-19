# PyTorch Refactoring Plan — autoresearch-mlx to PyTorch/Linux

**Goal:** Convert MLX-based codebase to PyTorch to enable execution on Linux/GPU without MLX native binary dependency.

**Scope:** 46 files with MLX imports across core training, diagnostics, and 40+ model implementations.

---

## Phase 1: Core Training Infrastructure (Enables 80% of functionality)

### Priority 1.1: Replace MLX tensors with PyTorch tensors

**Files to refactor:**
1. `train.py` (PRIMARY ENTRY POINT — largest payoff)
2. `core/trainer.py`
3. `core/losses.py`
4. `core/scaffold.py`

**Key conversions:**

```python
# MLX → PyTorch
import mlx.core as mx           → import torch
import mlx.nn as nn             → import torch.nn as nn
from mlx.optimizers import AdamW → from torch.optim import AdamW

# Tensor operations
mx.array(...)                   → torch.tensor(...)
mx.zeros(...), mx.ones(...)     → torch.zeros(...), torch.ones(...)
mx.linspace, mx.arange, etc.    → torch.linspace, torch.arange, etc.
mx.fft.rfft, mx.fft.fft2, etc.  → torch.fft.rfft, torch.fft.fft2, etc.

# Device management
mx.eval(model.parameters())     → model.to('cuda') or model.to('cpu')
mx.random.seed(seed)            → torch.manual_seed(seed)
mx.set_memory_limit(bytes)      → torch.cuda.set_per_process_memory_fraction()

# Parameter handling
tree_flatten(model.parameters()) → model.parameters()
model.load_weights(path)         → model.load_state_dict(torch.load(path))
```

### Priority 1.2: Refactor forward pass and loss computation

**Changes needed in train.py:**
- Replace `_forward()` function to work with PyTorch models
- Update `loss_fn()` to use PyTorch tensor operations
- Handle gradient computation with PyTorch's autograd
- Replace MLX's evaluation mode with PyTorch's eval/train mode

### Priority 1.3: Update Trainer class

**core/trainer.py changes:**
- Replace MLX training loop with PyTorch training loop
- Swap `mx.value_and_grad()` with `torch.autograd.grad()`
- Use PyTorch's learning rate scheduler instead of MLX's
- Handle checkpointing with `torch.save()/torch.load()`

---

## Phase 2: Model Implementations (40+ files)

### Priority 2.1: Core neural operator models (High-impact, used in experiments)

These models should be refactored first as they're in the experiment queue:

**FNO family:** (3 models)
- `models/fno.py` → `FNO`, `FNO2D`, `FFNO`

**RFNO family:** (2 models)
- `models/rfno.py` → `RFNO`, `RFNO2D`

**Attention variants:** (2 models)
- `models/afno.py` → `AFNO`
- `models/attention_fno.py` → `AttentionEnhancedFNO2D`

**Operator variants:**
- `models/wno.py` → `WNO`
- `models/deeponet.py` → `DeepONet`
- `models/pod_deeponet.py` → `PODDeepONet`

### Priority 2.2: Mid-priority models (Used in some experiments)

- `models/transolver.py`, `models/fedonet.py`, `models/snot.py` 
- `models/gnot.py`, `models/pinn.py`
- MambaNO, MemNO (newer architectures)

### Priority 2.3: Exploratory/Low-priority models

- `models/kan.py`, `models/chebyshev_kan.py` (KAN variants)
- `models/s4d.py` (Structured state-space)
- `models/neural_ode.py`, `models/hnn.py` (Physics-informed)
- etc.

---

## Phase 3: Utilities & Diagnostics

### Priority 3.1: Core utilities

- `core/utils.py` — mostly import-agnostic, minimal changes
- `core/research_plugins.py` — MODEL_REGISTRY refactoring

### Priority 3.2: Diagnostics and telemetry

- `core/diagnostics.py` — gradient norm extraction, spectral analysis
- `core/losses.py` — already partially done in Phase 1.1
- `data/prepare.py` — mostly numpy, minimal MLX usage

### Priority 3.3: Visualization and dashboard

- `core/viz.py` — can keep if tests pass, else simplify

---

## Key Conversion Patterns

### Pattern 1: Tensor creation and manipulation

```python
# MLX
x = mx.array(np.random.randn(10, 20))
y = mx.zeros((5, 5))
z = x + y

# PyTorch
x = torch.tensor(np.random.randn(10, 20), dtype=torch.float32, device=device)
y = torch.zeros((5, 5), device=device)
z = x + y
```

### Pattern 2: FFT operations

```python
# MLX
x_ft = mx.fft.rfft(x, axis=1)
x_spatial = mx.fft.irfft(x_ft, n=N, axis=1)

# PyTorch
x_ft = torch.fft.rfft(x, dim=1)
x_spatial = torch.fft.irfft(x_ft, n=N, dim=1)
```

### Pattern 3: Model layers

```python
# MLX
class FNOBlock(mx.nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.linear = mx.nn.Linear(in_dim, out_dim)
    
    def __call__(self, x):
        return self.linear(x)

# PyTorch
class FNOBlock(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.linear(x)
```

### Pattern 4: Training loop

```python
# MLX
def loss_and_grad(model, x, y):
    loss, grads = mx.value_and_grad(loss_fn)(model, x, y)
    return loss, grads

optimizer.update(model.parameters(), grads)

# PyTorch
optimizer.zero_grad()
loss = loss_fn(model, x, y)
loss.backward()
optimizer.step()
```

### Pattern 5: Checkpointing

```python
# MLX
mx.savez(path, **{k: v for k, v in model.items()})
model.load_weights(path)

# PyTorch
torch.save(model.state_dict(), path)
model.load_state_dict(torch.load(path))
```

---

## Execution Plan

### Phase 1: Core (Days 1-2 if done sequentially, hours if done in parallel)
1. ✅ Update pyproject.toml (DONE)
2. Refactor `train.py` → `train_pytorch.py` (or in-place)
3. Refactor `core/trainer.py`
4. Refactor `core/losses.py`
5. Refactor `core/scaffold.py`
6. Test with `python3 train.py --benchmark burgers_1d --model FNO --budget 60 --epochs 5`

### Phase 2: Models (Parallel-friendly)
1. Refactor FNO family (3 files)
2. Refactor RFNO family (2 files)
3. Refactor other high-priority models (10-15 files)
4. Test model loading with `MODEL_REGISTRY.build(...)`

### Phase 3: Utilities (2-3 hours)
1. Update diagnostics
2. Update model registry
3. Test end-to-end training

### Phase 4: Validation
1. Run 3-5 experiments from queue
2. Verify results.json compatibility
3. Test result consolidation with Mac M1 version

---

## Device Management Strategy

**MLX:** Metal GPU acceleration on Apple Silicon
**PyTorch:** 
- CUDA for NVIDIA GPUs
- CPU fallback (slower but works everywhere)
- MPS for Apple Silicon (if available in PyTorch)

**Implementation:**
```python
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
```

---

## Testing Strategy

1. **Unit tests:** Each refactored module tested in isolation
2. **Integration tests:** train.py executes 1 epoch on toy data
3. **Experiment tests:** Run 3-5 actual experiments from queue
4. **Regression tests:** Compare metrics to MLX version

---

## Risk Mitigation

**Risk:** Numerical differences between MLX and PyTorch
**Mitigation:** Cross-validate on same data, allow ±5% tolerance initially

**Risk:** Performance differences
**Mitigation:** Benchmark wall-clock time, adjust batch size if needed

**Risk:** Memory model differences
**Mitigation:** Test on varied GPU memory (8GB, 16GB, 24GB)

---

## Success Criteria

✅ All MLX imports removed  
✅ Can run: `python3 train.py --benchmark burgers_1d --model FNO --budget 300`  
✅ Results saved to results.json  
✅ Gradient diagnostics working  
✅ At least 5 experiments complete successfully  
✅ Results compatible with Mac M1 version for consolidation  

---

## Files Modified

**Modified:**
- `pyproject.toml`
- `train.py`
- `core/trainer.py`
- `core/losses.py`
- `core/scaffold.py`
- `core/research_plugins.py`
- All `models/*.py` (40+ files)

**Added:**
- `PYTORCH_REFACTORING_PLAN.md` (this file)
- `pytorch_compat.py` (optional compatibility layer)

**Unchanged:**
- `experiments.yaml` (format compatible)
- `results.json` (format compatible)
- Data loading and evaluation functions (mostly numpy)

---

**Estimated effort:** 40-60 hours if done sequentially, 10-15 hours with parallel work
**Expected completion:** Within 2-3 days with focused effort
