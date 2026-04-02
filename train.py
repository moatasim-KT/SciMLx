"""
SciML experiment script — edit this file freely.

Baseline: Fourier Neural Operator (FNO) for multiple SciML benchmarks.
Goal: minimise val_l2_rel (lower is better) within the 5-minute budget.

Available MODEL_TYPE values:
  "FNO"        - Fourier Neural Operator                  (Li et al. 2020)
  "RFNO"       - Residual FNO with Pre-LN blocks          (unlocks l≥10)
  "AFNO"       - Adaptive FNO: block-diagonal MLP + softshrink (Guibas 2022)
  "FFNO"       - Factorized FNO: diagonal per-mode weights (Tran et al. 2023)
  "UNO"        - U-shaped Neural Operator                 (Rahman et al. 2022)
  "WNO"        - Wavelet Neural Operator                  (Tripura et al. 2022)
  "DeepONet"   - Deep Operator Network                    (Lu et al. 2019)
  "PODDeepONet"- POD-based DeepONet                       (Lu et al. 2022)

Available LOSS_TYPE values:
  "l2_rel"    - relative L2 (default)
  "h1"        - H1 Sobolev: L2 + a·L2(∂u/∂x)  → targets shock fronts
  "h1_strong" - H1 with a=1.0
  "spectral"  - frequency-weighted L2
  "l1_rel"    - relative L1

CLI flags (all optional; module-level constants below are the defaults):
  --benchmark  burgers_1d|darcy_2d|kdv_1d|wave_1d
  --model      FNO|RFNO|AFNO|FFNO|UNO|WNO|DeepONet|PODDeepONet
  --loss       l2_rel|h1|h1_strong|spectral|l1_rel
  --h1_alpha   weight for H1 derivative term (default 0.1)
  --modes      Fourier modes (FNO/UNO/RFNO/AFNO)
  --levels     Haar decomposition levels (WNO)
  --hidden     channel width
  --layers     depth
  --lr         learning rate
  --batch_size batch size
  --grad_clip  max gradient norm (0 = disabled)
  --pino_lambda weight for physics residual loss (0 = disabled)
"""

import gc
import math
import time
import argparse

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from prepare import GRID_SIZE, TIME_BUDGET, evaluate_l2_rel, make_dataloader
from benchmarks_ext import EXT_BENCHMARKS, make_ext_dataloader, evaluate_l2_rel_ext
from losses import get_loss_fn
from research_plugins import MODEL_REGISTRY, BENCHMARK_REGISTRY

# ── Hyperparameters (module-level defaults) ───────────────────────────────────
BENCHMARK    = "burgers_1d"   # "burgers_1d" | "darcy_2d" | "kdv_1d" | "wave_1d"
MODEL_TYPE   = "FNO"          # see module docstring
LOSS_TYPE    = "l2_rel"       # "l2_rel" | "h1" | "h1_strong" | "spectral" | "l1_rel"
H1_ALPHA     = 0.1            # weight for H1 derivative term
N_MODES      = 16             # Fourier modes to keep  (FNO, UNO; ≤ GRID_SIZE//2)
N_LEVELS     = 3              # Haar decomposition levels (WNO; 3 → N/8 approx)
HIDDEN_DIM   = 64             # channel width
N_LAYERS     = 4              # depth
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0            # max gradient L2 norm; 0 = disabled
PINO_LAMBDA  = 0.0            # physics residual loss weight (0 = data-only)

# Scheduler
ADAM_BETAS     = (0.9, 0.999)
WARMUP_RATIO   = 0.05
WARMDOWN_RATIO = 0.4
FINAL_LR_FRAC  = 0.01


# ── CLI override ──────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="SciML Training Script")
    p.add_argument("--benchmark",   default=BENCHMARK)
    p.add_argument("--model",       default=MODEL_TYPE)
    p.add_argument("--loss",        default=LOSS_TYPE,
                   choices=["l2_rel", "h1", "h1_strong", "spectral", "l1_rel", "mse"])
    p.add_argument("--h1_alpha",    type=float, default=H1_ALPHA)
    p.add_argument("--modes",       type=int,   default=N_MODES)
    p.add_argument("--levels",      type=int,   default=N_LEVELS)
    p.add_argument("--hidden",      type=int,   default=HIDDEN_DIM)
    p.add_argument("--layers",      type=int,   default=N_LAYERS)
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--grad_clip",   type=float, default=GRAD_CLIP)
    p.add_argument("--pino_lambda", type=float, default=PINO_LAMBDA)
    return p.parse_args()

args = _parse_args()
BENCHMARK   = args.benchmark
MODEL_TYPE  = args.model
LOSS_TYPE   = args.loss
H1_ALPHA    = args.h1_alpha
N_MODES     = args.modes
N_LEVELS    = args.levels
HIDDEN_DIM  = args.hidden
N_LAYERS    = args.layers
BATCH_SIZE  = args.batch_size
LR          = args.lr
GRAD_CLIP   = args.grad_clip
PINO_LAMBDA = args.pino_lambda


# ── Optimiser ─────────────────────────────────────────────────────────────────

class AdamW:
    """AdamW with runtime learning-rate control."""

    def __init__(self, lr: float, weight_decay: float,
                 betas=(0.9, 0.999), eps: float = 1e-8):
        self.lr   = lr
        self.wd   = weight_decay
        self.b1, self.b2 = betas
        self.eps  = eps
        self._s: dict = {}
        self._t   = 0

    def _set(self, model, path, val):
        parts = path.split(".")
        obj   = model
        for p in parts[:-1]:
            obj = obj[int(p)] if isinstance(obj, list) else (
                  obj[p]      if isinstance(obj, dict)  else getattr(obj, p))
        last = parts[-1]
        if   isinstance(obj, list): obj[int(last)] = val
        elif isinstance(obj, dict): obj[last]      = val
        else:                       setattr(obj, last, val)

    def update(self, model, grads):
        self._t += 1
        flat_g = dict(tree_flatten(grads))
        flat_p = dict(tree_flatten(model.parameters()))
        b1, b2 = self.b1, self.b2

        for path, g in flat_g.items():
            p   = flat_p[path].astype(mx.float32)
            g   = g.astype(mx.float32)
            if path not in self._s:
                self._s[path] = {"m": mx.zeros_like(g), "v": mx.zeros_like(g)}
            s      = self._s[path]
            s["m"] = b1 * s["m"] + (1 - b1) * g
            s["v"] = b2 * s["v"] + (1 - b2) * g * g
            mh     = s["m"] / (1 - b1 ** self._t)
            vh     = s["v"] / (1 - b2 ** self._t)
            p      = p * (1 - self.lr * self.wd) - self.lr * mh / (mx.sqrt(vh) + self.eps)
            self._set(model, path, p.astype(flat_p[path].dtype))

    @property
    def state_arrays(self):
        out = []
        for s in self._s.values():
            out += [s["m"], s["v"]]
        return out


def lr_schedule(progress: float) -> float:
    """Linear warmup → flat → cosine warmdown."""
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    t = (1.0 - progress) / WARMDOWN_RATIO
    return t + (1 - t) * FINAL_LR_FRAC


# ── Gradient utilities ────────────────────────────────────────────────────────

def _scale_tree(tree, scale: float):
    """Recursively multiply every array in a nested tree by a scalar."""
    if isinstance(tree, mx.array):
        return tree * scale
    if isinstance(tree, dict):
        return {k: _scale_tree(v, scale) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_scale_tree(v, scale) for v in tree]
    return tree


def clip_grad_norm(grads, max_norm: float):
    """Clip gradient tree by global L2 norm.  Returns (clipped_grads, norm)."""
    flat_g = dict(tree_flatten(grads))
    sq_sum = sum(float(mx.sum(g * g).item()) for g in flat_g.values())
    norm   = sq_sum ** 0.5
    if norm > max_norm:
        grads = _scale_tree(grads, max_norm / (norm + 1e-6))
    return grads, norm


# ── Physics residuals (for PINO) ──────────────────────────────────────────────

def burgers_residual(u_pred: mx.array, nu: float = 0.01 / math.pi) -> mx.array:
    """Spectral Burgers residual: u·∂u/∂x - v·∂²u/∂x² evaluated at u_pred.

    Works on a uniform periodic grid [0, 2π).
    Returns [B, N] residual field; minimise its L2 norm as physics loss.
    """
    _, N   = u_pred.shape
    k      = mx.arange(N // 2 + 1, dtype=mx.float32)   # wave numbers 0…N//2
    u_ft   = mx.fft.rfft(u_pred, axis=1)               # [B, N//2+1] complex

    # ∂u/∂x  ↔  multiply Fourier coeff by ik  →  real=-imag*k, imag=real*k
    ux_ft_r = -u_ft.imag * k[None, :]
    ux_ft_i =  u_ft.real * k[None, :]
    ux      = mx.fft.irfft(ux_ft_r + 1j * ux_ft_i, n=N, axis=1)

    # ∂²u/∂x²  ↔  multiply by -k²
    uxx_ft_r = -(k ** 2)[None, :] * u_ft.real
    uxx_ft_i = -(k ** 2)[None, :] * u_ft.imag
    uxx      = mx.fft.irfft(uxx_ft_r + 1j * uxx_ft_i, n=N, axis=1)

    return u_pred * ux - nu * uxx


# ── Model factory (registry-driven) ──────────────────────────────────────────

t_start = time.time()

# Route benchmark → dataloader and eval function via registry
train_loader = BENCHMARK_REGISTRY.make_loader(BENCHMARK, "train", BATCH_SIZE)
_eval_fn     = lambda model: BENCHMARK_REGISTRY.evaluate(BENCHMARK, model)

x_init, y_init = next(train_loader)
t_data         = time.time()
print(f"Data ready in {t_data - t_start:.1f}s")

is_1d = BENCHMARK.endswith("_1d")

# Route model type → model instance via registry
# 2D benchmarks use a FNO2D key; all others route directly
_model_key = ("FNO2D" if MODEL_TYPE == "FNO" and not is_1d else MODEL_TYPE)
model = MODEL_REGISTRY.build(
    _model_key,
    n_modes=N_MODES, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, n_levels=N_LEVELS,
)

mx.eval(model.parameters())
n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
print(f"Benchmark: {BENCHMARK}")
print(f"Model    : {MODEL_TYPE}  layers={N_LAYERS}  hidden={HIDDEN_DIM}")
print(f"Params   : {n_params / 1e6:.3f}M")
print(f"Budget   : {TIME_BUDGET}s | batch={BATCH_SIZE} | "
      f"grad_clip={GRAD_CLIP} | pino_λ={PINO_LAMBDA} | loss={LOSS_TYPE}")

optimizer = AdamW(lr=LR, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)


# ── Loss function ─────────────────────────────────────────────────────────────

def _get_coords(B: int) -> mx.array:
    """Evaluation coordinates for DeepONet-family models."""
    if is_1d:
        coords = mx.linspace(0, 1, GRID_SIZE).reshape(1, GRID_SIZE, 1)
        return mx.broadcast_to(coords, (B, GRID_SIZE, 1))
    g1 = mx.broadcast_to(mx.linspace(0, 1, GRID_SIZE).reshape(1, GRID_SIZE, 1),
                          (1, GRID_SIZE, GRID_SIZE))
    g2 = mx.broadcast_to(mx.linspace(0, 1, GRID_SIZE).reshape(1, 1, GRID_SIZE),
                          (1, GRID_SIZE, GRID_SIZE))
    coords = mx.stack([g1, g2], axis=-1)
    return mx.broadcast_to(coords, (B, GRID_SIZE, GRID_SIZE, 2)).reshape(B, -1, 2)


def _forward(model, x: mx.array) -> mx.array:
    """Unified forward pass for all model types."""
    if MODEL_TYPE == "DeepONet":
        B    = x.shape[0]
        u_in = x if is_1d else x.reshape(B, -1)
        pred = model(u_in, _get_coords(B))
        return pred if is_1d else pred.reshape(B, GRID_SIZE, GRID_SIZE)
    if MODEL_TYPE == "PODDeepONet":
        B    = x.shape[0]
        u_in = x if is_1d else x.reshape(B, -1)
        return model(u_in)
    return model(x)


# Build the core loss function once (supports l2_rel, h1, spectral, etc.)
_loss_kwargs = {"alpha": H1_ALPHA} if LOSS_TYPE.startswith("h1") else {}
_core_loss   = get_loss_fn(LOSS_TYPE, **_loss_kwargs)


def loss_fn(model, x, y):
    pred = _forward(model, x)
    data_loss = _core_loss(pred, y)

    if PINO_LAMBDA > 0 and BENCHMARK == "burgers_1d":
        axes = tuple(range(1, y.ndim))
        res  = burgers_residual(pred)
        # Normalise physics loss (relative L2) so PINO_LAMBDA is a true ratio.
        phys_loss = mx.mean(
            mx.sqrt(mx.mean(res ** 2, axis=axes))
            / (mx.sqrt(mx.mean(pred ** 2, axis=axes)) + 1e-8)
        )
        return data_loss + PINO_LAMBDA * phys_loss
    return data_loss


loss_grad_fn = nn.value_and_grad(model, loss_fn)


# ── Training loop ─────────────────────────────────────────────────────────────

x, y             = x_init, y_init
step             = 0
total_train_time = 0.0
smooth_loss      = 0.0
t_compiled       = None
max_grad_norm    = 0.0

while True:
    t0 = time.time()

    loss, grads = loss_grad_fn(model, x, y)
    mx.eval(loss, grads)

    if t_compiled is None:
        t_compiled = time.time()
        print(f"Compiled in {t_compiled - t_data:.1f}s")

    # Gradient clipping
    if GRAD_CLIP > 0:
        grads, gnorm = clip_grad_norm(grads, GRAD_CLIP)
        max_grad_norm = max(max_grad_norm, gnorm)

    progress      = min(total_train_time / TIME_BUDGET, 1.0)
    optimizer.lr  = LR * lr_schedule(progress)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), *optimizer.state_arrays)

    x, y = next(train_loader)

    loss_f = float(loss.item())
    if not math.isfinite(loss_f) or loss_f > 100.0:
        print("FAIL — loss diverged")
        raise SystemExit(1)

    dt = time.time() - t0
    if step > 0:
        total_train_time += dt

    ema         = 0.95
    smooth_loss = ema * smooth_loss + (1 - ema) * loss_f
    debiased    = smooth_loss / (1 - ema ** (step + 1))
    pct         = 100.0 * progress
    remaining   = max(0.0, TIME_BUDGET - total_train_time)

    print(
        f"\rstep {step:05d} ({pct:.1f}%) | "
        f"loss: {debiased:.6f} | "
        f"lr: {optimizer.lr:.2e} | "
        f"dt: {dt*1000:.0f}ms | "
        f"remaining: {remaining:.0f}s    ",
        end="", flush=True,
    )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

    step += 1
    if step > 0 and total_train_time >= TIME_BUDGET:
        break

print()
t_train = time.time()
print(f"Training done in {t_train - t_compiled:.1f}s")

print("Evaluating...")
val_l2_rel = _eval_fn(lambda x: _forward(model, x))
t_eval     = time.time()
print(f"Eval done in {t_eval - t_train:.1f}s")

peak_vram_mb = mx.get_peak_memory() / 1024 / 1024

print("---")
print(f"val_l2_rel:       {val_l2_rel:.6f}")
print(f"training_seconds: {total_train_time:.1f}")
print(f"total_seconds:    {t_eval - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {n_params / 1e6:.3f}")
print(f"architecture:     {MODEL_TYPE}-{BENCHMARK}")
print(f"batch_size:       {BATCH_SIZE}")
print(f"max_grad_norm:    {max_grad_norm:.4f}")
