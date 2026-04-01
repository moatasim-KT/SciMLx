"""
SciML experiment script — edit this file freely.

Baseline: Fourier Neural Operator (FNO) for multiple SciML benchmarks.
Goal: minimise val_l2_rel (lower is better) within the 5-minute budget.
"""

import gc
import math
import time
import argparse

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from prepare import GRID_SIZE, TIME_BUDGET, evaluate_l2_rel, make_dataloader
from models import FNO1d, FNO2d, DeepONet

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Benchmark Selection
BENCHMARK = "darcy_2d"  # Choices: "burgers_1d", "darcy_2d", "navier_stokes_2d"

# Architecture
MODEL_TYPE = "FNO"       # Choices: "FNO", "DeepONet"
N_MODES    = 16          # Fourier modes to keep (≤ GRID_SIZE // 2)
HIDDEN_DIM = 64          # channel width
N_LAYERS   = 4           # number of blocks

# Optimiser
BATCH_SIZE     = 32
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
ADAM_BETAS     = (0.9, 0.999)
WARMUP_RATIO   = 0.05
WARMDOWN_RATIO = 0.4
FINAL_LR_FRAC  = 0.01


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


# ── Training loop ─────────────────────────────────────────────────────────────

t_start = time.time()

train_loader   = make_dataloader(BENCHMARK, "train", BATCH_SIZE)
x_init, y_init = next(train_loader)
t_data         = time.time()
print(f"Data ready in {t_data - t_start:.1f}s")

# Initialize Model based on benchmark and type
if MODEL_TYPE == "FNO":
    if BENCHMARK.endswith("_1d"):
        model = FNO1d(n_modes=N_MODES, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    else:
        model = FNO2d(n_modes1=N_MODES, n_modes2=N_MODES, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
elif MODEL_TYPE == "DeepONet":
    # Simple coordinate grid for Trunk Net
    if BENCHMARK.endswith("_1d"):
        model = DeepONet(branch_dim=GRID_SIZE, trunk_dim=1, hidden_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM)
    else:
        model = DeepONet(branch_dim=GRID_SIZE*GRID_SIZE, trunk_dim=2, hidden_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM)
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

mx.eval(model.parameters())
n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
print(f"Benchmark: {BENCHMARK}")
print(f"Model    : {MODEL_TYPE}  layers={N_LAYERS}  hidden={HIDDEN_DIM}")
print(f"Params   : {n_params / 1e6:.3f}M")
print(f"Budget   : {TIME_BUDGET}s | batch={BATCH_SIZE}")

optimizer = AdamW(lr=LR, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)


def loss_fn(model, x, y):
    if MODEL_TYPE == "DeepONet":
        # DeepONet requires coordinates
        B = x.shape[0]
        if BENCHMARK.endswith("_1d"):
            coords = mx.linspace(0, 1, GRID_SIZE).reshape(1, GRID_SIZE, 1)
            coords = mx.broadcast_to(coords, (B, GRID_SIZE, 1))
            u_in = x
        else:
            grid1 = mx.linspace(0, 1, GRID_SIZE).reshape(1, GRID_SIZE, 1)
            grid2 = mx.linspace(0, 1, GRID_SIZE).reshape(1, 1, GRID_SIZE)
            coords = mx.stack([mx.broadcast_to(grid1, (1, GRID_SIZE, GRID_SIZE)), 
                             mx.broadcast_to(grid2, (1, GRID_SIZE, GRID_SIZE))], axis=-1)
            coords = mx.broadcast_to(coords, (B, GRID_SIZE, GRID_SIZE, 2)).reshape(B, -1, 2)
            u_in = x.reshape(B, -1)
        pred = model(u_in, coords)
        if not BENCHMARK.endswith("_1d"):
            pred = pred.reshape(B, GRID_SIZE, GRID_SIZE)
    else:
        pred = model(x)
        
    diff = pred - y
    axes = tuple(range(1, y.ndim))
    # Per-sample relative L2, averaged over batch
    return mx.mean(
        mx.sqrt(mx.mean(diff ** 2, axis=axes))
        / (mx.sqrt(mx.mean(y ** 2, axis=axes)) + 1e-8)
    )


loss_grad_fn = nn.value_and_grad(model, loss_fn)

x, y              = x_init, y_init
step              = 0
total_train_time  = 0.0
smooth_loss       = 0.0
t_compiled        = None

while True:
    t0 = time.time()

    loss, grads = loss_grad_fn(model, x, y)
    mx.eval(loss, grads)

    if t_compiled is None:
        t_compiled = time.time()
        print(f"Compiled in {t_compiled - t_data:.1f}s")

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
# Wrapper for evaluation if DeepONet is used
def eval_model(x):
    if MODEL_TYPE == "DeepONet":
        B = x.shape[0]
        if BENCHMARK.endswith("_1d"):
            coords = mx.linspace(0, 1, GRID_SIZE).reshape(1, GRID_SIZE, 1)
            coords = mx.broadcast_to(coords, (B, GRID_SIZE, 1))
            u_in = x
        else:
            grid1 = mx.linspace(0, 1, GRID_SIZE).reshape(1, GRID_SIZE, 1)
            grid2 = mx.linspace(0, 1, GRID_SIZE).reshape(1, 1, GRID_SIZE)
            coords = mx.stack([mx.broadcast_to(grid1, (1, GRID_SIZE, GRID_SIZE)), 
                             mx.broadcast_to(grid2, (1, GRID_SIZE, GRID_SIZE))], axis=-1)
            coords = mx.broadcast_to(coords, (B, GRID_SIZE, GRID_SIZE, 2)).reshape(B, -1, 2)
            u_in = x.reshape(B, -1)
        pred = model(u_in, coords)
        if not BENCHMARK.endswith("_1d"):
            pred = pred.reshape(B, GRID_SIZE, GRID_SIZE)
        return pred
    return model(x)

val_l2_rel = evaluate_l2_rel(BENCHMARK, eval_model)
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
