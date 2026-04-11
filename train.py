"""
SciML experiment script — edit this file freely.

Baseline: Fourier Neural Operator (FNO) for multiple SciML benchmarks.
Goal: minimise val_l2_rel (lower is better) within the 5-minute budget.

Available MODEL_TYPE values:
  "FNO", "RFNO", "AFNO", "FFNO", "UNO", "WNO", "DeepONet", "PODDeepONet", "S4NO", "GNOT"
"""

import gc
import math
import time
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent

import mlx.core as mx
import numpy as np
import mlx.nn as nn
from mlx.utils import tree_flatten

from prepare import GRID_SIZE, TIME_BUDGET, evaluate_l2_rel, make_dataloader
from benchmarks_ext import EXT_BENCHMARKS, make_ext_dataloader, evaluate_l2_rel_ext
from simulations import SIM_BENCHMARKS, SIM_IS_MC, SIM_N_CHANNELS
from losses import get_loss_fn
from research_plugins import MODEL_REGISTRY, BENCHMARK_REGISTRY
from trainer import Trainer, get_lr_schedule
from mlx.optimizers import AdamW

# ── Hyperparameters (module-level defaults) ───────────────────────────────────
BENCHMARK    = "burgers_1d"
MODEL_TYPE   = "FNO"
LOSS_TYPE    = "l2_rel"
H1_ALPHA     = 0.1
N_MODES      = 16
N_LEVELS     = 3
N_HEAD       = 4
SLICE_NUM    = 32
HIDDEN_DIM   = 64
N_LAYERS     = 4
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
PINO_LAMBDA  = 0.0
SPARSITY     = 0.01
AUGMENT      = False
CURRICULUM   = False
SAVE_CKPT    = False

# Scheduler
ADAM_BETAS     = (0.9, 0.999)
WARMUP_RATIO   = 0.05
WARMDOWN_RATIO = 0.2
FINAL_LR_FRAC  = 0.01

def _parse_args():
    p = argparse.ArgumentParser(description="SciML Training Script")
    p.add_argument("--benchmark",   default=BENCHMARK)
    p.add_argument("--model",       default=MODEL_TYPE)
    p.add_argument("--loss",        default=LOSS_TYPE,
                   choices=["l2_rel", "h1", "h1_strong", "spectral", "l1_rel", "mse"])
    p.add_argument("--h1_alpha",    type=float, default=H1_ALPHA)
    p.add_argument("--modes",       type=int,   default=N_MODES)
    p.add_argument("--levels",      type=int,   default=N_LEVELS)
    p.add_argument("--n_head",      type=int,   default=N_HEAD)
    p.add_argument("--slice_num",   type=int,   default=SLICE_NUM)
    p.add_argument("--hidden",      type=int,   default=HIDDEN_DIM)
    p.add_argument("--layers",      type=int,   default=N_LAYERS)
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--grad_clip",   type=float, default=GRAD_CLIP)
    p.add_argument("--pino_lambda", type=float, default=PINO_LAMBDA)
    p.add_argument("--sparsity",    type=float, default=SPARSITY)
    p.add_argument("--budget",      type=int,   default=TIME_BUDGET)
    p.add_argument("--name",        default="",
                   help="Experiment name written to telemetry file for dashboard tracking.")
    p.add_argument("--augment",     action="store_true", default=AUGMENT)
    p.add_argument("--curriculum",  action="store_true", default=CURRICULUM)
    p.add_argument("--save_ckpt",   action="store_true", default=SAVE_CKPT)
    p.add_argument("--resume",      action="store_true", help="Resume from best checkpoint if exists")
    p.add_argument("--resume_from", default="", help="Resume from specific checkpoint name/path")
    p.add_argument("--max_vram_gb", type=float, default=5.0,
                   help="Abort training if peak VRAM exceeds this (GB). 0=disabled.")
    return p.parse_args()

args = _parse_args()
BENCHMARK   = args.benchmark
MODEL_TYPE  = args.model
LOSS_TYPE   = args.loss
H1_ALPHA    = args.h1_alpha
N_MODES     = args.modes
N_LEVELS    = args.levels
N_HEAD      = args.n_head
SLICE_NUM   = args.slice_num
HIDDEN_DIM  = args.hidden
N_LAYERS    = args.layers
BATCH_SIZE  = args.batch_size
LR          = args.lr
GRAD_CLIP   = args.grad_clip
PINO_LAMBDA = args.pino_lambda
SPARSITY    = args.sparsity
TIME_BUDGET  = args.budget
AUGMENT      = args.augment
CURRICULUM   = args.curriculum
SAVE_CKPT    = args.save_ckpt
MAX_VRAM_GB  = args.max_vram_gb
EXP_NAME     = args.name or f"{MODEL_TYPE}_{BENCHMARK}"

# Apply hard memory limit at startup (Metal will raise OOM before swapping)
if MAX_VRAM_GB > 0:
    try:
        _limit_bytes = int(MAX_VRAM_GB * 1024 ** 3)
        mx.set_memory_limit(_limit_bytes)
        print(f"VRAM limit set to {MAX_VRAM_GB:.1f} GB")
    except Exception:
        pass

# ── Physics residuals (for PINO) ──────────────────────────────────────────────

def burgers_residual(u_pred: mx.array, nu: float = 0.01 / math.pi) -> mx.array:
    _, N   = u_pred.shape
    k      = mx.arange(N // 2 + 1, dtype=mx.float32)
    u_ft   = mx.fft.rfft(u_pred, axis=1)
    ux_ft_r = -u_ft.imag * k[None, :]
    ux_ft_i =  u_ft.real * k[None, :]
    ux      = mx.fft.irfft(ux_ft_r + 1j * ux_ft_i, n=N, axis=1)
    uxx_ft_r = -(k ** 2)[None, :] * u_ft.real
    uxx_ft_i = -(k ** 2)[None, :] * u_ft.imag
    uxx      = mx.fft.irfft(uxx_ft_r + 1j * uxx_ft_i, n=N, axis=1)
    return u_pred * ux - nu * uxx

# ── Model factory (registry-driven) ──────────────────────────────────────────

t_start = time.time()
train_loader = BENCHMARK_REGISTRY.make_loader(BENCHMARK, "train", BATCH_SIZE)
_eval_fn     = lambda model_fn: BENCHMARK_REGISTRY.evaluate(BENCHMARK, model_fn)

x_init, y_init = next(train_loader)
t_data         = time.time()
print(f"Data ready in {t_data - t_start:.1f}s")

is_1d  = BENCHMARK.endswith("_1d")
is_mc  = SIM_IS_MC.get(BENCHMARK, False)   # multi-channel benchmark?
n_ch   = SIM_N_CHANNELS.get(BENCHMARK, 1)  # number of physical channels

if MODEL_TYPE == "FNO" and is_mc:
    _model_key = "FNO_MC"
elif MODEL_TYPE == "FNO" and not is_1d:
    _model_key = "FNO2D"
else:
    _model_key = MODEL_TYPE

model = MODEL_REGISTRY.build(
    _model_key,
    n_modes=N_MODES, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, n_levels=N_LEVELS,
    n_head=N_HEAD, slice_num=SLICE_NUM,
    sparsity=SPARSITY,
    in_channels=n_ch, out_channels=n_ch,   # absorbed by **kw for non-MC models
)

# Resumption logic: load best weights if available
if (args.resume or args.resume_from) and EXP_NAME:
    # Use explicit resume_from if provided, otherwise fallback to current EXP_NAME
    source_name = args.resume_from if args.resume_from else EXP_NAME
    # If source_name doesn't end in .npz, assume it's an experiment name and append _best.npz
    if not source_name.endswith(".npz"):
        ckpt_path = REPO_ROOT / "checkpoints" / f"{source_name}_best.npz"
    else:
        ckpt_path = Path(source_name)
        if not ckpt_path.is_absolute():
            ckpt_path = REPO_ROOT / "checkpoints" / ckpt_path

    if ckpt_path.exists():
        print(f"Resuming from checkpoint: {ckpt_path.name}")
        model.load_weights(str(ckpt_path))
        mx.eval(model.parameters())
    else:
        if args.resume_from:
            print(f"Warning: Checkpoint {ckpt_path} not found. Starting from scratch.")

mx.eval(model.parameters())
n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
print(f"Benchmark: {BENCHMARK}")
print(f"Model    : {MODEL_TYPE}  layers={N_LAYERS}  hidden={HIDDEN_DIM}")
print(f"Params   : {n_params / 1e6:.3f}M")

optimizer = AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY, betas=list(ADAM_BETAS))

# ── Forward & Loss ─────────────────────────────────────────────────────────────

def _get_coords(B: int) -> mx.array:
    if is_1d:
        coords = mx.linspace(0, 1, GRID_SIZE).reshape(1, GRID_SIZE, 1)
        return mx.broadcast_to(coords, (B, GRID_SIZE, 1))
    g1 = mx.broadcast_to(mx.linspace(0, 1, GRID_SIZE).reshape(1, GRID_SIZE, 1), (1, GRID_SIZE, GRID_SIZE))
    g2 = mx.broadcast_to(mx.linspace(0, 1, GRID_SIZE).reshape(1, 1, GRID_SIZE), (1, GRID_SIZE, GRID_SIZE))
    coords = mx.stack([g1, g2], axis=-1)
    return mx.broadcast_to(coords, (B, GRID_SIZE, GRID_SIZE, 2)).reshape(B, -1, 2)

def _forward(model, x: mx.array) -> mx.array:
    if MODEL_TYPE == "DeepONet":
        B = x.shape[0]
        u_in = x if is_1d else x.reshape(B, -1)
        pred = model(u_in, _get_coords(B))
        return pred if is_1d else pred.reshape(B, GRID_SIZE, GRID_SIZE)
    if MODEL_TYPE == "PODDeepONet":
        B = x.shape[0]
        u_in = x if is_1d else x.reshape(B, -1)
        return model(u_in)
    return model(x)

_loss_kwargs = {"alpha": H1_ALPHA} if LOSS_TYPE.startswith("h1") else {}
_core_loss   = get_loss_fn(LOSS_TYPE, **_loss_kwargs)

def loss_fn(model, x, y):
    pred = _forward(model, x)
    data_loss = _core_loss(pred, y)
    if PINO_LAMBDA > 0 and BENCHMARK == "burgers_1d":
        res = burgers_residual(pred)
        phys_loss = mx.mean(mx.sum(res**2, axis=1) / (mx.sum(y**2, axis=1) + 1e-6))
        return data_loss + PINO_LAMBDA * phys_loss
    return data_loss

# ── Training ─────────────────────────────────────────────────────────────────

lr_sch = get_lr_schedule(WARMUP_RATIO, WARMDOWN_RATIO, FINAL_LR_FRAC)
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    forward_fn=_forward,
    eval_fn=_eval_fn,
    grad_clip=GRAD_CLIP,
    time_budget=TIME_BUDGET,
    lr_base=LR,
    lr_schedule_fn=lr_sch,
    max_vram_gb=MAX_VRAM_GB,
    curriculum=CURRICULUM,
    exp_name=EXP_NAME,
)

print(f"Starting training (budget {TIME_BUDGET}s)...")
steps, max_grad_norm, total_train_time = trainer.train(train_loader, t_data)

print("Evaluating...")
val_l2_rel = trainer.evaluate()
t_eval     = time.time()

from diagnostics import calculate_spectral_bias, generate_experiment_comparison

peak_vram_mb = mx.get_peak_memory() / 1024 / 1024
print("---")
print(f"val_l2_rel:       {val_l2_rel:.6f}")
print(f"training_seconds: {total_train_time:.1f}")
print(f"total_seconds:    {t_eval - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {steps}")
print(f"num_params_M:     {n_params / 1e6:.3f}")
print(f"architecture:     {MODEL_TYPE}-{BENCHMARK}")

# Generate Diagnostics
try:
    val_loader = BENCHMARK_REGISTRY.make_loader(BENCHMARK, "val", 8)
    x_val, y_val = next(val_loader)
    y_pred = _forward(model, x_val)
    mx.eval(y_pred)
    
    spec_bias = calculate_spectral_bias(np.array(y_pred), np.array(y_val))
    print(f"diag_low_freq_error: {spec_bias['low_freq_error']:.6f}")
    print(f"diag_high_freq_error: {spec_bias['high_freq_error']:.6f}")
    
    # Generate Inspector PNG
    exp_id = f"{MODEL_TYPE}_{BENCHMARK}_{int(time.time())}"
    generate_experiment_comparison(exp_id, np.array(x_val), np.array(y_val), np.array(y_pred), BENCHMARK)
    print(f"inspect_id: {exp_id}")
except Exception as e:
    print(f"diag_error: {e}")

if SAVE_CKPT:
    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{MODEL_TYPE}_{BENCHMARK}_val{val_l2_rel:.4f}.npz"
    mx.savez(str(ckpt_path), **dict(tree_flatten(model.parameters())))
    print(f"checkpoint_path:  {ckpt_path}")
