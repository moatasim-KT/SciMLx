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

from core.loader import EXPERIMENTS
from core.utils import REPO_ROOT

import torch
import torch.nn as nn
import numpy as np

from data.prepare import GRID_SIZE, TIME_BUDGET, evaluate_l2_rel, make_dataloader
from data.benchmarks_ext import EXT_BENCHMARKS, EXT_N_CHANNELS, make_ext_dataloader, evaluate_l2_rel_ext
from data.simulations import SIM_BENCHMARKS, SIM_IS_MC, SIM_N_CHANNELS
from core.losses import get_loss_fn, spectral_grad_1d, spectral_grad2_1d, relative_l2
from core.research_plugins import MODEL_REGISTRY, BENCHMARK_REGISTRY
from core.trainer import Trainer, get_lr_schedule
from torch.optim import AdamW

# ── Device management ────────────────────────────────────────────────────────
# Auto-detect GPU; fallback to CPU
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

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
NUM_EPOCHS   = 100

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
                   choices=["l2_rel", "h1", "h1_strong", "h1_adaptive", "spectral", "l1_rel", "mse"])
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
    p.add_argument("--patience",    type=int,   default=5,
                   help="Early stopping: halt if val does not improve for this many consecutive "
                        "10%%-budget evaluations. 0 disables early stopping (default: 5).")
    p.add_argument("--name",        default="",
                   help="Experiment name written to telemetry file for dashboard tracking.")
    p.add_argument("--augment",     action="store_true", default=AUGMENT)
    p.add_argument("--curriculum",  action="store_true", default=CURRICULUM)
    p.add_argument("--curriculum_epochs", type=int, default=0,
                   help="Number of epochs to ramp up spectral modes.")
    p.add_argument("--save_ckpt",   action="store_true", default=SAVE_CKPT)
    p.add_argument("--resume",      action="store_true", help="Resume from best checkpoint if exists")
    p.add_argument("--resume_from", default="", help="Resume from specific checkpoint name/path")
    p.add_argument("--max_vram_gb", type=float, default=5.0,
                   help="Abort training if peak VRAM exceeds this (GB). 0=disabled.")
    p.add_argument("--refine_grid", action="store_true",
                   help="Phase 11: Enable Adaptive Grid Extension (doubling G at 30%% and 60%% budget)")
    p.add_argument("--degree",      type=int,   default=5,
                   help="Chebyshev polynomial degree for cPIKAN models.")
    p.add_argument("--snapshot_ensemble", type=int, default=1,
                   help="Number of snapshots to save for ensemble UQ (default: 1).")
    p.add_argument("--lr_schedule", default="warmup_cosine",
                   choices=["warmup_cosine", "cosine", "onecycle", "none"],
                   help="LR schedule: warmup_cosine (default), cosine, onecycle, none.")
    p.add_argument("--seed", type=int, default=42,
                   help="Global random seed for reproducibility (default: 42).")
    p.add_argument("--ema_decay", type=float, default=0.0,
                   help="EMA decay for model weights (0=disabled, 0.999 recommended).")
    p.add_argument("--probe", action="store_true",
                   help="Enable high-fidelity layer-wise telemetry (Scientific Debugging).")
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
CURRICULUM_EPOCHS = args.curriculum_epochs
SAVE_CKPT    = args.save_ckpt
MAX_VRAM_GB  = args.max_vram_gb
EXP_NAME     = args.name or f"{MODEL_TYPE}_{BENCHMARK}"
N_ENSEMBLE   = args.snapshot_ensemble
LR_SCHEDULE  = args.lr_schedule
SEED         = args.seed
EMA_DECAY    = args.ema_decay

# Seed global RNG for reproducibility
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# Apply hard memory limit at startup (CUDA will raise OOM before swapping)
if MAX_VRAM_GB > 0 and DEVICE == "cuda":
    try:
        _limit_bytes = int(MAX_VRAM_GB * 1024 ** 3)
        torch.cuda.set_per_process_memory_fraction(_limit_bytes / torch.cuda.get_device_properties(0).total_memory)
        print(f"VRAM limit set to {MAX_VRAM_GB:.1f} GB")
    except Exception:
        pass

# ── Physics residuals (for PINO) ──────────────────────────────────────────────

def burgers_residual(u_pred: torch.Tensor, u0: torch.Tensor, nu: float = 0.01 / math.pi) -> torch.Tensor:
    """Computes u_t + u*u_x - nu*u_xx = 0 using backward Euler at the endpoint.

    The 'Endpoint Problem' fix: instead of just testing u*u_x - nu*u_xx = 0 (steady state),
    we estimate u_t as (u_pred - u0) / T.
    """
    T_final = 1.0 # Standard for burgers_1d
    ut      = (u_pred - u0) / T_final
    ux      = spectral_grad_1d(u_pred)
    uxx     = spectral_grad2_1d(u_pred)
    return ut + u_pred * ux - nu * uxx

def darcy_residual(u_pred: torch.Tensor, a_in: torch.Tensor) -> torch.Tensor:
    """PCG-equivalent Darcy residual for PINO loss: -div(a grad u) - f = 0.

    Includes the fixed source term f from the Darcy benchmark setup.
    """
    B, N, _ = u_pred.shape if u_pred.ndim == 3 else (u_pred.shape[0], u_pred.shape[1], 1)

    # Physical wavenumbers on [0,1]^2: 2pi * k
    k = 2 * math.pi * torch.fft.fftfreq(N, device=u_pred.device)
    kx, ky = torch.meshgrid(k, k, indexing='ij')

    # Compute grad u: [B, N, N, 2]
    u_hat = torch.fft.fft2(u_pred, dim=(1, 2))
    ux = torch.fft.ifft2(1j * kx[None] * u_hat, dim=(1, 2)).real
    uy = torch.fft.ifft2(1j * ky[None] * u_hat, dim=(1, 2)).real

    # -div(a grad u)
    flux_x_hat = torch.fft.fft2(a_in * ux, dim=(1, 2))
    flux_y_hat = torch.fft.fft2(a_in * uy, dim=(1, 2))

    div_a_grad_u = torch.fft.ifft2(
        1j * kx[None] * flux_x_hat + 1j * ky[None] * flux_y_hat,
        dim=(1, 2)
    ).real

    # Fixed source term f (must match data/benchmarks_ext.py _darcy_fix_ic)
    from data.prepare import _random_ic_2d
    f_rng = np.random.RandomState(12345)
    f_single = _random_ic_2d(1, N, f_rng, n_modes=5, scale=1.0, offset=0.0)
    f = torch.tensor(np.broadcast_to(f_single, (B, N, N)), dtype=u_pred.dtype, device=u_pred.device)

    return -div_a_grad_u - f

def apply_spectral_mask(x: torch.Tensor, k_max: int) -> torch.Tensor:
    """Filters x to only include frequencies up to k_max."""
    if x.ndim == 3: # 1D [B, N, C]
        _, N, _ = x.shape
        x_ft = torch.fft.rfft(x, dim=1)
        mask = torch.zeros_like(x_ft)
        mask[:, :k_max, :] = 1.0
        return torch.fft.irfft(x_ft * mask, n=N, dim=1)
    elif x.ndim == 4: # 2D [B, H, W, C]
        _, H, W, _ = x.shape
        x_ft = torch.fft.rfft2(x, dim=(1, 2))
        mask = torch.zeros_like(x_ft)
        # rfft2 last dim is W//2 + 1
        k_max_w = min(k_max, x_ft.shape[2])
        k_max_h = min(k_max, x_ft.shape[1])
        mask[:, :k_max_h, :k_max_w, :] = 1.0
        # Also need to handle the negative frequencies in the first axis if it was fft2,
        # but rfft2 only has one real axis. Wait, rfft2 axes (1,2) -> (1 is full, 2 is half).
        # So mask[:k_max] and mask[-k_max:] for the first axis.
        mask[:, -k_max_h:, :k_max_w, :] = 1.0
        return torch.fft.irfft2(x_ft * mask, s=(H, W), dim=(1, 2))
    return x

# ── Model factory (registry-driven) ──────────────────────────────────────────

t_start = time.time()
train_loader = BENCHMARK_REGISTRY.make_loader(BENCHMARK, "train", BATCH_SIZE)
_eval_fn     = lambda model_fn: BENCHMARK_REGISTRY.evaluate(BENCHMARK, model_fn)

x_init, y_init = next(train_loader)
# Convert numpy to torch tensors and move to device
if isinstance(x_init, np.ndarray):
    x_init = torch.tensor(x_init, dtype=torch.float32, device=DEVICE)
else:
    x_init = x_init.to(DEVICE)
if isinstance(y_init, np.ndarray):
    y_init = torch.tensor(y_init, dtype=torch.float32, device=DEVICE)
else:
    y_init = y_init.to(DEVICE)

t_data         = time.time()
print(f"Data ready in {t_data - t_start:.1f}s")

is_1d  = BENCHMARK.endswith("_1d")
is_mc  = SIM_IS_MC.get(BENCHMARK, False) or (BENCHMARK == "mhd_2d")
n_ch   = SIM_N_CHANNELS.get(BENCHMARK, EXT_N_CHANNELS.get(BENCHMARK, 1))

if MODEL_TYPE == "FNO" and is_mc:
    _model_key = "FNO_MC"
elif MODEL_TYPE == "FNO" and not is_1d:
    _model_key = "FNO2D"
elif MODEL_TYPE == "RFNO" and not is_1d:
    _model_key = "RFNO2D"
else:
    _model_key = MODEL_TYPE

model = MODEL_REGISTRY.build(
    _model_key,
    n_modes=N_MODES, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, n_levels=N_LEVELS,
    n_head=N_HEAD, slice_num=SLICE_NUM,
    sparsity=SPARSITY, n_sensors=GRID_SIZE,
    in_channels=n_ch, out_channels=n_ch,   # absorbed by **kw for non-MC models
    degree=args.degree,                    # For Chebyshev models
)

# Move model to device
model = model.to(DEVICE)

# Resumption logic: load best weights if available
if (args.resume or args.resume_from) and EXP_NAME:
    source_name = args.resume_from if args.resume_from else EXP_NAME

    # champion:<benchmark> — resolve via model registry
    if source_name.startswith("champion:"):
        _bm = source_name[len("champion:"):]
        try:
            from core.model_versioning import get_champion_path as _gcp
            _champ = _gcp(_bm, MODEL_TYPE)
            if _champ is None:
                _champ = _gcp(_bm)   # any model family
            ckpt_path = _champ if _champ else Path("__not_found__")
            if _champ:
                print(f"Champion for {_bm}: {ckpt_path.name}")
        except Exception as _e:
            print(f"[ModelRegistry] champion lookup failed: {_e}")
            ckpt_path = Path("__not_found__")
    elif not source_name.endswith(".pt"):
        ckpt_path = REPO_ROOT / "checkpoints" / f"{source_name}_best.pt"
    else:
        ckpt_path = Path(source_name)
        if not ckpt_path.is_absolute():
            ckpt_path = REPO_ROOT / "checkpoints" / ckpt_path

    if ckpt_path.exists():
        print(f"Resuming from checkpoint: {ckpt_path.name}")
        state_dict = torch.load(str(ckpt_path), map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        if args.resume_from:
            print(f"Warning: Checkpoint {ckpt_path} not found. Starting from scratch.")

model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Benchmark: {BENCHMARK}")
print(f"Model    : {MODEL_TYPE}  layers={N_LAYERS}  hidden={HIDDEN_DIM}")
print(f"Params   : {n_params / 1e6:.3f}M")

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=list(ADAM_BETAS))

# ── Forward & Loss ─────────────────────────────────────────────────────────────

def _get_coords(B: int) -> torch.Tensor:
    if is_1d:
        coords = torch.linspace(0, 1, GRID_SIZE, device=DEVICE).reshape(1, GRID_SIZE, 1)
        return coords.broadcast_to(B, GRID_SIZE, 1)
    g1 = torch.linspace(0, 1, GRID_SIZE, device=DEVICE).reshape(1, GRID_SIZE, 1).broadcast_to(1, GRID_SIZE, GRID_SIZE)
    g2 = torch.linspace(0, 1, GRID_SIZE, device=DEVICE).reshape(1, 1, GRID_SIZE).broadcast_to(1, GRID_SIZE, GRID_SIZE)
    coords = torch.stack([g1, g2], dim=-1)
    return coords.broadcast_to(B, GRID_SIZE, GRID_SIZE, 2).reshape(B, -1, 2)

def _forward(model, x: torch.Tensor) -> torch.Tensor:
    # Scalar-input benchmarks (e.g. poiseuille_flow_1d, couette_flow_1d):
    # inputs are (B, 1) — a single parameter per sample.
    # Broadcast to (B, GRID_SIZE) so spatial models (FNO, RFNO, …) work correctly.
    if is_1d and x.ndim == 2 and x.shape[1] == 1:
        x = x.broadcast_to(x.shape[0], GRID_SIZE)

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

CURRENT_K_MAX = 128 # Default to high

def loss_fn(model, x, y):
    pred = _forward(model, x)
    data_loss = _core_loss(pred, y)

    if PINO_LAMBDA > 0:
        if BENCHMARK == "burgers_1d":
            res = burgers_residual(pred, x)
            # Use relative L2 norm for physics loss stability
            phys_loss = torch.mean(
                torch.sqrt(torch.mean(res**2, dim=1)) / (torch.sqrt(torch.mean(y**2, dim=1)) + 1e-6)
            )
            return data_loss + PINO_LAMBDA * phys_loss
        elif BENCHMARK == "darcy_2d":
            if x.ndim == 4:
                a_in = x[..., 0]
            else:
                a_in = x
            u_pr = pred[..., 0] if pred.ndim == 4 else pred
            res = darcy_residual(u_pr, a_in)
            y_2d = y[..., 0] if y.ndim == 4 else y
            phys_loss = torch.mean(
                torch.sqrt(torch.mean(res**2, dim=(1, 2))) / (torch.sqrt(torch.mean(y_2d**2, dim=(1, 2))) + 1e-6)
            )
            return data_loss + PINO_LAMBDA * phys_loss
    return data_loss

# ── Phase 11: Adaptive Grid Hook ───────────────────────────────────────────

def refine_grid_callback(step: int, progress: float):
    # Only refine if flag is set and we're at key milestones
    if not args.refine_grid:
        return

    # Doubling milestones: 30% and 60%
    milestones = [0.3, 0.6]
    for m in milestones:
        # Check if we just crossed the milestone
        # Note: Progress is elapsed/budget
        if progress >= m and progress < m + 0.01:
            # Check if we already did this one (persistent state via model? No, just check current grid)
            for _, mod in model.named_modules():
                if hasattr(mod, "update_grid"):
                    current_g = getattr(mod, "grid_size")
                    # Crude check to avoid re-running same milestone:
                    # if we haven't doubled yet for this milestone
                    target_g = 5 * (2 if m == 0.3 else 4)
                    if current_g < target_g:
                        mod.update_grid(target_g)

# ── Training Loop ────────────────────────────────────────────────────────────
train_losses = []
val_losses   = []
min_val_loss = float("inf")

# ── Training ─────────────────────────────────────────────────────────────────

lr_sch = get_lr_schedule(
    warmup_ratio=WARMUP_RATIO,
    wardown_ratio=WARMDOWN_RATIO,
    final_lr_frac=FINAL_LR_FRAC,
    cyclical=(N_ENSEMBLE > 1),
    n_cycles=N_ENSEMBLE,
    schedule_type=LR_SCHEDULE,
)
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
    step_callback=refine_grid_callback,
    n_ensemble=N_ENSEMBLE,
    ema_decay=EMA_DECAY,
    probe_mode=args.probe,
    device=DEVICE,
)

print(f"Starting training (budget {TIME_BUDGET}s)...")
steps, max_grad_norm, total_train_time = trainer.train(train_loader, t_data)

print("Evaluating...")
val_l2_rel = trainer.evaluate()
t_eval     = time.time()

from core.diagnostics import calculate_spectral_bias, generate_experiment_comparison

# Get peak memory usage (PyTorch-compatible)
peak_vram_mb = 0.0
if DEVICE == "cuda":
    try:
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

print("---")
print(f"val_l2_rel:       {val_l2_rel:.6f}")
print(f"score:            {val_l2_rel:.6f}") # Standardized score output
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
    # Convert to tensors and move to device
    if isinstance(x_val, np.ndarray):
        x_val = torch.tensor(x_val, dtype=torch.float32, device=DEVICE)
    else:
        x_val = x_val.to(DEVICE)
    if isinstance(y_val, np.ndarray):
        y_val = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)
    else:
        y_val = y_val.to(DEVICE)

    y_pred = _forward(model, x_val)

    spec_bias = calculate_spectral_bias(y_pred.detach().cpu().numpy(), y_val.detach().cpu().numpy())
    print(f"diag_low_freq_error: {spec_bias['low_freq_error']:.6f}")
    print(f"diag_high_freq_error: {spec_bias['high_freq_error']:.6f}")

    # Generate Inspector PNG
    exp_id = f"{MODEL_TYPE}_{BENCHMARK}_{int(time.time())}"
    generate_experiment_comparison(exp_id, x_val.detach().cpu().numpy(), y_val.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), BENCHMARK)
    print(f"inspect_id: {exp_id}")
except Exception as e:
    print(f"diag_error: {e}")

if SAVE_CKPT:
    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{MODEL_TYPE}_{BENCHMARK}_val{val_l2_rel:.4f}.pt"
    torch.save(model.state_dict(), str(ckpt_path))
    print(f"checkpoint_path:  {ckpt_path}")

# ── MLflow Logging ─────────────────────────────────────────────────────────────
try:
    from core.mlflow_integration import log_run

    _mlflow_params = {
        "benchmark":    BENCHMARK,
        "model":        MODEL_TYPE,
        "loss":         LOSS_TYPE,
        "n_modes":      N_MODES,
        "hidden_dim":   HIDDEN_DIM,
        "n_layers":     N_LAYERS,
        "lr":           LR,
        "batch_size":   BATCH_SIZE,
        "grad_clip":    GRAD_CLIP,
        "budget_s":     TIME_BUDGET,
        "h1_alpha":     H1_ALPHA,
        "weight_decay": WEIGHT_DECAY,
        "augment":      AUGMENT,
        "curriculum":   CURRICULUM,
    }

    _mlflow_metrics: dict = {
        "val_l2_rel":       float(val_l2_rel),
        "training_seconds": float(total_train_time),
        "peak_vram_mb":     float(peak_vram_mb),
        "num_steps":        float(steps),
        "num_params_M":     float(n_params / 1e6),
    }

    # Fold in spectral diagnostics if they were computed above
    try:
        _mlflow_metrics["diag_low_freq_error"]  = float(spec_bias["low_freq_error"])
        _mlflow_metrics["diag_high_freq_error"] = float(spec_bias["high_freq_error"])
    except Exception:
        pass

    # Collect artifact paths: mid-run best checkpoint + final SAVE_CKPT checkpoint
    _artifacts: list[str] = []
    _mid_run_ckpt = REPO_ROOT / "checkpoints" / f"{EXP_NAME}_best.pt"
    if _mid_run_ckpt.exists():
        _artifacts.append(str(_mid_run_ckpt))
    if SAVE_CKPT and "ckpt_path" in dir():
        _artifacts.append(str(ckpt_path))
    # Include run log if it exists
    _log_file = REPO_ROOT / "logs" / f"{EXP_NAME}.log"
    if _log_file.exists():
        _artifacts.append(str(_log_file))

    _run_id = log_run(
        benchmark=BENCHMARK,
        model=MODEL_TYPE,
        exp_name=EXP_NAME,
        params=_mlflow_params,
        metrics=_mlflow_metrics,
        artifact_paths=_artifacts,
    )
    if _run_id:
        print(f"mlflow_run_id: {_run_id}")
except Exception as _mlflow_err:
    print(f"[MLflow] logging skipped: {_mlflow_err}")
else:
    _run_id = None  # ensure _run_id is always defined

# ── Model Registry ──────────────────────────────────────────────────────────
# Register the best checkpoint so `--resume_from champion:<benchmark>` works
# and the MLflow Model Registry shows the current champion per benchmark.
try:
    from core.model_versioning import register as _register_model

    # Use the mid-run best checkpoint (always saved); fall back to SAVE_CKPT path
    _reg_ckpt = REPO_ROOT / "checkpoints" / f"{EXP_NAME}_best.pt"
    if not _reg_ckpt.exists() and SAVE_CKPT and "ckpt_path" in dir():
        _reg_ckpt = ckpt_path

    if _reg_ckpt.exists():
        _version_id = _register_model(
            ckpt_path=_reg_ckpt,
            benchmark=BENCHMARK,
            model=MODEL_TYPE,
            exp_name=EXP_NAME,
            val_l2_rel=float(val_l2_rel),
            config=_mlflow_params if "_mlflow_params" in dir() else {},
            mlflow_run_id=_run_id,
        )
        print(f"model_version_id: {_version_id}")
except Exception as _reg_err:
    print(f"[ModelRegistry] registration skipped: {_reg_err}")
