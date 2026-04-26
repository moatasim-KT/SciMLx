"""
SciML experiment script (PyTorch/CUDA) — NVIDIA GPU optimized.
"""

import gc
import math
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from core.device import DEVICE, to_device
from core.research_plugins import MODEL_REGISTRY, BENCHMARK_REGISTRY
from core.trainer import Trainer, get_lr_schedule
from core.losses import get_loss_fn
from core.utils import REPO_ROOT

# ── Hyperparameters (module-level defaults) ───────────────────────────────────
BENCHMARK    = "burgers_1d"
MODEL_TYPE   = "FNO"
LOSS_TYPE    = "l2_rel"
H1_ALPHA     = 0.1
N_MODES      = 16
HIDDEN_DIM   = 64
N_LAYERS     = 4
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
TIME_BUDGET  = 300
EMA_DECAY    = 0.999
USE_AMP      = True
COMPILE      = True

def _parse_args():
    p = argparse.ArgumentParser(description="SciML Training Script (CUDA)")
    p.add_argument("--benchmark",   default=BENCHMARK)
    p.add_argument("--model",       default=MODEL_TYPE)
    p.add_argument("--loss",        default=LOSS_TYPE)
    p.add_argument("--h1_alpha",    type=float, default=H1_ALPHA)
    p.add_argument("--modes",       type=int,   default=N_MODES)
    p.add_argument("--hidden",      type=int,   default=HIDDEN_DIM)
    p.add_argument("--layers",      type=int,   default=N_LAYERS)
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--grad_clip",   type=float, default=GRAD_CLIP)
    p.add_argument("--budget",      type=int,   default=TIME_BUDGET)
    p.add_argument("--name",        default="")
    p.add_argument("--ema_decay",   type=float, default=EMA_DECAY)
    p.add_argument("--no_amp",      action="store_false", dest="use_amp")
    p.add_argument("--no_compile",  action="store_false", dest="compile")
    return p.parse_args()

def main():
    args = _parse_args()
    
    # 1. Environment Setup
    torch.set_float32_matmul_precision('high')
    t_start = time.time()
    print(f"Device: {DEVICE}")

    # 2. Data
    print(f"Loading {args.benchmark} data...")
    train_loader = BENCHMARK_REGISTRY.make_loader(args.benchmark, "train", args.batch_size)
    
    # 3. Model
    model = MODEL_REGISTRY.build(
        args.model, 
        benchmark=args.benchmark,
        n_modes=args.modes,
        hidden_dim=args.hidden,
        n_layers=args.layers
    )
    model = to_device(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params/1e6:.3f}M params)")

    # 4. Training Components
    loss_fn = get_loss_fn(args.loss, alpha=args.h1_alpha if "h1" in args.loss else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_sch = get_lr_schedule(schedule_type="warmup_cosine")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        forward_fn=lambda m, x: m(x),
        eval_fn=lambda fn: BENCHMARK_REGISTRY.evaluate(args.benchmark, fn),
        grad_clip=args.grad_clip,
        time_budget=args.budget,
        lr_base=args.lr,
        lr_schedule_fn=lr_sch,
        exp_name=args.name or f"{args.model}_{args.benchmark}",
        ema_decay=args.ema_decay,
        use_amp=args.use_amp,
        compile=args.compile
    )

    # 5. Loop
    print(f"Starting training (budget {args.budget}s)...")
    steps, max_grad_norm, total_train_time = trainer.train(train_loader, t_start)

    # 6. Evaluation
    print("Evaluating...")
    val_l2_rel = trainer.evaluate()
    t_end = time.time()

    print("---")
    print(f"val_l2_rel:       {val_l2_rel:.6f}")
    print(f"training_seconds: {total_train_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"num_steps:        {steps}")
    print(f"peak_vram_mb:     {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")

if __name__ == "__main__":
    main()
