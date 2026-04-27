"""Unified Trainer for SciML experiments on NVIDIA GPUs (PyTorch/CUDA)."""

import time
import math
import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Callable, Any, Dict, Optional, Tuple, List
from core.utils import TELEMETRY_DIR, REPO_ROOT
from core.device import DEVICE, to_device

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model: nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self):
        if self.decay <= 0:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    new_average = (1.0 - self.decay) * param.data + self.shadow[name] * self.decay
                    self.shadow[name].copy_(new_average)

    def apply_shadow(self):
        self.backup = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

class SpectralBiasGovernor:
    """
    Monitors the Fourier spectrum of residuals and suggests loss weight adjustments.
    Prevents the 'Spectral Bias' where models fail to learn high-frequency details.
    """
    def __init__(self, n_modes: int, update_interval: int = 50):
        self.n_modes = n_modes
        self.update_interval = update_interval
        self.current_weights = None
        self._step_count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self._step_count += 1
        if self._step_count % self.update_interval != 0:
            return self.current_weights
            
        with torch.no_grad():
            residual = pred - target
            if residual.ndim == 2:
                res_ft = torch.fft.rfft(residual, dim=1).abs().mean(dim=0)
                # Identify where residual is high relative to its own mean
                # to emphasize those frequencies
                norm_res = res_ft / (res_ft.mean() + 1e-8)
                self.current_weights = 1.0 + torch.clamp(norm_res - 1.0, min=0.0)
            elif residual.ndim == 3:
                res_ft = torch.fft.rfft2(residual, dim=(1, 2)).abs().mean(dim=0)
                norm_res = res_ft / (res_ft.mean() + 1e-8)
                self.current_weights = 1.0 + torch.clamp(norm_res - 1.0, min=0.0)
                
        return self.current_weights

class Trainer:
    """Encapsulates training loop, evaluation, and metrics for PyTorch/CUDA."""
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        forward_fn: Callable,
        eval_fn: Callable,
        grad_clip: float = 1.0,
        time_budget: int = 300,
        lr_base: float = 1e-3,
        lr_schedule_fn: Optional[Callable[[float], float]] = None,
        max_vram_gb: float = 0.0,
        curriculum: bool = False,
        exp_name: str = "",
        step_callback: Optional[Callable[[int, float], None]] = None,
        patience: int = 5,
        n_ensemble: int = 1,
        ema_decay: float = 0.0,
        use_amp: bool = True,
        compile: bool = True,
    ):
        self.model = to_device(model)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.forward_fn = forward_fn
        self.eval_fn = eval_fn
        self.grad_clip = grad_clip
        self.time_budget = time_budget
        self.lr_base = lr_base
        self.lr_schedule_fn = lr_schedule_fn
        self.max_vram_gb = max_vram_gb
        self.curriculum = curriculum
        self.exp_name = exp_name
        self.step_callback = step_callback
        self.patience = patience
        self.n_ensemble = n_ensemble
        self.ema_decay = ema_decay
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Spectral Bias Governor (Phase 3)
        self.governor = SpectralBiasGovernor(n_modes=32) if "spectral" in str(loss_fn) else None

        # EMA initialization
        self.ema = EMA(self.model, ema_decay) if ema_decay > 0 else None
        
        # Mixed Precision Scaler
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Compiled model for speed
        if compile and hasattr(torch, 'compile'):
            try:
                print("[Trainer] Compiling model with torch.compile...", flush=True)
                self.compiled_model = torch.compile(self.model)
            except Exception as e:
                print(f"[Trainer] Compilation failed: {e}. Using raw model.", flush=True)
                self.compiled_model = self.model
        else:
            self.compiled_model = self.model

        self._loss_history: list = []
        self._snapshot_val_scores: list = []
        
        from pathlib import Path as _Path
        _slug = exp_name.replace("/", "_").replace(" ", "_") if exp_name else ""
        self._telemetry_path = TELEMETRY_DIR / (
            f".vram_telemetry_{_slug}" if _slug else ".vram_telemetry"
        )

    def train(self, train_loader, t_start: float):
        self.model.train()
        total_steps = 0
        max_grad_norm = 0.0
        current_gnorm = 0.0
        total_train_time = 0
        t_last_log = t_start

        best_val = float("inf")
        best_state = None
        next_eval_progress = 0.10
        EVAL_INTERVAL = 0.10
        no_improve_count = 0
        
        initial_loss = None
        loss_at_50 = None
        extensions_count = 0
        MAX_EXTENSIONS = 5
        
        next_snapshot_progress = 1.0 / self.n_ensemble if self.n_ensemble > 1 else 2.0
        snapshot_count = 0

        while True:
            t_now = time.time()
            elapsed = t_now - t_start
            if elapsed > self.time_budget:
                if extensions_count < MAX_EXTENSIONS and initial_loss and loss_at_50 and len(self._loss_history) > 0:
                    if self._loss_history[-1][1] < 0.8 * initial_loss and \
                       self._loss_history[-1][1] < 0.9 * loss_at_50:
                        extension = int(self.time_budget * 0.2)
                        self.time_budget += extension
                        extensions_count += 1
                        print(f"\n[Dynamic Budget] Loss is decreasing well ({self._loss_history[-1][1]:.6f}). "
                              f"Extending budget by {extension}s to {self.time_budget}s.", flush=True)
                    else:
                        break
                else:
                    break

            progress = elapsed / self.time_budget
            if self.lr_schedule_fn:
                new_lr = self.lr_base * self.lr_schedule_fn(progress)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

            try:
                x, y = next(train_loader)
                x, y = to_device(x), to_device(y)
            except StopIteration:
                break

            # Curriculum: spectral smoothing (omitted here for brevity, but should be ported)
            # ... (TBD: port curriculum spectral smoothing if critical)

            t_step_start = time.time()
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                pred = self.compiled_model(x)
                
                # Spectral Bias Governor Update
                loss_kwargs = {}
                if self.governor:
                    weights = self.governor.update(pred, y)
                    if weights is not None:
                        loss_kwargs["weights"] = weights

                # Physical Consistency Check (Phase 2)
                if hasattr(self.model, "input_units") and hasattr(self.model, "output_units"):
                    from core.units import SciMLTensor, check_consistency
                    # This is a placeholder for actual dimensional analysis of the forward pass
                    pass

                loss = self.loss_fn(pred, y, **loss_kwargs)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                else:
                    # Need to unscale for gnorm monitoring even if not clipping
                    self.scaler.unscale_(self.optimizer)
                    gnorm = torch.tensor(0.0) # To be calculated if needed
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                else:
                    gnorm = torch.tensor(0.0)
                self.optimizer.step()

            loss_val = loss.item()
            current_gnorm = float(gnorm)
            max_grad_norm = max(max_grad_norm, current_gnorm)

            if initial_loss is None: initial_loss = loss_val
            if progress >= 0.5 and loss_at_50 is None: loss_at_50 = loss_val

            if math.isnan(loss_val) or math.isinf(loss_val):
                raise RuntimeError(f"NaN/Inf loss at step {total_steps}")

            if self.ema:
                self.ema.update()

            if self.step_callback:
                self.step_callback(total_steps, progress)

            total_train_time += (time.time() - t_step_start)
            total_steps += 1

            if total_steps % 20 == 0:
                t_now2 = time.time()
                dt_ms = (t_now2 - t_last_log) / 20 * 1000
                remaining = max(0.0, self.time_budget - (t_now2 - t_start))
                
                print(f"step {total_steps:05d} ({progress*100:.1f}%) | loss: {loss_val:.6f} | "
                      f"lr: {self.optimizer.param_groups[0]['lr']:.2e} | gnorm: {current_gnorm:.3f} | "
                      f"dt: {dt_ms:.0f}ms | remaining: {remaining:.0f}s", flush=True)
                
                t_last_log = t_now2

                # VRAM guard + live telemetry
                try:
                    active_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    peak_mb   = torch.cuda.max_memory_allocated() / 1024 / 1024
                except Exception:
                    active_mb = peak_mb = 0.0
                
                self._loss_history.append([total_steps, loss_val])
                if len(self._loss_history) > 100:
                    self._loss_history = self._loss_history[-100:]
                
                if self.max_vram_gb > 0 and peak_mb / 1024 > self.max_vram_gb:
                    raise RuntimeError(f"VRAM limit exceeded: {peak_mb/1024:.2f} GB")

            # Mid-run validation & Snapshots (logic similar to MLX, using self.ema.apply_shadow())
            if progress >= next_eval_progress:
                self.model.eval()
                if self.ema:
                    self.ema.apply_shadow()
                
                with torch.no_grad():
                    val = self.eval_fn(lambda x: self.compiled_model(to_device(x)))
                
                if self.ema:
                    self.ema.restore()
                
                if val < best_val:
                    no_improve_count = 0
                    best_val = val
                    best_state = copy.deepcopy(self.model.state_dict())
                    if self.exp_name:
                        ckpt_path = REPO_ROOT / "checkpoints" / f"{self.exp_name}_best.pt"
                        ckpt_path.parent.mkdir(exist_ok=True)
                        torch.save(self.model.state_dict(), ckpt_path)
                else:
                    no_improve_count += 1

                print(f"val@{progress*100:.0f}%: {val:.6f}", flush=True)
                next_eval_progress += EVAL_INTERVAL
                self.model.train()

                if self.patience > 0 and no_improve_count >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return total_steps, max_grad_norm, total_train_time

    def evaluate(self):
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        
        with torch.no_grad():
            result = self.eval_fn(lambda x: self.compiled_model(to_device(x)))
        
        if self.ema:
            self.ema.restore()
        return result

def get_lr_schedule(
    warmup_ratio: float = 0.05,
    wardown_ratio: float = 0.2,
    final_lr_frac: float = 0.01,
    cyclical: bool = False,
    n_cycles: int = 1,
    schedule_type: str = "warmup_cosine",
) -> Callable[[float], float]:
    def lr_schedule(progress: float) -> float:
        if schedule_type == "none": return 1.0
        if schedule_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress)) * (1.0 - final_lr_frac) + final_lr_frac
        if schedule_type == "onecycle":
            peak_frac = 0.3
            if progress < peak_frac: return progress / peak_frac
            t = (progress - peak_frac) / (1.0 - peak_frac)
            return 0.5 * (1.0 + math.cos(math.pi * t)) * (1.0 - final_lr_frac) + final_lr_frac
        
        if progress < warmup_ratio: return progress / warmup_ratio if warmup_ratio > 0 else 1.0
        if progress < 1.0 - wardown_ratio: return 1.0
        t = (1.0 - (progress - (1.0 - wardown_ratio)) / wardown_ratio)
        return t + (1 - t) * final_lr_frac
    return lr_schedule
