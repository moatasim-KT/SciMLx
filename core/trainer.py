"""Unified Trainer for SciML experiments supporting PyTorch and MLX."""

import time
import math
import numpy as np
import copy
from typing import Callable, Any, Dict, Optional, Tuple, List, Union
from core.utils import TELEMETRY_DIR, REPO_ROOT
from core.device import DEVICE, FRAMEWORK, to_device
from core.spectral_governor import SpectralBiasGovernor

# Optional imports for backends
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import mlx.core as mx
    import mlx.nn as mx_nn
    import mlx.optimizers as mxo
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

class EMA:
    """Exponential Moving Average for model parameters (Torch only)."""
    def __init__(self, model: Any, decay: float):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self):
        if self.decay <= 0:
            return
        import torch
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    new_average = (1.0 - self.decay) * param.data + self.shadow[name] * self.decay
                    self.shadow[name].copy_(new_average)

    def apply_shadow(self):
        import torch
        self.backup = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

class BaseTrainer:
    """Abstract base class with shared logic (budget tracking, logging)."""
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_fn: Callable,
        forward_fn: Callable,
        eval_fn: Callable,
        grad_clip: float = 1.0,
        time_budget: int = 300,
        lr_base: float = 1e-3,
        lr_schedule_fn: Optional[Callable[[float], float]] = None,
        max_vram_gb: float = 0.0,
        curriculum: bool = False,
        curriculum_epochs: int = 0,
        exp_name: str = "",
        step_callback: Optional[Callable[[int, float], None]] = None,
        patience: int = 5,
        snapshot_ensemble: int = 0,
        ema_decay: float = 0.0,
        pino_lambda: float = 0.0,
        save_ckpt: bool = False,
        resume: bool = False,
        resume_from: str = "",
    ):
        self.model = model
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
        self.curriculum_epochs = curriculum_epochs
        self.exp_name = exp_name
        self.step_callback = step_callback
        self.patience = patience
        self.snapshot_ensemble = snapshot_ensemble
        self.ema_decay = ema_decay
        self.pino_lambda = pino_lambda
        self.save_ckpt = save_ckpt
        self.resume = resume
        self.resume_from = resume_from
        
        self._loss_history: list = []
        self._snapshot_val_scores: list = []
        self._extensions_count = 0
        
        _slug = exp_name.replace("/", "_").replace(" ", "_") if exp_name else ""
        self._telemetry_path = TELEMETRY_DIR / (
            f".vram_telemetry_{_slug}" if _slug else ".vram_telemetry"
        )

    def _is_budget_exceeded(self, t_start: float, initial_loss: Optional[float], loss_at_50: Optional[float]) -> bool:
        t_now = time.time()
        elapsed = t_now - t_start
        if elapsed > self.time_budget:
            MAX_EXTENSIONS = 5
            if self._extensions_count < MAX_EXTENSIONS and initial_loss and loss_at_50 and len(self._loss_history) > 0:
                last_loss = self._loss_history[-1][1]
                if last_loss < 0.8 * initial_loss and last_loss < 0.9 * loss_at_50:
                    extension = int(self.time_budget * 0.2)
                    self.time_budget += extension
                    self._extensions_count += 1
                    print(f"\n[Dynamic Budget] Loss is decreasing well ({last_loss:.6f}). "
                          f"Extending budget by {extension}s to {self.time_budget}s.", flush=True)
                    return False
            return True
        return False

    def train(self, train_loader, t_start: float):
        raise NotImplementedError

    def train_step(self, x, y):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

class TrainerTorch(BaseTrainer):
    """The existing PyTorch/CUDA implementation."""
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_fn: Callable,
        forward_fn: Callable,
        eval_fn: Callable,
        grad_clip: float = 1.0,
        time_budget: int = 300,
        lr_base: float = 1e-3,
        lr_schedule_fn: Optional[Callable[[float], float]] = None,
        max_vram_gb: float = 0.0,
        curriculum: bool = False,
        curriculum_epochs: int = 0,
        exp_name: str = "",
        step_callback: Optional[Callable[[int, float], None]] = None,
        patience: int = 5,
        snapshot_ensemble: int = 0,
        ema_decay: float = 0.0,
        pino_lambda: float = 0.0,
        save_ckpt: bool = False,
        resume: bool = False,
        resume_from: str = "",
        use_amp: bool = True,
        compile: bool = True,
    ):
        super().__init__(
            model=to_device(model),
            optimizer=optimizer,
            loss_fn=loss_fn,
            forward_fn=forward_fn,
            eval_fn=eval_fn,
            grad_clip=grad_clip,
            time_budget=time_budget,
            lr_base=lr_base,
            lr_schedule_fn=lr_schedule_fn,
            max_vram_gb=max_vram_gb,
            curriculum=curriculum,
            curriculum_epochs=curriculum_epochs,
            exp_name=exp_name,
            step_callback=step_callback,
            patience=patience,
            snapshot_ensemble=snapshot_ensemble,
            ema_decay=ema_decay,
            pino_lambda=pino_lambda,
            save_ckpt=save_ckpt,
            resume=resume,
            resume_from=resume_from,
        )
        import torch
        self.use_amp = use_amp and torch.cuda.is_available()
        self.governor = SpectralBiasGovernor(n_modes=32) if "spectral" in str(loss_fn) else None
        self.ema = EMA(self.model, ema_decay) if ema_decay > 0 else None
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        if compile and hasattr(torch, 'compile'):
            try:
                print("[TrainerTorch] Compiling model with torch.compile...", flush=True)
                self.compiled_model = torch.compile(self.model)
            except Exception as e:
                print(f"[TrainerTorch] Compilation failed: {e}. Using raw model.", flush=True)
                self.compiled_model = self.model
        else:
            self.compiled_model = self.model

    def train_step(self, x, y):
        import torch
        self.optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            pred = self.compiled_model(x)
            loss_kwargs = {}
            if self.governor:
                weights = self.governor.update(pred, y)
                if weights is not None:
                    loss_kwargs["weights"] = weights
            loss = self.loss_fn(pred, y, **loss_kwargs)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            else:
                self.scaler.unscale_(self.optimizer)
                gnorm = torch.tensor(0.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip > 0:
                try:
                    gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                except RuntimeError as e:
                    if "norm ops are not supported for complex yet" in str(e):
                        # Fallback for complex gradients: compute norm of complex grads
                        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                        total_norm = torch.tensor(0.0, device=self.model.parameters().__next__().device)
                        for g in grads:
                            if torch.is_complex(g):
                                total_norm += torch.linalg.vector_norm(g.real)**2 + torch.linalg.vector_norm(g.imag)**2
                            else:
                                total_norm += torch.linalg.vector_norm(g)**2
                        total_norm = torch.sqrt(total_norm)
                        gnorm = total_norm
                        if self.grad_clip > 0:
                            clip_coef = self.grad_clip / (total_norm + 1e-6)
                            if clip_coef < 1:
                                for p in self.model.parameters():
                                    if p.grad is not None:
                                        p.grad.mul_(clip_coef)
                    else:
                        raise e
            else:
                gnorm = torch.tensor(0.0)
            self.optimizer.step()
        
        return loss.item(), float(gnorm)

    def train(self, train_loader, t_start: float):
        import torch
        self.model.train()
        total_steps = 0
        max_grad_norm = 0.0
        total_train_time = 0
        t_last_log = t_start

        best_val = float("inf")
        best_state = None
        next_eval_progress = 0.10
        EVAL_INTERVAL = 0.10
        no_improve_count = 0
        
        initial_loss = None
        loss_at_50 = None

        while True:
            if self._is_budget_exceeded(t_start, initial_loss, loss_at_50):
                break

            progress = (time.time() - t_start) / self.time_budget
            if self.lr_schedule_fn:
                new_lr = self.lr_base * self.lr_schedule_fn(progress)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

            try:
                x, y = next(train_loader)
                x, y = to_device(x), to_device(y)
            except StopIteration:
                break

            t_step_start = time.time()
            loss_val, current_gnorm = self.train_step(x, y)
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
                self._loss_history.append([total_steps, loss_val])
                if len(self._loss_history) > 100:
                    self._loss_history = self._loss_history[-100:]
                
                if self.max_vram_gb > 0:
                    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                    if peak_mb / 1024 > self.max_vram_gb:
                        raise RuntimeError(f"VRAM limit exceeded: {peak_mb/1024:.2f} GB")

            if progress >= next_eval_progress:
                val = self.evaluate()
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
        import torch
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        
        with torch.no_grad():
            result = self.eval_fn(lambda x: self.compiled_model(to_device(x)))
        
        if self.ema:
            self.ema.restore()
        return result

class TrainerMLX(BaseTrainer):
    """A new implementation using mlx.core and mlx.optimizers."""
    def __init__(self, *args, **kwargs):
        # Remove torch-specific kwargs if present
        kwargs.pop("use_amp", None)
        kwargs.pop("compile", None)
        super().__init__(*args, **kwargs)
        if self.optimizer is None and HAS_MLX:
            self.optimizer = mxo.AdamW(learning_rate=self.lr_base)
        self._step_fn = None

    def train_step(self, x, y):
        import mlx.core as mx
        import mlx.nn as mx_nn
        if self._step_fn is None:
            def loss_fn(model, x, y):
                pred = model(x)
                return self.loss_fn(pred, y)

            loss_and_grad_fn = mx_nn.value_and_grad(self.model, loss_fn)

            @mx.compile
            def step(x, y):
                loss, grads = loss_and_grad_fn(self.model, x, y)
                self.optimizer.update(self.model, grads)
                return loss
            self._step_fn = step

        loss = self._step_fn(x, y)
        mx.eval(self.model.parameters(), self.optimizer.state, loss)
        return loss.item(), 0.0

    def train(self, train_loader, t_start: float):
        import mlx.core as mx
        total_steps = 0
        total_train_time = 0
        best_val = float("inf")
        initial_loss = None
        loss_at_50 = None
        
        next_eval_progress = 0.10
        EVAL_INTERVAL = 0.10
        no_improve_count = 0
        t_last_log = t_start

        self.model.train()
        while True:
            if self._is_budget_exceeded(t_start, initial_loss, loss_at_50):
                break

            progress = (time.time() - t_start) / self.time_budget
            if self.lr_schedule_fn:
                self.optimizer.learning_rate = self.lr_base * self.lr_schedule_fn(progress)

            try:
                x, y = next(train_loader)
                if not isinstance(x, mx.array): x = mx.array(x)
                if not isinstance(y, mx.array): y = mx.array(y)
            except StopIteration:
                break

            t_step_start = time.time()
            loss_val, _ = self.train_step(x, y)
            
            if initial_loss is None: initial_loss = loss_val
            if progress >= 0.5 and loss_at_50 is None: loss_at_50 = loss_val
            
            self._loss_history.append([total_steps, loss_val])
            if len(self._loss_history) > 100:
                self._loss_history = self._loss_history[-100:]

            if self.step_callback:
                self.step_callback(total_steps, progress)

            total_train_time += (time.time() - t_step_start)
            total_steps += 1

            if total_steps % 20 == 0:
                t_now = time.time()
                dt_ms = (t_now - t_last_log) / 20 * 1000
                remaining = max(0.0, self.time_budget - (t_now - t_start))
                print(f"step {total_steps:05d} ({progress*100:.1f}%) | loss: {loss_val:.6f} | "
                      f"lr: {self.optimizer.learning_rate:.2e} | "
                      f"dt: {dt_ms:.0f}ms | remaining: {remaining:.0f}s", flush=True)
                t_last_log = t_now

            if progress >= next_eval_progress:
                val = self.evaluate()
                if val < best_val:
                    best_val = val
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                print(f"val@{progress*100:.0f}%: {val:.6f}", flush=True)
                next_eval_progress += EVAL_INTERVAL
                self.model.train()

                if self.patience > 0 and no_improve_count >= self.patience:
                    break

        return total_steps, 0.0, total_train_time

    def evaluate(self):
        import mlx.core as mx
        self.model.eval()
        val = self.eval_fn(lambda x: self.model(mx.array(x) if not isinstance(x, mx.array) else x))
        self.model.train()
        return val

def Trainer(*args, **kwargs):
    """Factory function that dispatches to the correct backend based on model type."""
    model = kwargs.get("model") or (args[0] if args else None)
    
    # Detect MLX model
    is_mlx = False
    if HAS_MLX:
        import mlx.nn as mx_nn
        if isinstance(model, mx_nn.Module):
            is_mlx = True
    
    if is_mlx:
        print("[Trainer] Detected MLX model. Using TrainerMLX.", flush=True)
        return TrainerMLX(*args, **kwargs)
    else:
        print("[Trainer] Using TrainerTorch.", flush=True)
        return TrainerTorch(*args, **kwargs)

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
