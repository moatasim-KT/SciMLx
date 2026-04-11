"""Unified Trainer for SciML experiments on Apple Silicon (MLX)."""

import time
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW
from mlx.utils import tree_flatten, tree_map
from typing import Callable, Any, Dict, Optional, Tuple

def clip_grad_norm(grads, max_norm: float) -> Tuple[Any, float]:
    """Clip gradient tree by global L2 norm. Returns (clipped_grads, norm)."""
    norm = mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(grads)))
    scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
    from mlx.utils import tree_map
    return tree_map(lambda g: g * scale, grads), norm


class Trainer:
    """Encapsulates training loop, evaluation, and metrics."""
    def __init__(
        self,
        model: nn.Module,
        optimizer: AdamW,
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
        self.exp_name = exp_name
        self._loss_history: list = []   # rolling (step, loss) pairs for live telemetry
        # Per-experiment telemetry file: .vram_telemetry_<name> so parallel runs don't clobber each other
        from pathlib import Path as _Path
        _slug = exp_name.replace("/", "_").replace(" ", "_") if exp_name else ""
        self._telemetry_path = _Path(__file__).resolve().parent / (
            f".vram_telemetry_{_slug}" if _slug else ".vram_telemetry"
        )

        # JIT compilation
        self.loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)

    def train(self, train_loader, t_start: float):
        total_steps = 0
        max_grad_norm = 0
        total_train_time = 0
        t_last_log = t_start

        # Best-checkpoint tracking: evaluate every 10% of budget, keep best weights
        best_val = float("inf")
        best_params = None
        next_eval_progress = 0.10   # first mid-run eval at 10%
        EVAL_INTERVAL = 0.10        # then every 10%

        while True:
            t_now = time.time()
            elapsed = t_now - t_start
            if elapsed > self.time_budget:
                break

            progress = elapsed / self.time_budget
            if self.lr_schedule_fn:
                self.optimizer.lr = self.lr_base * self.lr_schedule_fn(progress)

            try:
                x, y = next(train_loader)
            except StopIteration:
                # Dataloader should be infinite for 5-min budget
                break

            # Curriculum: smoothing for 1D benchmarks (Burgers shocks)
            # Progress 0.0 -> 0.5: smooth with decreasing kernel
            if self.curriculum and x.ndim == 3 and progress < 0.5:
                # Simple spatial moving average (low-pass filter)
                # Max kernel size 7 at t=0, 1 at t=0.5
                # Vectorized moving average using conv1d
                k = int(7 * (1 - progress / 0.5))
                if k > 1:
                    if k % 2 == 0: k += 1
                    weight = mx.ones((x.shape[-1], 1, k)) / k
                    # conv1d expects [B, N, C], permute to [B, C, N]
                    x_c = mx.transpose(x, (0, 2, 1))
                    y_c = mx.transpose(y, (0, 2, 1))
                    x = mx.transpose(nn.conv1d(x_c, weight, padding=k//2, groups=x.shape[-1]), (0, 2, 1))
                    y = mx.transpose(nn.conv1d(y_c, weight, padding=k//2, groups=x.shape[-1]), (0, 2, 1))


            t_step_start = time.time()
            loss, grads = self.loss_and_grad_fn(self.model, x, y)

            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                print(f"\nNaN/Inf loss detected at step {total_steps} — aborting.", flush=True)
                print(f"  lr={self.optimizer.lr:.2e}  grad_clip={self.grad_clip}", flush=True)
                print(f"  Diagnosis: likely lr too high or exploding gradients.", flush=True)
                print(f"  Fix: retry with lr/10 and grad_clip=5.0", flush=True)
                raise RuntimeError(f"NaN/Inf loss at step {total_steps}")

            if self.grad_clip > 0:
                grads, gnorm = clip_grad_norm(grads, self.grad_clip)
                max_grad_norm = max(max_grad_norm, gnorm)

            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)

            total_train_time += (time.time() - t_step_start)
            total_steps += 1

            if total_steps % 20 == 0:
                t_now2 = time.time()
                dt_ms = (t_now2 - t_last_log) / 20 * 1000
                remaining = max(0.0, self.time_budget - (t_now2 - t_start))
                print(f"step {total_steps:05d} ({progress*100:.1f}%) | loss: {loss.item():.6f} | "
                      f"lr: {self.optimizer.lr:.2e} | dt: {dt_ms:.0f}ms | remaining: {remaining:.0f}s", flush=True)
                t_last_log = t_now2

                # VRAM guard + live telemetry file every 20 steps
                try:
                    active_mb = mx.metal.get_active_memory() / 1024 / 1024
                    peak_mb   = mx.get_peak_memory() / 1024 / 1024
                except Exception:
                    active_mb = peak_mb = 0.0
                # Asynchronous telemetry update
                import threading
                import json as _json

                def _async_write(path, data):
                    try:
                        path.write_text(_json.dumps(data))
                    except Exception as e:
                        print(f"[telemetry write error: {e}]", flush=True)

                self._loss_history.append([total_steps, loss_val])
                if len(self._loss_history) > 100:
                    self._loss_history = self._loss_history[-100:]
                
                payload = {
                    "experiment":     self.exp_name,
                    "vram_active_mb": active_mb,
                    "vram_peak_mb":   peak_mb,
                    "progress":       progress,
                    "remaining_s":    max(0.0, self.time_budget - (t_now2 - t_start)),
                    "step":           total_steps,
                    "loss":           loss_val,
                    "loss_history":   self._loss_history,
                }
                threading.Thread(target=_async_write, args=(self._telemetry_path, payload), daemon=True).start()

                if self.max_vram_gb > 0 and peak_mb / 1024 > self.max_vram_gb:
                    raise RuntimeError(
                        f"VRAM limit exceeded: {peak_mb/1024:.2f} GB > {self.max_vram_gb:.1f} GB"
                    )

            # Mid-run validation: checkpoint best weights
            if progress >= next_eval_progress:
                val = self.eval_fn(lambda x: self.forward_fn(self.model, x))
                marker = " ✓ best" if val < best_val else ""
                print(f"val@{progress*100:.0f}%: {val:.6f}{marker}", flush=True)
                if val < best_val:
                    best_val = val
                    from mlx.utils import tree_flatten, tree_unflatten
                    best_params = tree_unflatten(
                        [(k, v) for k, v in tree_flatten(self.model.parameters())]
                    )
                next_eval_progress += EVAL_INTERVAL

        print()
        # Restore best weights found during training before final evaluation
        if best_params is not None:
            self.model.update(best_params)
            mx.eval(self.model.parameters())
            print(f"Restored best checkpoint (val={best_val:.6f})", flush=True)
        # Clean up per-experiment telemetry file so dashboard stops showing stale data
        try:
            self._telemetry_path.unlink(missing_ok=True)
        except Exception:
            pass
        return total_steps, max_grad_norm, total_train_time

    def evaluate(self):
        return self.eval_fn(lambda x: self.forward_fn(self.model, x))

def get_lr_schedule(warmup_ratio: float, wardown_ratio: float, final_lr_frac: float) -> Callable[[float], float]:
    def lr_schedule(progress: float) -> float:
        """Linear warmup → flat → cosine warmdown."""
        if progress < warmup_ratio:
            return progress / warmup_ratio if warmup_ratio > 0 else 1.0
        if progress < 1.0 - wardown_ratio:
            return 1.0
        t = (1.0 - progress) / wardown_ratio
        return t + (1 - t) * final_lr_frac
    return lr_schedule
