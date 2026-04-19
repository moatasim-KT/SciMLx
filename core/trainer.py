"""Unified Trainer for SciML experiments on Apple Silicon (MLX)."""

import time
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW
from mlx.utils import tree_flatten, tree_map
from typing import Callable, Any, Dict, Optional, Tuple
from core.utils import TELEMETRY_DIR

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
        step_callback: Optional[Callable[[int, float], None]] = None,
        patience: int = 5,
        n_ensemble: int = 1,
        ema_decay: float = 0.0,
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
        self.step_callback = step_callback
        self.patience = patience          # evals without improvement before stopping (0 = disabled)
        self.n_ensemble = n_ensemble      # number of snapshots to save
        self.ema_decay = ema_decay        # 0 = disabled; 0.999 recommended
        self._loss_history: list = []   # rolling (step, loss) pairs for live telemetry
        self._snapshot_val_scores: list = []  # val scores at each snapshot (for weighted ensemble)
        # Per-experiment telemetry file: .vram_telemetry_<name> so parallel runs don't clobber each other
        from pathlib import Path as _Path
        _slug = exp_name.replace("/", "_").replace(" ", "_") if exp_name else ""
        self._telemetry_path = TELEMETRY_DIR / (
            f".vram_telemetry_{_slug}" if _slug else ".vram_telemetry"
        )

        # JIT compilation
        self.loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)

        # EMA shadow params (initialised lazily at first optimizer step)
        self._ema_params = None

    def train(self, train_loader, t_start: float):
        total_steps = 0
        max_grad_norm = 0.0
        current_gnorm = 0.0
        total_train_time = 0
        t_last_log = t_start

        # Best-checkpoint tracking: evaluate every 10% of budget, keep best weights
        best_val = float("inf")
        best_params = None
        next_eval_progress = 0.10   # first mid-run eval at 10%
        EVAL_INTERVAL = 0.10        # then every 10%
        no_improve_count = 0        # consecutive evals without improvement (early stopping)
        
        initial_loss = None
        loss_at_50 = None
        extensions_count = 0
        MAX_EXTENSIONS = 5
        
        # Snapshot ensemble tracking
        next_snapshot_progress = 1.0 / self.n_ensemble if self.n_ensemble > 1 else 2.0
        snapshot_count = 0

        while True:
            t_now = time.time()
            elapsed = t_now - t_start
            if elapsed > self.time_budget:
                # Dynamic budget extension: if loss is decreasing well at the end, add 20% more time
                if extensions_count < MAX_EXTENSIONS and initial_loss and loss_at_50 and len(self._loss_history) > 0:
                    # Heuristic: loss has dropped significantly from start and is still dropping from mid-point
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
                self.optimizer.lr = self.lr_base * self.lr_schedule_fn(progress)

            try:
                x, y = next(train_loader)
            except StopIteration:
                # Dataloader should be infinite for 5-min budget
                break

            # Curriculum: spectral smoothing — progressively reveal higher frequencies
            if self.curriculum and progress < 0.3:
                progress_c = progress / 0.3
                if x.ndim == 3:
                    # 1D path: ramp cutoff from 4 → 8 modes over first 30%
                    k_max = int(4 + progress_c * (8 - 4))
                    _, N, _ = x.shape
                    x_ft = mx.fft.rfft(x, axis=1)
                    y_ft = mx.fft.rfft(y, axis=1)
                    mask = mx.zeros_like(x_ft)
                    mask[:, :k_max, :] = 1.0
                    x = mx.fft.irfft(x_ft * mask, n=N, axis=1)
                    y = mx.fft.irfft(y_ft * mask, n=N, axis=1)
                elif x.ndim == 4:
                    # 2D path: ramp 2D spectral cutoff from 2 → n_modes over first 30%
                    B, N1, N2, C = x.shape
                    k_max = max(2, int(2 + progress_c * (min(N1, N2) // 4 - 2)))
                    x_2d = x.reshape(B * C, N1, N2) if C > 1 else x[..., 0]
                    y_2d = y.reshape(B * C, N1, N2) if C > 1 else y[..., 0]
                    x_ft = mx.fft.rfft2(x_2d, axes=(1, 2))
                    y_ft = mx.fft.rfft2(y_2d, axes=(1, 2))
                    mask = mx.zeros_like(x_ft)
                    mask[:, :k_max, :k_max] = 1.0
                    x_2d = mx.fft.irfft2(x_ft * mask, s=(N1, N2), axes=(1, 2))
                    y_2d = mx.fft.irfft2(y_ft * mask, s=(N1, N2), axes=(1, 2))
                    x = x_2d[..., None] if C == 1 else x_2d.reshape(B, N1, N2, C)
                    y = y_2d[..., None] if C == 1 else y_2d.reshape(B, N1, N2, C)


            t_step_start = time.time()
            loss, grads = self.loss_and_grad_fn(self.model, x, y)

            loss_val = loss.item()
            if initial_loss is None: initial_loss = loss_val
            if progress >= 0.5 and loss_at_50 is None: loss_at_50 = loss_val

            if math.isnan(loss_val) or math.isinf(loss_val):
                print(f"\nNaN/Inf loss detected at step {total_steps} — aborting.", flush=True)
                print(f"  lr={self.optimizer.lr:.2e}  grad_clip={self.grad_clip}", flush=True)
                print(f"  Diagnosis: likely lr too high or exploding gradients.", flush=True)
                print(f"  Fix: retry with lr/10 and grad_clip=5.0", flush=True)
                raise RuntimeError(f"NaN/Inf loss at step {total_steps}")

            if self.grad_clip > 0:
                grads, gnorm = clip_grad_norm(grads, self.grad_clip)
                current_gnorm = float(gnorm)
                max_grad_norm = max(max_grad_norm, current_gnorm)

            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)

            # EMA weight update
            if self.ema_decay > 0:
                if self._ema_params is None:
                    self._ema_params = tree_map(lambda p: mx.array(p), self.model.parameters())
                else:
                    d = self.ema_decay
                    self._ema_params = tree_map(
                        lambda e, p: d * e + (1.0 - d) * p,
                        self._ema_params, self.model.parameters()
                    )
                    mx.eval(self._ema_params)

            if self.step_callback:
                self.step_callback(total_steps, progress)

            total_train_time += (time.time() - t_step_start)
            total_steps += 1

            if total_steps % 20 == 0:
                t_now2 = time.time()
                dt_ms = (t_now2 - t_last_log) / 20 * 1000
                remaining = max(0.0, self.time_budget - (t_now2 - t_start))
                
                _recent_losses = [p[1] for p in self._loss_history[-20:]]
                _loss_volatility = (np.std(_recent_losses) / (np.mean(_recent_losses) + 1e-8)
                                    if len(_recent_losses) > 1 else 0.0)

                print(f"step {total_steps:05d} ({progress*100:.1f}%) | loss: {loss_val:.6f} | "
                      f"lr: {self.optimizer.lr:.2e} | gnorm: {current_gnorm:.3f} | "
                      f"dt: {dt_ms:.0f}ms | remaining: {remaining:.0f}s", flush=True)
                
                if total_steps % 100 == 0:
                    print(f"diag_stability_signature: volatility={_loss_volatility:.4f} gnorm_max={max_grad_norm:.3f}", flush=True)
                
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
                    "grad_norm":      current_gnorm,
                    "max_grad_norm":  max_grad_norm,
                }
                threading.Thread(target=_async_write, args=(self._telemetry_path, payload), daemon=True).start()

                if self.max_vram_gb > 0 and peak_mb / 1024 > self.max_vram_gb:
                    raise RuntimeError(
                        f"VRAM limit exceeded: {peak_mb/1024:.2f} GB > {self.max_vram_gb:.1f} GB"
                    )

            # Snapshot ensemble saving (at the end of each cycle)
            if progress >= next_snapshot_progress and snapshot_count < self.n_ensemble - 1:
                snapshot_count += 1
                # Evaluate snapshot (use EMA params if available)
                _snap_params = self.model.parameters()
                if self.ema_decay > 0 and self._ema_params is not None:
                    _snap_params = self._ema_params
                snap_val = self.eval_fn(lambda x: self.forward_fn(self.model, x))
                self._snapshot_val_scores.append(snap_val)
                if self.exp_name:
                    from core.utils import REPO_ROOT
                    ckpt_dir = REPO_ROOT / "checkpoints"
                    ckpt_dir.mkdir(exist_ok=True)
                    ckpt_path = ckpt_dir / f"{self.exp_name}_snapshot_{snapshot_count}.npz"
                    params_to_save = _snap_params
                    mx.savez(str(ckpt_path), **dict(tree_flatten(params_to_save)))
                    print(f"\n[Snapshot] Saved ensemble member {snapshot_count} "
                          f"(val={snap_val:.6f}) to {ckpt_path.name}", flush=True)
                next_snapshot_progress += 1.0 / self.n_ensemble

            # Mid-run validation: checkpoint best weights + early stopping
            if progress >= next_eval_progress:
                if self.ema_decay > 0 and self._ema_params is not None:
                    _live = tree_map(lambda p: mx.array(p), self.model.parameters())
                    self.model.update(self._ema_params)
                    mx.eval(self.model.parameters())
                    val = self.eval_fn(lambda x: self.forward_fn(self.model, x))
                    self.model.update(_live)
                    mx.eval(self.model.parameters())
                else:
                    val = self.eval_fn(lambda x: self.forward_fn(self.model, x))
                if val < best_val:
                    no_improve_count = 0
                    marker = " ✓ best"
                    best_val = val
                    from mlx.utils import tree_flatten, tree_unflatten
                    best_params = tree_unflatten(
                        [(k, v) for k, v in tree_flatten(self.model.parameters())]
                    )
                    # Periodic checkpointing to disk for resumption
                    if self.exp_name:
                        from core.utils import REPO_ROOT
                        ckpt_dir = REPO_ROOT / "checkpoints"
                        ckpt_dir.mkdir(exist_ok=True)
                        ckpt_path = ckpt_dir / f"{self.exp_name}_best.npz"
                        mx.savez(str(ckpt_path), **dict(tree_flatten(self.model.parameters())))
                        print(f"  [Checkpoint] Saved best weights so far to {ckpt_path.name}", flush=True)
                else:
                    no_improve_count += 1
                    marker = f" (no improvement {no_improve_count}/{self.patience})" if self.patience > 0 else ""

                print(f"val@{progress*100:.0f}%: {val:.6f}{marker}", flush=True)
                next_eval_progress += EVAL_INTERVAL

                # Early stopping: halt if val hasn't improved for `patience` consecutive evals
                if self.patience > 0 and no_improve_count >= self.patience:
                    print(
                        f"\n[EarlyStopping] No improvement for {self.patience} consecutive evals "
                        f"(best={best_val:.6f}). Stopping at {progress*100:.0f}% of budget.",
                        flush=True,
                    )
                    print("diag_early_stopped=True", flush=True)
                    break

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
        # Standard single-model evaluation (use EMA params if available)
        if self.n_ensemble <= 1:
            if self.ema_decay > 0 and self._ema_params is not None:
                _live = tree_map(lambda p: mx.array(p), self.model.parameters())
                self.model.update(self._ema_params)
                mx.eval(self.model.parameters())
                result = self.eval_fn(lambda x: self.forward_fn(self.model, x))
                self.model.update(_live)
                mx.eval(self.model.parameters())
                return result
            return self.eval_fn(lambda x: self.forward_fn(self.model, x))
            
        # Ensemble evaluation
        print(f"\n[Ensemble] Evaluating ensemble of {self.n_ensemble} members...")
        
        # 1. Collect all checkpoint paths
        from core.utils import REPO_ROOT
        ckpt_dir = REPO_ROOT / "checkpoints"
        snapshots = []
        
        # Best model (always member 0 or implicitly the current state)
        # But for consistency, we use the saved best file if it exists
        best_path = ckpt_dir / f"{self.exp_name}_best.npz"
        if best_path.exists():
            snapshots.append(best_path)
            
        # Add other snapshots
        for i in range(1, self.n_ensemble):
            snap_path = ckpt_dir / f"{self.exp_name}_snapshot_{i}.npz"
            if snap_path.exists():
                snapshots.append(snap_path)
                
        if not snapshots:
            print("[Ensemble] Warning: No snapshots found, falling back to single model.")
            return self.eval_fn(lambda x: self.forward_fn(self.model, x))
            
        print(f"[Ensemble] Found {len(snapshots)} snapshots.")
        
        # 2. Define ensemble prediction function
        def ensemble_pred_fn(x):
            preds = []
            current_params = tree_map(lambda p: mx.array(p), self.model.parameters())

            for path in snapshots:
                self.model.load_weights(str(path))
                mx.eval(self.model.parameters())
                preds.append(self.forward_fn(self.model, x))

            self.model.update(current_params)
            mx.eval(self.model.parameters())

            stacked = mx.stack(preds, axis=0)  # [M, B, ...]

            # Weighted ensemble: inverse-val-error weights (better snapshots get more weight)
            scores = self._snapshot_val_scores[-len(snapshots):]
            if len(scores) == len(snapshots) and len(scores) > 1:
                import numpy as _np
                w = 1.0 / (_np.array(scores) + 1e-8)
                w = w / w.sum()
                w_mx = mx.array(w, dtype=mx.float32)
                # reshape weights to broadcast: [M, 1, 1, ...]
                shape = [len(w)] + [1] * (stacked.ndim - 1)
                w_mx = w_mx.reshape(shape)
                return mx.sum(stacked * w_mx, axis=0)
            return mx.mean(stacked, axis=0)
            
        # 3. Evaluate using the averaged predictions
        return self.eval_fn(ensemble_pred_fn)

def get_lr_schedule(
    warmup_ratio: float = 0.05,
    wardown_ratio: float = 0.2,
    final_lr_frac: float = 0.01,
    cyclical: bool = False,
    n_cycles: int = 1,
    schedule_type: str = "warmup_cosine",
) -> Callable[[float], float]:
    """Return a progress→lr_multiplier function for the chosen schedule.

    schedule_type options:
      "warmup_cosine"  (default) — linear warmup → flat → cosine wardown
      "cosine"         — pure cosine annealing from 1.0 to final_lr_frac
      "onecycle"       — linear ramp to peak for first 30%, cosine decay for rest
      "none"           — constant 1.0 (no schedule)
    """
    def lr_schedule(progress: float) -> float:
        if schedule_type == "none":
            return 1.0

        if schedule_type == "cosine":
            # Pure cosine: 0.5*(1+cos(pi*p)) scaled to [final_lr_frac, 1.0]
            return 0.5 * (1.0 + math.cos(math.pi * progress)) * (1.0 - final_lr_frac) + final_lr_frac

        if schedule_type == "onecycle":
            # Linear warmup to peak for first 30%, cosine decay back to final_lr_frac
            peak_frac = 0.3
            if progress < peak_frac:
                return progress / peak_frac
            t = (progress - peak_frac) / (1.0 - peak_frac)
            return 0.5 * (1.0 + math.cos(math.pi * t)) * (1.0 - final_lr_frac) + final_lr_frac

        # Default: "warmup_cosine" — linear warmup → flat → cosine wardown (single or cyclical)
        if not cyclical:
            if progress < warmup_ratio:
                return progress / warmup_ratio if warmup_ratio > 0 else 1.0
            if progress < 1.0 - wardown_ratio:
                return 1.0
            t = (1.0 - (progress - (1.0 - wardown_ratio)) / wardown_ratio)
            return t + (1 - t) * final_lr_frac
        else:
            # Cyclical cosine annealing
            cycle_progress = (progress * n_cycles) % 1.0
            return 0.5 * (1.0 + math.cos(math.pi * cycle_progress)) * (1.0 - final_lr_frac) + final_lr_frac

    return lr_schedule
