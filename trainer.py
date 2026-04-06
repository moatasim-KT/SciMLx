"""Unified Trainer for SciML experiments on Apple Silicon (MLX)."""

import time
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from typing import Callable, Any, Dict, Optional, Tuple

def _set(model, path, val):
    parts = path.split(".")
    obj   = model
    for p in parts[:-1]:
        obj = obj[int(p)] if isinstance(obj, list) else (
              obj[p]      if isinstance(obj, dict)  else getattr(obj, p))
    last = parts[-1]
    if   isinstance(obj, list): obj[int(last)] = val
    elif isinstance(obj, dict): obj[last]      = val
    else:                       setattr(obj, last, val)

class AdamW:
    """AdamW with runtime learning-rate control and robust MLX parameter updates."""

    def __init__(self, lr: float, weight_decay: float,
                 betas=(0.9, 0.999), eps: float = 1e-8):
        self.lr   = lr
        self.wd   = weight_decay
        self.b1, self.b2 = betas
        self.eps  = eps
        self._s: dict = {}
        self._t   = 0

    def update(self, model: nn.Module, grads: Any):
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
            _set(model, path, p.astype(flat_p[path].dtype))

    @property
    def state_arrays(self):
        out = []
        for s in self._s.values():
            out += [s["m"], s["v"]]
        return out

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

def clip_grad_norm(grads, max_norm: float) -> Tuple[Any, float]:
    """Clip gradient tree by global L2 norm. Returns (clipped_grads, norm)."""
    flat_g = dict(tree_flatten(grads))
    sq_sum = sum(float(mx.sum(g * g).item()) for g in flat_g.values())
    norm   = sq_sum ** 0.5
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        def _scale(tree):
            if isinstance(tree, mx.array): return tree * scale
            if isinstance(tree, dict): return {k: _scale(v) for k, v in tree.items()}
            if isinstance(tree, list): return [_scale(v) for v in tree]
            return tree
        grads = _scale(grads)
    return grads, norm

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
        
        # JIT compilation
        self.loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)

    def train(self, train_loader, t_start: float):
        total_steps = 0
        max_grad_norm = 0
        total_train_time = 0
        
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

            t_step_start = time.time()
            loss, grads = self.loss_and_grad_fn(self.model, x, y)
            
            if self.grad_clip > 0:
                grads, gnorm = clip_grad_norm(grads, self.grad_clip)
                max_grad_norm = max(max_grad_norm, gnorm)
            
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state_arrays)
            
            total_train_time += (time.time() - t_step_start)
            total_steps += 1
            
            if total_steps % 20 == 0:
                print(f"Step {total_steps:5d} | loss {loss.item():.4f} | "
                      f"lr {self.optimizer.lr:.2e} | {elapsed:.1f}s", flush=True)

        print()
        return total_steps, max_grad_norm, total_train_time

    def evaluate(self):
        return self.eval_fn(lambda x: self.forward_fn(self.model, x))
