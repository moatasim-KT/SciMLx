"""Research plugin system for autonomous SciML experimentation.

Provides:
  1. ModelRegistry — auto-discovers and registers models without editing train.py
  2. BenchmarkRegistry — maps benchmark names to dataloader/eval functions
  3. Abstract interfaces aligned with MLX's nn.Module pattern

Adding a new model:
    1. Implement in models/<name>.py (subclass nn.Module, standard __call__)
    2. Register via @MODEL_REGISTRY.register("MYMODEL") decorator, or
       call MODEL_REGISTRY.register_class("MYMODEL", MyModel1d) in models/__init__.py
    3. Add ExperimentConfig entries in experiments.py — no changes to train.py needed

Adding a new benchmark:
    1. Implement solver in benchmarks_ext.py
    2. Call BENCHMARK_REGISTRY.register(...) with make_dataloader and evaluate_fn
    3. train.py routing picks it up automatically
"""

from __future__ import annotations

import mlx.nn as nn
import mlx.core as mx
from typing import Callable, Dict, Any, Optional, Type


# ── Model Registry ────────────────────────────────────────────────────────────

class ModelRegistry:
    """Central registry mapping MODEL_TYPE strings to factory callables.

    Usage:
        # Register a class (called with n_modes, hidden_dim, n_layers kwargs)
        MODEL_REGISTRY.register_class("FNO", FNO1d)

        # Register a custom factory
        @MODEL_REGISTRY.register("MYFNO")
        def _make_myfno(n_modes, hidden_dim, n_layers, **kw):
            return MyFNO1d(n_modes, hidden_dim, n_layers, block_size=kw.get("block_size", 32))

        # Instantiate from registry
        model = MODEL_REGISTRY.build("FNO", n_modes=24, hidden_dim=128, n_layers=8)
    """

    def __init__(self):
        self._registry: Dict[str, Callable] = {}

    def register_class(self, name: str, cls: Type[nn.Module],
                       **fixed_kwargs) -> None:
        """Register an nn.Module class. fixed_kwargs are always passed."""
        def _factory(**kwargs):
            kwargs.update(fixed_kwargs)
            return cls(**kwargs)
        _factory.__name__ = f"factory_{name}"
        self._registry[name] = _factory

    def register(self, name: str) -> Callable:
        """Decorator to register a factory function."""
        def _decorator(fn: Callable) -> Callable:
            self._registry[name] = fn
            return fn
        return _decorator

    def build(self, name: str, **kwargs) -> nn.Module:
        """Instantiate a registered model."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise ValueError(
                f"Unknown model {name!r}. Available: {available}"
            )
        return self._registry[name](**kwargs)

    @property
    def available(self):
        return sorted(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry


MODEL_REGISTRY = ModelRegistry()


# ── Benchmark Registry ────────────────────────────────────────────────────────

class BenchmarkRegistry:
    """Maps benchmark names to (make_dataloader, evaluate_fn) pairs.

    Usage:
        BENCHMARK_REGISTRY.register(
            "my_pde",
            make_loader=make_my_dataloader,
            evaluate=evaluate_my_l2_rel,
            sota=0.005,
            description="My PDE description",
        )

        loader = BENCHMARK_REGISTRY.make_loader("my_pde", "train", batch_size=32)
        score  = BENCHMARK_REGISTRY.evaluate("my_pde", model)
    """

    def __init__(self):
        self._loaders:  Dict[str, Callable] = {}
        self._evals:    Dict[str, Callable] = {}
        self._sota:     Dict[str, float]    = {}
        self._desc:     Dict[str, str]      = {}

    def register(self, name: str,
                 make_loader: Callable,
                 evaluate: Callable,
                 sota: Optional[float] = None,
                 description: str = "") -> None:
        self._loaders[name] = make_loader
        self._evals[name]   = evaluate
        self._sota[name]    = sota
        self._desc[name]    = description

    def make_loader(self, name: str, split: str, batch_size: int):
        if name not in self._loaders:
            raise ValueError(f"Unknown benchmark {name!r}")
        return self._loaders[name](name, split, batch_size)

    def evaluate(self, name: str, model_fn: Callable) -> float:
        if name not in self._evals:
            raise ValueError(f"Unknown benchmark {name!r}")
        return self._evals[name](name, model_fn)

    def sota(self, name: str) -> Optional[float]:
        return self._sota.get(name)

    @property
    def available(self):
        return sorted(self._loaders)

    def __contains__(self, name: str) -> bool:
        return name in self._loaders


BENCHMARK_REGISTRY = BenchmarkRegistry()


# ── Populate registries from existing codebase ────────────────────────────────

def _register_defaults():
    """Register all built-in models and benchmarks."""
    from models import (
        FNO1d, FNO2d, UNO1d, RFNO1d,
        AFNO1d, FFNO1d,
        WNO1d, DeepONet, PODDeepONet,
    )
    from prepare import GRID_SIZE, make_dataloader, evaluate_l2_rel
    from benchmarks_ext import (
        EXT_BENCHMARKS, make_ext_dataloader, evaluate_l2_rel_ext
    )

    # ── 1-D models (factories accept **kw to absorb unused params) ───────────
    @MODEL_REGISTRY.register("FNO")
    def _make_fno(n_modes=16, hidden_dim=64, n_layers=4, **kw):
        return FNO1d(n_modes=n_modes, hidden_dim=hidden_dim, n_layers=n_layers)

    @MODEL_REGISTRY.register("RFNO")
    def _make_rfno(n_modes=16, hidden_dim=64, n_layers=4, **kw):
        return RFNO1d(n_modes=n_modes, hidden_dim=hidden_dim, n_layers=n_layers)

    @MODEL_REGISTRY.register("AFNO")
    def _make_afno(n_modes=16, hidden_dim=64, n_layers=4, **kw):
        return AFNO1d(n_modes=n_modes, hidden_dim=hidden_dim, n_layers=n_layers)

    @MODEL_REGISTRY.register("FFNO")
    def _make_ffno(n_modes=16, hidden_dim=64, n_layers=4, **kw):
        return FFNO1d(n_modes=n_modes, hidden_dim=hidden_dim, n_layers=n_layers)

    @MODEL_REGISTRY.register("UNO")
    def _make_uno(n_modes=16, hidden_dim=64, n_layers=4, **kw):
        return UNO1d(n_modes=n_modes, hidden_dim=hidden_dim, n_layers=n_layers)

    @MODEL_REGISTRY.register("WNO")
    def _make_wno(n_modes=16, hidden_dim=64, n_layers=4, n_levels=3, **kw):
        return WNO1d(n_levels=n_levels, hidden_dim=hidden_dim, n_layers=n_layers)

    @MODEL_REGISTRY.register("DeepONet")
    def _make_deeponet(n_modes=16, hidden_dim=64, n_layers=4, **kw):
        return DeepONet(branch_dim=GRID_SIZE, trunk_dim=1,
                        hidden_dim=hidden_dim, out_dim=hidden_dim,
                        n_layers=n_layers)

    @MODEL_REGISTRY.register("PODDeepONet")
    def _make_pod(n_modes=16, hidden_dim=64, n_layers=4, **kw):
        return PODDeepONet(branch_dim=GRID_SIZE, n_basis=hidden_dim,
                           hidden_dim=hidden_dim, n_layers=n_layers)

    @MODEL_REGISTRY.register("FNO2D")
    def _make_fno2d(n_modes=12, hidden_dim=64, n_layers=4, **kw):
        return FNO2d(n_modes1=n_modes, n_modes2=n_modes,
                     hidden_dim=hidden_dim, n_layers=n_layers)

    # ── Benchmarks ────────────────────────────────────────────────────────────
    _std = {"burgers_1d", "darcy_2d"}
    for bm in _std:
        BENCHMARK_REGISTRY.register(
            bm,
            make_loader=make_dataloader,
            evaluate=lambda name, fn: evaluate_l2_rel(name, fn),
            sota={"burgers_1d": 0.0149, "darcy_2d": 0.0108}.get(bm),
            description={"burgers_1d": "1D viscous Burgers",
                         "darcy_2d": "2D steady Darcy"}.get(bm, ""),
        )
    for bm in EXT_BENCHMARKS:
        BENCHMARK_REGISTRY.register(
            bm,
            make_loader=make_ext_dataloader,
            evaluate=lambda name, fn: evaluate_l2_rel_ext(name, fn),
            sota={"kdv_1d": 0.010, "wave_1d": 0.005}.get(bm),
            description={"kdv_1d": "KdV soliton (ETDRK4)",
                         "wave_1d": "1D wave u_tt=c²u_xx"}.get(bm, ""),
        )


_register_defaults()
