import torch.nn as nn
import torch
import inspect
from typing import Callable, Dict, Any, Optional, Type, List

# Must stay in sync with BENCHMARKS_2D in autorun.py.
_2D_BENCHMARKS = frozenset({
    "darcy_2d", "ns_2d", "swe_2d", "allen_cahn_2d", "ns_hre_2d", "mhd_2d",
    "elasticity_2d", "wavebench_2d", "pdebench_2d", "multiphysics_2d",
    "radiative_2d", "poisson_2d", "ellipse_2d"
})

# ── Model Registry ────────────────────────────────────────────────────────────

class ModelRegistry:
    def __init__(self):
        self._registry: Dict[str, Callable] = {}
        self._lazy_imports: Dict[str, tuple[str, str]] = {}

    def register_lazy(self, name: str, module_name: str, class_name: str) -> None:
        self._lazy_imports[name] = (module_name, class_name)

    def register(self, name: str) -> Callable:
        def _decorator(fn: Callable) -> Callable:
            self._registry[name] = fn
            return fn
        return _decorator

    def build(self, name: str, benchmark: str = "", **kwargs) -> nn.Module:
        if name not in self._registry and name in self._lazy_imports:
            mod_name, cls_name = self._lazy_imports[name]
            import importlib
            module = importlib.import_module(f"models.{mod_name}")
            cls = getattr(module, cls_name)
            self.register_class(name, cls)
            
        if name not in self._registry:
            available = ", ".join(sorted(set(self._registry) | set(self._lazy_imports)))
            raise ValueError(f"Unknown model {name!r}. Available: {available}")

        # Map n_modes -> (n_modes1, n_modes2) for 2D models
        if (name.endswith("2D") or name.endswith("2d")) and "n_modes" in kwargs:
            m = kwargs.pop("n_modes")
            kwargs["n_modes1"] = m
            kwargs["n_modes2"] = m

        return self._registry[name](**kwargs)

    def register_class(self, name: str, cls: Type[nn.Module], **fixed_kwargs) -> None:
        def _factory(**kwargs):
            kwargs.update(fixed_kwargs)
            sig = inspect.signature(cls.__init__)
            valid_args = {
                k: v for k, v in kwargs.items()
                if k in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            }
            return cls(**valid_args)
        _factory.__name__ = f"factory_{name}"
        self._registry[name] = _factory

    @property
    def available(self):
        return sorted(set(self._registry) | set(self._lazy_imports))

MODEL_REGISTRY = ModelRegistry()

# ── Benchmark Registry ────────────────────────────────────────────────────────

class BenchmarkRegistry:
    def __init__(self):
        self._loaders:  Dict[str, Callable] = {}
        self._evals:    Dict[str, Callable] = {}
        self._sota:     Dict[str, float]    = {}
        self._desc:     Dict[str, str]      = {}

    def register(self, name: str, make_loader: Callable, evaluate: Callable, sota: Optional[float] = None, description: str = "") -> None:
        self._loaders[name] = make_loader
        self._evals[name]   = evaluate
        self._sota[name]    = sota
        self._desc[name]    = description

    def make_loader(self, name: str, split: str, batch_size: int):
        if name not in self._loaders: raise ValueError(f"Unknown benchmark {name!r}")
        return self._loaders[name](name, split, batch_size)

    def evaluate(self, name: str, model_fn: Callable) -> float:
        if name not in self._evals: raise ValueError(f"Unknown benchmark {name!r}")
        return self._evals[name](name, model_fn)

BENCHMARK_REGISTRY = BenchmarkRegistry()

# ── Populate registries from existing codebase ────────────────────────────────

def _register_defaults():
    from data.prepare import make_dataloader, evaluate_l2_rel
    from data.benchmarks_ext import EXT_BENCHMARKS, EXT_SOTA, make_ext_dataloader, evaluate_l2_rel_ext

    # ── Model Registrations (Lazy) ───────────────────────────────────────────
    MODEL_REGISTRY.register_lazy("FNO", "fno", "FNO1d")
    MODEL_REGISTRY.register_lazy("RFNO", "fno", "RFNO1d")
    MODEL_REGISTRY.register_lazy("AFNO", "afno", "AFNO1d")
    MODEL_REGISTRY.register_lazy("FFNO", "afno", "FFNO1d")
    MODEL_REGISTRY.register_lazy("UNO", "fno", "UNO1d")
    MODEL_REGISTRY.register_lazy("WNO", "wno", "WNO1d")
    MODEL_REGISTRY.register_lazy("KAN_FNO", "kan", "KAN_FNO")
    MODEL_REGISTRY.register_lazy("cPIKAN_FNO", "chebyshev_kan", "cPIKAN_FNO")
    MODEL_REGISTRY.register_lazy("DeepONet", "deeponet", "DeepONet")
    MODEL_REGISTRY.register_lazy("PODDeepONet", "deeponet", "PODDeepONet")
    MODEL_REGISTRY.register_lazy("S4NO", "s4d", "S4NO1d")
    MODEL_REGISTRY.register_lazy("SSNO", "ssno", "SSNO1d")
    MODEL_REGISTRY.register_lazy("UNO2d", "fno", "UNO2d")
    MODEL_REGISTRY.register_lazy("WNO2d", "wno", "WNO2d")
    MODEL_REGISTRY.register_lazy("GNOT", "gnot", "GNOT1d")
    MODEL_REGISTRY.register_lazy("GNOT2d", "gnot", "GNOT2d")
    MODEL_REGISTRY.register_lazy("MambaNO", "mamba_no", "MambaNO1d")
    MODEL_REGISTRY.register_lazy("PINO", "pinn", "PINO1d")
    MODEL_REGISTRY.register_lazy("PINN", "pinn", "PINN")
    MODEL_REGISTRY.register_lazy("ModalPINN", "pinn", "ModalPINN")
    MODEL_REGISTRY.register_lazy("FNO2D", "fno", "FNO2d")
    MODEL_REGISTRY.register_lazy("TFNO", "tfno", "TFNO1d")
    MODEL_REGISTRY.register_lazy("TFNO2D", "tfno", "TFNO2d")
    MODEL_REGISTRY.register_lazy("Transolver", "transolver", "Transolver1d")
    MODEL_REGISTRY.register_lazy("Transolver2D", "transolver", "Transolver2d")
    MODEL_REGISTRY.register_lazy("TimeDeepONet", "time_deeponet", "TimeDeepONet1d")
    MODEL_REGISTRY.register_lazy("HNN", "hnn", "HamiltonianNO1d")
    MODEL_REGISTRY.register_lazy("NeuralODE", "neural_ode", "NeuralODE1d")
    MODEL_REGISTRY.register_lazy("PACMANN", "pacmann", "PACMANN")
    MODEL_REGISTRY.register_lazy("VSMNO2D", "vsmno", "VSMNO2d")
    MODEL_REGISTRY.register_lazy("MemNO", "mem_no", "MemNO1d")
    MODEL_REGISTRY.register_lazy("SARModel2d", "sar", "SARModel2d")

    # ── Benchmarks ───────────────────────────────────────────────────────────
    BENCHMARK_REGISTRY.register("burgers_1d", make_dataloader, evaluate_l2_rel, 0.0149, "1D Burgers")
    for bm in EXT_BENCHMARKS:
        BENCHMARK_REGISTRY.register(bm, make_ext_dataloader, evaluate_l2_rel_ext, EXT_SOTA.get(bm))

    # ── Simulation Benchmarks ───────────────────────────────────────────────
    from data.simulations import SIM_BENCHMARKS, SIM_SOTA, SIM_METADATA, make_sim_dataloader, evaluate_l2_rel_sim
    for bm in SIM_BENCHMARKS:
        BENCHMARK_REGISTRY.register(bm, make_sim_dataloader, evaluate_l2_rel_sim, SIM_SOTA.get(bm))

_register_defaults()
