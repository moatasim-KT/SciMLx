import mlx.nn as nn
import mlx.core as mx
import inspect
from typing import Callable, Dict, Any, Optional, Type, List

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
        # Safety constraint: 2D benchmark model size limits
        _2d_benchmarks = {"darcy_2d", "ns_2d", "swe_2d", "allen_cahn_2d", "ns_hre_2d", "mhd_2d"}
        if benchmark in _2d_benchmarks:
            hidden = kwargs.get("hidden_dim", 64)
            layers = kwargs.get("n_layers", 4)
            if hidden >= 64 or layers >= 8:
                raise ValueError(
                    f"Configuration (hidden={hidden}, layers={layers}) is too large for 2D "
                    f"benchmark {benchmark!r} and will crash (OOM). Use hidden<64, layers<8."
                )

        if name not in self._registry and name in self._lazy_imports:
            mod_name, cls_name = self._lazy_imports[name]
            import importlib
            module = importlib.import_module(f"models.{mod_name}")
            cls = getattr(module, cls_name)
            self.register_class(name, cls)
            
        if name not in self._registry:
            available = ", ".join(sorted(set(self._registry) | set(self._lazy_imports)))
            raise ValueError(
                f"Unknown model {name!r}. Available: {available}"
            )

        # Map n_modes -> (n_modes1, n_modes2) for 2D models
        if name.endswith("2D") and "n_modes" in kwargs:
            m = kwargs.pop("n_modes")
            kwargs["n_modes1"] = m
            kwargs["n_modes2"] = m

        return self._registry[name](**kwargs)

    def register_class(self, name: str, cls: Type[nn.Module], **fixed_kwargs) -> None:
        def _factory(**kwargs):
            kwargs.update(fixed_kwargs)
            # Filter kwargs to only those accepted by cls.__init__
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

    def __contains__(self, name: str) -> bool:
        return name in self._registry or name in self._lazy_imports

MODEL_REGISTRY = ModelRegistry()

# ── Benchmark Registry ────────────────────────────────────────────────────────

class BenchmarkRegistry:
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
    from data.prepare import GRID_SIZE, make_dataloader, evaluate_l2_rel
    from data.benchmarks_ext import (
        EXT_BENCHMARKS, EXT_SOTA, make_ext_dataloader, evaluate_l2_rel_ext
    )

    # ── 1-D models (Lazy registration) ────────────────────────────────────────
    MODEL_REGISTRY.register_lazy("FNO", "fno", "FNO1d")
    MODEL_REGISTRY.register_lazy("RFNO", "fno", "RFNO1d")
    MODEL_REGISTRY.register_lazy("AFNO", "afno", "AFNO1d")
    MODEL_REGISTRY.register_lazy("FFNO", "afno", "FFNO1d")
    MODEL_REGISTRY.register_lazy("UNO", "fno", "UNO1d")
    MODEL_REGISTRY.register_lazy("WNO", "wno", "WNO1d")
    MODEL_REGISTRY.register_lazy("WNO_GNOT", "wno", "WNO_GNOT")
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
    MODEL_REGISTRY.register_lazy("GNOT_Axial2d", "gnot", "GNOT_Axial2d")
    MODEL_REGISTRY.register_lazy("GNOT_FFNO", "gnot", "GNOT_FFNO")
    MODEL_REGISTRY.register_lazy("MambaNO", "mamba_no", "MambaNO1d")
    MODEL_REGISTRY.register_lazy("PINO", "pinn", "PINO1d")
    MODEL_REGISTRY.register_lazy("FNO2D", "fno", "FNO2d")
    MODEL_REGISTRY.register_lazy("RFNO2D", "fno", "RFNO2d")
    MODEL_REGISTRY.register_lazy("FNO_MC", "fno", "FNO1dMC")
    MODEL_REGISTRY.register_lazy("TFNO", "tfno", "TFNO1d")
    MODEL_REGISTRY.register_lazy("RTFNO", "tfno", "RTFNO1d")
    MODEL_REGISTRY.register_lazy("CPFNO", "tfno", "CPFNO1d")
    MODEL_REGISTRY.register_lazy("TFNO2D", "tfno", "TFNO2d")
    MODEL_REGISTRY.register_lazy("Transolver", "transolver", "Transolver1d")
    MODEL_REGISTRY.register_lazy("Transolver2D", "transolver", "Transolver2d")
    MODEL_REGISTRY.register_lazy("TimeDeepONet", "time_deeponet", "TimeDeepONet1d")
    MODEL_REGISTRY.register_lazy("DualDeepONet", "time_deeponet", "DualBranchDeepONet1d")
    MODEL_REGISTRY.register_lazy("HNN", "hnn", "HamiltonianNO1d")
    MODEL_REGISTRY.register_lazy("EnergyFNO", "hnn", "EnergyConservingFNO1d")
    MODEL_REGISTRY.register_lazy("NeuralODE", "neural_ode", "NeuralODE1d")
    MODEL_REGISTRY.register_lazy("UDE", "neural_ode", "UniversalDE1d")
    MODEL_REGISTRY.register_lazy("LatentODE", "neural_ode", "LatentODE1d")
    MODEL_REGISTRY.register_lazy("PACMANN", "pacmann", "PACMANN")
    
    # ── 2024-2025 SOTA Models ────────────────────────────────────────────────
    MODEL_REGISTRY.register_lazy("MambaNO", "mamba_no", "MambaNO1d")
    MODEL_REGISTRY.register_lazy("MambaNO1d", "mamba_no", "MambaNO1d")   # alias: experiments.yaml uses both
    MODEL_REGISTRY.register_lazy("MemNO", "mem_no", "MemNO1d")
    MODEL_REGISTRY.register_lazy("ModifiedKAN_FNO", "kan", "ModifiedKAN_FNO")  # missing registration
    MODEL_REGISTRY.register_lazy("GNOT2D", "gnot", "GNOT2d")              # alias: GNOT2D → GNOT2d
    
    # ── Novel Hybrid Models ───────────────────────────────────────────────────
    MODEL_REGISTRY.register_lazy("HybridDecoderDeepONet2D", "hybrid_decoder_deeponet", "HybridDecoderDeepONet2d")
    MODEL_REGISTRY.register_lazy("AttentionEnhancedFNO2D", "attention_fno", "AttentionEnhancedFNO2d")
    MODEL_REGISTRY.register_lazy("HybridFNODeepONet2D", "hybrid_fno_deeponet", "HybridFNODeepONet2d")
    MODEL_REGISTRY.register_lazy("FEDONet2D", "fedonet", "FEDONet2d")
    MODEL_REGISTRY.register_lazy("HANO2D", "hano", "HANO2d")
    MODEL_REGISTRY.register_lazy("SNO2D", "sno", "SNO2d")
    MODEL_REGISTRY.register_lazy("VSMNO2D", "vsmno", "VSMNO2d")

    @MODEL_REGISTRY.register("FNO")
    def _make_fno(n_modes=16, hidden_dim=64, n_layers=4, **kw):
        from models.fno import FNO1d
        return FNO1d(n_modes=n_modes, hidden_dim=hidden_dim, n_layers=n_layers)

    @MODEL_REGISTRY.register("RFNO")
    def _make_rfno(n_modes=16, hidden_dim=64, n_layers=4, **kw):
        from models.fno import RFNO1d
        return RFNO1d(n_modes=n_modes, hidden_dim=hidden_dim, n_layers=n_layers)

    # ── Benchmarks (standard + ext) ───────────────────────────────────────────
    _std = {"burgers_1d"}
    for bm in _std:
        BENCHMARK_REGISTRY.register(
            bm,
            make_loader=make_dataloader,
            evaluate=lambda name, fn: evaluate_l2_rel(name, fn),
            sota={"burgers_1d": 0.0149}.get(bm),
            description={"burgers_1d": "1D viscous Burgers"}.get(bm, ""),
        )
    for bm in EXT_BENCHMARKS:
        BENCHMARK_REGISTRY.register(
            bm,
            make_loader=make_ext_dataloader,
            evaluate=lambda name, fn: evaluate_l2_rel_ext(name, fn),
            sota=EXT_SOTA.get(bm),
            description={
                "kdv_1d":       "KdV soliton (ETDRK4)",
                "wave_1d":      "1D wave u_tt=c²u_xx",
                "darcy_2d":     "2D Darcy -∇·(a∇u)=f (corrected solver)",
                "ns_2d":        "2D NS vorticity (CFL-stable ICs)",
                "ns_hre_2d":    "2D NS Re=1000 turbulence challenge",
                "swe_2d":       "2D Shallow Water Equations",
                "allen_cahn_2d": "2D Allen-Cahn phase separation",
                "mhd_2d":        "2D Magnetohydrodynamics",
            }.get(bm, ""),
        )

    # ── High-fidelity simulation benchmarks ───────────────────────────────────
    from data.simulations import (
        SIM_BENCHMARKS, SIM_SOTA, SIM_METADATA,
        make_sim_dataloader, evaluate_l2_rel_sim,
    )
    for bm in SIM_BENCHMARKS:
        meta = SIM_METADATA[bm]
        BENCHMARK_REGISTRY.register(
            bm,
            make_loader=make_sim_dataloader,
            evaluate=lambda name, fn: evaluate_l2_rel_sim(name, fn),
            sota=SIM_SOTA.get(bm),
            description=meta.get("pde", ""),
        )

_register_defaults()
