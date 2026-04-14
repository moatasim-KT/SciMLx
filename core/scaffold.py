"""Gated Model Code Generation for SciML AutoResearch.

Wave 3 · Item 25: Allows agent_loop.py (or an external agent) to generate new
model architecture stubs that are automatically validated before entering the
experiment queue.

Gate contract
─────────────
A new model must pass three checks before being registered or queued:
  1. Syntax  — the generated .py file parses without error
  2. Import  — the class can be imported from its module
  3. Smoke   — model(random_input) returns the correct output shape

Only after all three pass does the model get:
  • registered in MODEL_REGISTRY (research_plugins.py)
  • added to models/__init__.py exports
  • given a priority-3 ExperimentConfig in experiments.yaml

Usage (external agent or agent_loop.py):
    from model_scaffold import ModelGate, generate_stub

    # Generate a starter stub
    code = generate_stub("MySuperFNO", base="FNO", notes="Add attention after each block")
    Path("models/my_super_fno.py").write_text(code)
    # ... edit the stub ...

    # Validate and register
    gate = ModelGate()
    ok, report = gate.validate("MySuperFNO", "models/my_super_fno.py")
    if ok:
        gate.register_and_queue("MySuperFNO", "models/my_super_fno.py",
                                benchmarks=["burgers_1d", "kdv_1d"])

CLI:
    uv run -m core.scaffold --stub MySuperFNO --base FNO
    uv run -m core.scaffold --validate MySuperFNO models/my_super_fno.py
    uv run model_scaffold.py --list          # show registered models
"""

import argparse
import ast
import importlib.util
import sys
import textwrap
from pathlib import Path
from typing import Optional

from core.utils import REPO_ROOT

# ── Stub templates ────────────────────────────────────────────────────────────

_STUB_TEMPLATE = '''"""
{name} — SciML Neural Operator

Auto-generated stub by model_scaffold.py.
Base architecture: {base}
Notes: {notes}

Edit this file to implement the model, then validate with:
    uv run model_scaffold.py --validate {name} models/{module}.py
"""

import mlx.core as mx
import mlx.nn as nn
from .fno import SpectralConv1d   # reuse existing building blocks


class {name}(nn.Module):
    """
    {name}: extend description here.

    Args:
        n_modes   : number of Fourier modes to keep
        hidden_dim: channel width
        n_layers  : number of operator blocks
    """

    def __init__(self, n_modes: int = 16, hidden_dim: int = 64,
                 n_layers: int = 4, **kwargs):
        super().__init__()
        self.n_modes    = n_modes
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.lift = nn.Linear(1, hidden_dim)

        # TODO: replace with your custom operator blocks
        self.blocks = [
            SpectralConv1d(hidden_dim, hidden_dim, n_modes)
            for _ in range(n_layers)
        ]
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, N] input field
        Returns:
            out: [B, N] output field
        """
        # Lift to hidden dim
        h = self.lift(x[..., None])          # [B, N, hidden_dim]

        # Apply operator blocks
        for block in self.blocks:
            h = h + block(h)                 # residual

        # Project back to scalar field
        out = self.proj(h).squeeze(-1)       # [B, N]
        return out
'''

_STUB_TEMPLATE_2D = '''"""
{name}2d — 2D SciML Neural Operator (auto-generated stub)
"""

import mlx.core as mx
import mlx.nn as nn
from .fno import SpectralConv2d


class {name}2d(nn.Module):
    def __init__(self, n_modes: int = 12, hidden_dim: int = 32,
                 n_layers: int = 4, **kwargs):
        super().__init__()
        self.n_modes    = n_modes
        self.hidden_dim = hidden_dim
        self.lift       = nn.Linear(1, hidden_dim)
        self.blocks     = [
            SpectralConv2d(hidden_dim, hidden_dim, n_modes, n_modes)
            for _ in range(n_layers)
        ]
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def __call__(self, x: mx.array) -> mx.array:
        """x: [B, N, N]  →  out: [B, N, N]"""
        B, N, _ = x.shape
        h = self.lift(x.reshape(B, N*N, 1))          # [B, N*N, H]
        for block in self.blocks:
            h = h + block(h.reshape(B, N, N, -1)).reshape(B, N*N, -1)
        return self.proj(h).squeeze(-1).reshape(B, N, N)
'''


def generate_stub(name: str, base: str = "FNO",
                  notes: str = "", two_d: bool = False) -> str:
    """Return Python source for a new model stub."""
    module = name.lower()
    template = _STUB_TEMPLATE_2D if two_d else _STUB_TEMPLATE
    return template.format(name=name, base=base,
                           notes=notes or "fill in architecture details",
                           module=module)


# ── Validation gate ───────────────────────────────────────────────────────────

class ModelGate:
    """Three-stage validation gate for new model code."""

    def validate(self, name: str, path: str) -> tuple[bool, dict]:
        """
        Run all three gate checks. Returns (passed: bool, report: dict).

        report keys:
            syntax_ok, import_ok, smoke_ok,
            error (if any check failed),
            output_shape (if smoke passed)
        """
        report: dict = {}
        model_path = Path(path)

        # ── Gate 1: Syntax ────────────────────────────────────────────────────
        try:
            source = model_path.read_text()
            ast.parse(source)
            report["syntax_ok"] = True
        except SyntaxError as e:
            report.update(syntax_ok=False, import_ok=False,
                          smoke_ok=False, error=f"SyntaxError: {e}")
            return False, report

        # ── Gate 2: Import ────────────────────────────────────────────────────
        try:
            spec = importlib.util.spec_from_file_location(name, model_path)
            mod  = importlib.util.module_from_spec(spec)
            # Add repo root to path so relative imports resolve
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            spec.loader.exec_module(mod)
            cls = getattr(mod, name, None)
            if cls is None:
                raise ImportError(f"Class {name!r} not found in {model_path}")
            report["import_ok"] = True
        except Exception as e:
            report.update(import_ok=False, smoke_ok=False,
                          error=f"ImportError: {e}")
            return False, report

        # ── Gate 3: Smoke test ────────────────────────────────────────────────
        try:
            import mlx.core as mx
            import numpy as np
            model_inst = cls(n_modes=8, hidden_dim=16, n_layers=2)
            # Test both 1D [B,N] and 2D [B,N,N] shapes
            x1d = mx.array(np.random.randn(4, 64).astype(np.float32))
            x2d = mx.array(np.random.randn(4, 16, 16).astype(np.float32))
            out_shape = None
            for x in (x1d, x2d):
                try:
                    out = model_inst(x)
                    mx.eval(out)
                    if out.shape == x.shape:
                        out_shape = out.shape
                        break
                except Exception:
                    continue
            if out_shape is None:
                raise ValueError("Model output shape does not match input shape "
                                 "for either 1D [B,N] or 2D [B,N,N] input")
            report.update(smoke_ok=True, output_shape=str(out_shape))
        except Exception as e:
            report.update(smoke_ok=False, error=f"SmokeError: {e}")
            return False, report

        return True, report

    def register_and_queue(self, name: str, path: str,
                           benchmarks: Optional[list] = None,
                           priority: int = 3) -> bool:
        """
        After validation passes:
          1. Copy model file to models/ if not already there
          2. Add export to models/__init__.py
          3. Add MODEL_REGISTRY.register_class() call to research_plugins.py
          4. Append ExperimentConfig entries to experiments.yaml
        """
        model_path = Path(path)
        target     = REPO_ROOT / "models" / model_path.name

        # 1. Copy to models/ if needed
        if not target.exists():
            target.write_text(model_path.read_text())
            print(f"  Copied {model_path.name} → models/")

        # 2. Add to models/__init__.py
        init_path = REPO_ROOT / "models" / "__init__.py"
        init_src  = init_path.read_text()
        module    = target.stem
        if f"from .{module}" not in init_src:
            import_line = f"from .{module}  import {name}\n"
            export_line = f'    "{name}",\n'
            # Insert import before __all__
            init_src = init_src.replace(
                "\n__all__", f"\n{import_line}__all__")
            # Insert into __all__ list
            init_src = init_src.replace(
                ']\n', f'{export_line}]\n', 1)
            init_path.write_text(init_src)
            print(f"  Added {name} export to models/__init__.py")

        # 3. Add to research_plugins.py MODEL_REGISTRY
        plugins_path = REPO_ROOT / "core" / "research_plugins.py"
        plugins_src  = plugins_path.read_text()
        reg_line     = f'    MODEL_REGISTRY.register_class("{name}", {name})\n'
        if reg_line.strip() not in plugins_src:
            # Find the end of _register_defaults and insert before closing line
            marker = "BENCHMARK_REGISTRY = BenchmarkRegistry()"
            if marker in plugins_src:
                # Add import at top of _register_defaults
                import_stmt = f"    from models.{module} import {name}\n"
                plugins_src = plugins_src.replace(
                    marker,
                    f"    # {name}\n{import_stmt}{reg_line}\n{marker}",
                )
                plugins_path.write_text(plugins_src)
                print(f"  Registered {name} in MODEL_REGISTRY")

        # 4. Append ExperimentConfigs to experiments.yaml
        bms = benchmarks or ["burgers_1d"]
        yaml_path = REPO_ROOT / "experiments.yaml"
        
        import yaml
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or []
        else:
            data = []

        existing_names = {exp["name"] for exp in data if "name" in exp}
        
        new_configs = []
        for bm in bms:
            exp_name = f"{name.lower()}_{bm[:5]}_baseline"
            if exp_name in existing_names:
                continue
            is_2d = "2d" in bm
            new_configs.append({
                "name": exp_name,
                "benchmark": bm,
                "model": name,
                "hidden_dim": 64,
                "n_layers": 4,
                "n_modes": 16,
                "budget_s": 480 if is_2d else 300,
                "priority": priority,
                "rationale": f"Auto-generated baseline for {name} on {bm}",
            })
        
        if new_configs:
            data.extend(new_configs)
            with open(yaml_path, "w") as f:
                yaml.dump(data, f, sort_keys=False)
            print(f"  Appended {len(new_configs)} ExperimentConfig(s) to experiments.yaml")

        return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Gated model code generation and registration")
    p.add_argument("--stub",     metavar="NAME",
                   help="Generate a model stub file for NAME")
    p.add_argument("--base",     default="FNO",
                   help="Base architecture to inherit from (default: FNO)")
    p.add_argument("--notes",    default="",
                   help="Free-text notes added to the stub docstring")
    p.add_argument("--2d",       dest="two_d", action="store_true",
                   help="Generate a 2D model stub")
    p.add_argument("--validate", nargs=2, metavar=("NAME", "PATH"),
                   help="Validate NAME from file PATH through all three gates")
    p.add_argument("--register", nargs=2, metavar=("NAME", "PATH"),
                   help="Validate then register NAME from PATH + queue experiments")
    p.add_argument("--benchmarks", nargs="+", default=["burgers_1d"],
                   help="Benchmarks for the new model (used with --register)")
    p.add_argument("--list",     action="store_true",
                   help="List all registered models")
    args = p.parse_args()

    if args.list:
        from core.research_plugins import MODEL_REGISTRY
        print("Registered models:", MODEL_REGISTRY.available)
        return

    if args.stub:
        code = generate_stub(args.stub, args.base, args.notes, args.two_d)
        out  = REPO_ROOT / "models" / f"{args.stub.lower()}.py"
        out.write_text(code)
        print(f"Stub written to {out}")
        print(f"Edit it, then run:")
        print(f"  uv run model_scaffold.py --register {args.stub} {out}")
        return

    if args.validate:
        name, path = args.validate
        gate = ModelGate()
        ok, report = gate.validate(name, path)
        status = "PASSED" if ok else "FAILED"
        print(f"Gate {status} for {name}:")
        for k, v in report.items():
            print(f"  {k}: {v}")
        return

    if args.register:
        name, path = args.register
        gate = ModelGate()
        ok, report = gate.validate(name, path)
        if not ok:
            print(f"Validation failed — aborting registration:")
            print(f"  {report.get('error')}")
            sys.exit(1)
        print(f"All gates passed. Registering {name}...")
        gate.register_and_queue(name, path, benchmarks=args.benchmarks)
        print(f"Done. Run `uv run autorun.py --model {name}` to execute.")
        return

    p.print_help()


if __name__ == "__main__":
    main()
