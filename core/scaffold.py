"""Gated Model Code Generation for SciML AutoResearch - PyTorch/CUDA Optimized."""

import argparse
import ast
import importlib.util
import sys
from pathlib import Path
from typing import Optional

from core.utils import REPO_ROOT
from core.device import DEVICE, FRAMEWORK

# ── Stub templates ────────────────────────────────────────────────────────────

_STUB_TEMPLATE = '''"""
{name} — SciML Neural Operator (PyTorch)

Auto-generated stub by model_scaffold.py.
Base architecture: {base}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fno import SpectralConv1d

class {name}(nn.Module):
    """
    {name}: extend description here.
    """

    def __init__(self, n_modes: int = 16, hidden_dim: int = 64,
                 n_layers: int = 4, **kwargs):
        super().__init__()
        self.n_modes    = n_modes
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.lift = nn.Linear(1, hidden_dim)

        # TODO: replace with your custom operator blocks
        self.blocks = nn.ModuleList([
            SpectralConv1d(hidden_dim, hidden_dim, n_modes)
            for _ in range(n_layers)
        ])
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N] input field
        Returns:
            out: [B, N] output field
        """
        # Lift to hidden dim
        h = self.lift(x.unsqueeze(-1))          # [B, N, hidden_dim]

        # Apply operator blocks
        for block in self.blocks:
            h = h + block(h)                 # residual

        # Project back to scalar field
        out = self.proj(h).squeeze(-1)       # [B, N]
        return out
'''

_STUB_TEMPLATE_2D = '''"""
{name}2d — 2D SciML Neural Operator (PyTorch auto-generated stub)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fno import SpectralConv2d

class {name}2d(nn.Module):
    def __init__(self, n_modes: int = 12, hidden_dim: int = 32,
                 n_layers: int = 4, **kwargs):
        super().__init__()
        self.n_modes    = n_modes
        self.hidden_dim = hidden_dim
        self.lift       = nn.Linear(1, hidden_dim)
        self.blocks     = nn.ModuleList([
            SpectralConv2d(hidden_dim, hidden_dim, n_modes, n_modes)
            for _ in range(n_layers)
        ])
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, N]  →  out: [B, N, N]"""
        B, N, _ = x.shape
        h = self.lift(x.view(B, N*N, 1))          # [B, N*N, H]
        for block in self.blocks:
            h = h + block(h.view(B, N, N, -1)).view(B, N*N, -1)
        return self.proj(h).squeeze(-1).view(B, N, N)
'''

_MLX_STUB_TEMPLATE = '''"""
{name} — SciML Neural Operator (MLX)

Auto-generated stub by model_scaffold.py.
Base architecture: {base}
"""

import mlx.core as mx
import mlx.nn as nn

# from models.layers.mlx_spectral import SpectralConv1d

class {name}(nn.Module):
    def __init__(self, n_modes: int = 16, hidden_dim: int = 64,
                 n_layers: int = 4, **kwargs):
        super().__init__()
        self.n_modes    = n_modes
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.lift = nn.Linear(1, hidden_dim)
        self.blocks = [
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ]
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.lift(x[..., None])          # [B, N, H]
        for block in self.blocks:
            h = h + block(h)
        return self.proj(h).squeeze(-1)      # [B, N]
'''

_MLX_STUB_TEMPLATE_2D = '''"""
{name}2d — 2D SciML Neural Operator (MLX)
"""

import mlx.core as mx
import mlx.nn as nn

class {name}2d(nn.Module):
    def __init__(self, n_modes: int = 12, hidden_dim: int = 32,
                 n_layers: int = 4, **kwargs):
        super().__init__()
        self.n_modes    = n_modes
        self.hidden_dim = hidden_dim
        self.lift       = nn.Linear(1, hidden_dim)
        self.blocks     = [
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ]
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, N, _ = x.shape
        h = self.lift(x.reshape(B, N*N, 1))
        for block in self.blocks:
            h = h + block(h)
        return self.proj(h).squeeze(-1).reshape(B, N, N)
'''

def generate_stub(name: str, base: str = "FNO",
                  notes: str = "", two_d: bool = False,
                  framework: Optional[str] = None) -> str:
    """Return Python source for a new model stub."""
    fw = framework or FRAMEWORK
    module = name.lower()

    if fw == "mlx":
        template = _MLX_STUB_TEMPLATE_2D if two_d else _MLX_STUB_TEMPLATE
    else:
        template = _STUB_TEMPLATE_2D if two_d else _STUB_TEMPLATE

    return template.format(name=name, base=base,
                           notes=notes or "fill in architecture details",
                           module=module)

# ── Validation gate ───────────────────────────────────────────────────────────

class ModelGate:
    def validate(self, name: str, path: str) -> tuple[bool, dict]:
        report: dict = {}
        model_path = Path(path)

        # ── Gate 1: Syntax ────────────────────────────────────────────────────
        try:
            source = model_path.read_text()
            ast.parse(source)
            report["syntax_ok"] = True
        except SyntaxError as e:
            report.update(syntax_ok=False, import_ok=False, smoke_ok=False, error=f"SyntaxError: {e}")
            return False, report

        # ── Gate 2: Import ────────────────────────────────────────────────────
        try:
            spec = importlib.util.spec_from_file_location(name, model_path)
            mod  = importlib.util.module_from_spec(spec)
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            spec.loader.exec_module(mod)
            cls = getattr(mod, name, None)
            if cls is None:
                raise ImportError(f"Class {name!r} not found in {model_path}")
            report["import_ok"] = True
        except Exception as e:
            report.update(import_ok=False, smoke_ok=False, error=f"ImportError: {e}")
            return False, report

        # ── Gate 3: Smoke test ────────────────────────────────────────────────
        try:
            out_shape = None
            if FRAMEWORK == "mlx":
                import mlx.core as mx
                model_inst = cls(n_modes=8, hidden_dim=16, n_layers=2)
                x1d = mx.random.normal((2, 64))
                x2d = mx.random.normal((2, 16, 16))
                for x in (x1d, x2d):
                    try:
                        out = model_inst(x)
                        mx.eval(out)
                        if list(out.shape) == list(x.shape):
                            out_shape = out.shape
                            break
                    except Exception:
                        continue
            else:
                import torch
                model_inst = cls(n_modes=8, hidden_dim=16, n_layers=2).to(DEVICE)
                x1d = torch.randn(2, 64).to(DEVICE)
                x2d = torch.randn(2, 16, 16).to(DEVICE)
                for x in (x1d, x2d):
                    try:
                        out = model_inst(x)
                        if out.shape == x.shape:
                            out_shape = out.shape
                            break
                    except Exception:
                        continue

            if out_shape is None:
                raise ValueError("Model output shape does not match input shape")
            report.update(smoke_ok=True, output_shape=str(out_shape))
        except Exception as e:
            report.update(smoke_ok=False, error=f"SmokeError: {e}")
            return False, report

        return True, report

    def register_and_queue(self, name: str, path: str,
                           benchmarks: Optional[list] = None,
                           priority: int = 3) -> bool:
        model_path = Path(path)
        target = REPO_ROOT / "models" / model_path.name
        if not target.exists():
            target.write_text(model_path.read_text())

        # Update research_plugins.py
        plugins_path = REPO_ROOT / "core" / "research_plugins.py"
        plugins_src  = plugins_path.read_text()
        reg_line = f'    MODEL_REGISTRY.register_lazy("{name}", "{model_path.stem}", "{name}")\n'
        if f'"{name}"' not in plugins_src:
            marker = "# ── Model Registrations (Lazy) ───────────────────────────────────────────"
            plugins_src = plugins_src.replace(marker, f"{marker}\n{reg_line}")
            plugins_path.write_text(plugins_src)
            print(f"  Registered {name} in MODEL_REGISTRY")

        # Append to experiments.yaml
        bms = benchmarks or ["burgers_1d"]
        yaml_path = REPO_ROOT / "experiments.yaml"
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or []
        
        new_configs = []
        for bm in bms:
            new_configs.append({
                "name": f"{name.lower()}_{bm[:5]}_baseline",
                "benchmark": bm,
                "model": name,
                "hidden_dim": 64,
                "n_layers": 4,
                "n_modes": 16,
                "budget_s": 600,
                "priority": priority,
            })
        data.extend(new_configs)
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        return True

def main() -> None:
    p = argparse.ArgumentParser(description="Gated model code generation")
    p.add_argument("--stub", help="Generate a model stub file")
    p.add_argument("--base", default="FNO")
    p.add_argument("--2d", dest="two_d", action="store_true")
    p.add_argument("--validate", nargs=2, metavar=("NAME", "PATH"))
    p.add_argument("--register", nargs=2, metavar=("NAME", "PATH"))
    args = p.parse_args()

    if args.stub:
        code = generate_stub(args.stub, args.base, two_d=args.two_d)
        out = REPO_ROOT / "models" / f"{args.stub.lower()}.py"
        out.write_text(code)
        print(f"Stub written to {out}")
    elif args.validate:
        name, path = args.validate
        ok, report = ModelGate().validate(name, path)
        print(f"Validation {'PASSED' if ok else 'FAILED'}: {report}")
    elif args.register:
        name, path = args.register
        gate = ModelGate()
        ok, report = gate.validate(name, path)
        if ok:
            gate.register_and_queue(name, path)
            print(f"Successfully registered {name}")

if __name__ == "__main__":
    main()
