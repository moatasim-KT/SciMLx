"""Visualization script for SciML models.

Loads a random (untrained) model, runs it on validation samples, and plots
prediction vs ground-truth.  To visualize a trained model you would save
and reload weights — this script focuses on architecture sanity-checking.

Usage:
    uv run vis.py                                     # FNO on burgers_1d
    uv run vis.py --benchmark burgers_1d --model UNO
    uv run vis.py --benchmark burgers_1d --model WNO --levels 3
    uv run vis.py --benchmark darcy_2d   --model FNO
"""

import argparse
import os

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt

from prepare import make_dataloader, GRID_SIZE
from models import FNO1d, FNO2d, UNO1d, WNO1d, DeepONet, PODDeepONet


def build_model(benchmark: str, model_type: str,
                modes: int, levels: int, hidden: int, layers: int):
    is_1d = benchmark.endswith("_1d")
    if model_type == "FNO":
        return (FNO1d(n_modes=modes, hidden_dim=hidden, n_layers=layers)
                if is_1d else
                FNO2d(n_modes1=modes, n_modes2=modes, hidden_dim=hidden, n_layers=layers))
    if model_type == "UNO":
        if not is_1d:
            raise ValueError("UNO 2D not yet implemented.")
        return UNO1d(n_modes=modes, hidden_dim=hidden, n_layers=layers)
    if model_type == "WNO":
        if not is_1d:
            raise ValueError("WNO 2D not yet implemented.")
        return WNO1d(n_levels=levels, hidden_dim=hidden, n_layers=layers)
    if model_type == "DeepONet":
        branch = GRID_SIZE if is_1d else GRID_SIZE * GRID_SIZE
        trunk  = 1         if is_1d else 2
        return DeepONet(branch_dim=branch, trunk_dim=trunk,
                        hidden_dim=hidden, out_dim=hidden, n_layers=layers)
    if model_type == "PODDeepONet":
        if not is_1d:
            raise ValueError("PODDeepONet 2D not yet implemented.")
        return PODDeepONet(branch_dim=GRID_SIZE, n_basis=hidden,
                           hidden_dim=hidden, n_layers=layers)
    raise ValueError(f"Unknown model type: {model_type!r}")


def forward(model, model_type: str, benchmark: str, x: mx.array) -> mx.array:
    is_1d = benchmark.endswith("_1d")
    if model_type == "DeepONet":
        B = x.shape[0]
        if is_1d:
            coords = mx.linspace(0, 1, GRID_SIZE).reshape(1, GRID_SIZE, 1)
            coords = mx.broadcast_to(coords, (B, GRID_SIZE, 1))
            return model(x, coords)
        g1 = mx.broadcast_to(mx.linspace(0, 1, GRID_SIZE).reshape(1, GRID_SIZE, 1),
                              (1, GRID_SIZE, GRID_SIZE))
        g2 = mx.broadcast_to(mx.linspace(0, 1, GRID_SIZE).reshape(1, 1, GRID_SIZE),
                              (1, GRID_SIZE, GRID_SIZE))
        coords = mx.stack([g1, g2], axis=-1)
        coords = mx.broadcast_to(coords, (B, GRID_SIZE, GRID_SIZE, 2)).reshape(B, -1, 2)
        return model(x.reshape(B, -1), coords).reshape(B, GRID_SIZE, GRID_SIZE)
    if model_type == "PODDeepONet":
        B = x.shape[0]
        return model(x if is_1d else x.reshape(B, -1))
    return model(x)


def visualize(benchmark: str, model_type: str,
              modes: int, levels: int, hidden: int, layers: int,
              num_samples: int = 3) -> None:
    model = build_model(benchmark, model_type, modes, levels, hidden, layers)
    mx.eval(model.parameters())

    loader    = make_dataloader(benchmark, "val", num_samples)
    x, y      = next(loader)
    pred      = forward(model, model_type, benchmark, x)
    mx.eval(pred)

    x_np   = np.array(x)
    y_np   = np.array(y)
    pred_np = np.array(pred)

    is_1d = benchmark.endswith("_1d")

    if is_1d:
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples))
        if num_samples == 1:
            axes = [axes]
        grid = np.linspace(0, 1, GRID_SIZE)
        for i in range(num_samples):
            ax = axes[i]
            ax.plot(grid, x_np[i],    "k--", alpha=0.7, label="Input u₀")
            ax.plot(grid, y_np[i],    "b-",  linewidth=2, label="True uT")
            ax.plot(grid, pred_np[i], "r--", linewidth=2, label="Pred uT")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"Sample {i+1}  (untrained {model_type})")
            ax.set_xlabel("x")
    else:
        fig, axes = plt.subplots(num_samples, 3,
                                 figsize=(15, 4 * num_samples))
        if num_samples == 1:
            axes = axes[None, :]
        for i in range(num_samples):
            vmax = float(max(y_np[i].max(), pred_np[i].max()))
            vmin = float(min(y_np[i].min(), pred_np[i].min()))
            kw = dict(cmap="RdBu_r", vmin=vmin, vmax=vmax)
            im0 = axes[i, 0].imshow(x_np[i],    cmap="viridis")
            im1 = axes[i, 1].imshow(y_np[i],    **kw)
            im2 = axes[i, 2].imshow(pred_np[i], **kw)
            axes[i, 0].set_title(f"Input {i+1}")
            axes[i, 1].set_title(f"True {i+1}")
            axes[i, 2].set_title(f"Pred {i+1} ({model_type})")
            for ax, im in zip(axes[i], [im0, im1, im2]):
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = f"vis_{benchmark}_{model_type}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SciML model visualizer")
    parser.add_argument("--benchmark", default="burgers_1d")
    parser.add_argument("--model",     default="FNO",
                        choices=["FNO", "UNO", "WNO", "DeepONet", "PODDeepONet"])
    parser.add_argument("--modes",    type=int, default=16)
    parser.add_argument("--levels",   type=int, default=3)
    parser.add_argument("--hidden",   type=int, default=64)
    parser.add_argument("--layers",   type=int, default=4)
    parser.add_argument("--samples",  type=int, default=3)
    args = parser.parse_args()

    visualize(args.benchmark, args.model, args.modes, args.levels,
              args.hidden, args.layers, args.samples)
