#!/usr/bin/env python3
"""Comprehensive visualization module for SciML autoresearch results.

Generates plots for:
  leaderboard   - results.tsv bar charts + model comparison per benchmark
  training      - loss / grad-norm curves from .log files
  data          - sample input→output pairs for each benchmark
  validation    - solver accuracy vs analytical solutions (wave, KdV soliton)
  spectral      - Fourier spectrum of inputs/outputs (motivates mode count)
  concerns      - automated concern report (flags anomalies + regressions)
  all           - all of the above (default)

Usage:
    uv run viz.py                                 # all modes, save to figs/
    uv run viz.py --mode leaderboard              # results comparison only
    uv run viz.py --mode training                 # all logs
    uv run viz.py --mode training --log <path>    # single log
    uv run viz.py --mode data --benchmark kdv_1d  # data samples
    uv run viz.py --mode validation               # solver accuracy
    uv run viz.py --mode spectral                 # frequency analysis
    uv run viz.py --mode concerns                 # concern dashboard
    uv run viz.py --mode arch --benchmark burgers_1d --model FNO  # arch sanity check
    uv run viz.py --show                          # display (don't save)
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── Matplotlib setup ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")                   # overridden below if --show
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import LogLocator, NullFormatter

from utils import REPO_ROOT as REPO, FIGS_DIR as FIGS, LOGS_DIR as LOGS, RESULTS_FILE as RESULTS, SOTA, load_results as _load_rows

BM_COLORS = {
    "burgers_1d": "#e74c3c",
    "kdv_1d":     "#3498db",
    "wave_1d":    "#2ecc71",
    "darcy_2d":   "#9b59b6",
}
MODEL_COLORS = {
    "FNO": "#2980b9", "RFNO": "#27ae60", "AFNO": "#e67e22",
    "FFNO": "#8e44ad", "UNO": "#16a085", "WNO": "#c0392b",
    "DeepONet": "#d35400", "PODDeepONet": "#7f8c8d",
}
STATUS_COLORS = {"keep": "#27ae60", "discard": "#e74c3c", "crash": "#95a5a6"}

SHOW = False


# ── I/O ───────────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    FIGS.mkdir(exist_ok=True)
    path = FIGS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if SHOW:
        plt.show()
    plt.close(fig)


def load_results() -> list[dict]:
    rows = _load_rows()
    for row in rows:
        row["val"] = row["val_l2_rel"]  # alias expected by viz internals
    return rows


def parse_log(path: Path) -> dict:
    """Extract step timeseries, final result, and metadata from a .log file."""
    steps, losses, lrs, dts = [], [], [], []
    val   = None
    meta  = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.rstrip()
                # Progress line (ends with spaces, starts with \r in file but
                # newlines are normalised above)
                m = re.search(
                    r"step\s+(\d+)\s+\([\d.]+%\)\s+\|\s+loss:\s+([\d.]+)\s+\|\s+"
                    r"lr:\s+([\d.e+-]+)\s+\|\s+dt:\s+([\d.]+)ms",
                    line,
                )
                if m:
                    steps.append(int(m.group(1)))
                    losses.append(float(m.group(2)))
                    lrs.append(float(m.group(3)))
                    dts.append(float(m.group(4)))
                elif line.startswith("val_l2_rel:"):
                    val = float(line.split(":")[1].strip())
                elif line.startswith("Model"):
                    meta["model_line"] = line
                elif line.startswith("Benchmark:"):
                    meta["benchmark"] = line.split(":")[-1].strip()
                elif line.startswith("num_steps:"):
                    meta["num_steps"] = int(line.split(":")[-1].strip())
                elif line.startswith("peak_vram_mb:"):
                    meta["peak_vram_mb"] = float(line.split(":")[-1].strip())
    except Exception:
        pass
    return {
        "steps": np.array(steps),
        "losses": np.array(losses),
        "lrs": np.array(lrs),
        "dts": np.array(dts),
        "val": val,
        "meta": meta,
        "name": path.stem,
    }


# ── 1. LEADERBOARD ────────────────────────────────────────────────────────────

def plot_leaderboard():
    """Bar chart of best-per-model per benchmark vs SOTA."""
    rows = load_results()
    if not rows:
        print("  No results.tsv — skipping leaderboard.")
        return

    benchmarks = sorted({r["benchmark"] for r in rows})
    fig, axes  = plt.subplots(1, len(benchmarks),
                               figsize=(5.5 * len(benchmarks), 7))
    if len(benchmarks) == 1:
        axes = [axes]
    fig.suptitle("SciML Leaderboard — Best per Model per Benchmark",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, bm in zip(axes, benchmarks):
        sota   = SOTA.get(bm)
        bm_rows = [r for r in rows if r["benchmark"] == bm]

        # Best per model (keep only, or best overall)
        best: dict[str, float] = {}
        for r in bm_rows:
            m   = r.get("model", "?")
            val = r["val"]
            if not math.isnan(val):
                if m not in best or val < best[m]:
                    best[m] = val

        if not best:
            ax.text(0.5, 0.5, "No results yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(bm.replace("_", " "))
            continue

        models = sorted(best, key=lambda m: best[m])
        vals   = [best[m] for m in models]
        colors = [MODEL_COLORS.get(m, "#aaa") for m in models]

        bars = ax.barh(models, vals, color=colors, alpha=0.85, edgecolor="white",
                       linewidth=0.5)

        # Annotate values
        for bar, val in zip(bars, vals):
            ax.text(val * 1.03, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

        # SOTA line
        if sota:
            ax.axvline(sota, color="#2c3e50", linestyle="--", linewidth=1.5,
                       label=f"SOTA {sota:.4f}")
            ax.legend(fontsize=8, loc="lower right")

        ax.set_xlabel("val_l2_rel  (lower is better)", fontsize=10)
        ax.set_xscale("log")
        ax.set_title(bm.replace("_", " "), fontsize=12, fontweight="bold",
                     color=BM_COLORS.get(bm, "black"))
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "leaderboard")


def plot_experiment_timeline():
    """Scatter of all experiments, coloured by keep/discard/crash, with running best."""
    rows = load_results()
    if not rows:
        return

    benchmarks = sorted({r["benchmark"] for r in rows})
    fig, axes  = plt.subplots(len(benchmarks), 1,
                               figsize=(14, 4.5 * len(benchmarks)))
    if len(benchmarks) == 1:
        axes = [axes]
    fig.suptitle("Experiment Timeline — All Runs", fontsize=14,
                 fontweight="bold")

    for ax, bm in zip(axes, benchmarks):
        sota     = SOTA.get(bm)
        bm_rows  = [r for r in rows if r["benchmark"] == bm]
        if not bm_rows:
            continue

        xs      = list(range(1, len(bm_rows) + 1))
        vals    = [r["val"] for r in bm_rows]
        statuses= [r.get("status", "?") for r in bm_rows]
        models  = [r.get("model", "?")  for r in bm_rows]

        for x, v, s, m in zip(xs, vals, statuses, models):
            if math.isnan(v):
                continue
            color  = STATUS_COLORS.get(s, "#aaa")
            marker = "o" if s == "keep" else ("x" if s == "crash" else "^")
            ax.scatter(x, v, color=color, marker=marker, s=60, zorder=3,
                       alpha=0.85)
            if s == "keep":
                ax.annotate(m, (x, v), textcoords="offset points",
                            xytext=(4, 4), fontsize=6, color="#2c3e50")

        # Running best line
        best = float("inf")
        run_xs, run_ys = [], []
        for x, v, s in zip(xs, vals, statuses):
            if s == "keep" and not math.isnan(v) and v < best:
                best = v
                run_xs.append(x)
                run_ys.append(v)
        if run_xs:
            ax.step(run_xs, run_ys, color="#2c3e50", linewidth=2,
                    where="post", label="Running best")

        if sota:
            ax.axhline(sota, color="#e74c3c", linestyle="--", linewidth=1.5,
                       label=f"SOTA {sota:.4f}")

        ax.set_yscale("log")
        ax.set_xlabel("Experiment index")
        ax.set_ylabel("val_l2_rel")
        ax.set_title(bm.replace("_", " "), fontsize=12, fontweight="bold",
                     color=BM_COLORS.get(bm, "black"))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        # Legend for status colours
        for label, clr in STATUS_COLORS.items():
            ax.scatter([], [], color=clr, label=label, s=40)
        ax.legend(fontsize=7, ncol=4, loc="upper right")

    plt.tight_layout()
    _save(fig, "experiment_timeline")


# ── 2. TRAINING CURVES ────────────────────────────────────────────────────────

def _pick_logs(log_arg: Optional[str], n_best: int = 8) -> list[Path]:
    """Return list of log paths to plot."""
    if log_arg:
        return [Path(log_arg)]
    if not LOGS.exists():
        return []
    # Pick n_best logs with final val_l2_rel (keep status preferred)
    candidates = sorted(LOGS.glob("*.log"), key=os.path.getmtime, reverse=True)
    parsed = [(p, parse_log(p)) for p in candidates[:40]]
    parsed = [(p, d) for p, d in parsed if d["val"] is not None]
    parsed.sort(key=lambda x: x[1]["val"] if x[1]["val"] else 1e9)
    return [p for p, _ in parsed[:n_best]]


def plot_training_curves(log_arg: Optional[str] = None):
    """Loss + LR + step-time subplots for top logs."""
    paths = _pick_logs(log_arg)
    if not paths:
        print("  No logs found — skipping training curves.")
        return

    logs  = [parse_log(p) for p in paths]
    valid = [d for d in logs if len(d["steps"]) > 10]
    if not valid:
        return

    ncols = min(4, len(valid))
    nrows = math.ceil(len(valid) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if nrows * ncols > 1 else [axes]
    fig.suptitle("Training Curves — Top Experiments by val_l2_rel",
                 fontsize=13, fontweight="bold")

    for ax, d in zip(axes, valid):
        s = d["steps"]
        l = d["losses"]
        # Smooth loss with EMA
        ema, alpha = l[0], 0.95
        smooth = []
        for v in l:
            ema = alpha * ema + (1 - alpha) * v
            smooth.append(ema)
        ax.plot(s, smooth, linewidth=1.8, color="#2980b9")
        ax.plot(s, l, alpha=0.2, linewidth=0.6, color="#2980b9")

        if d["val"] is not None:
            ax.axhline(d["val"], linestyle="--", color="#e74c3c",
                       linewidth=1.2, label=f"val={d['val']:.4f}")
            ax.legend(fontsize=7)

        ax.set_yscale("log")
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Smoothed loss", fontsize=9)
        ax.set_title(d["name"][:35], fontsize=8, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        # Mean step time annotation
        if len(d["dts"]) > 5:
            mean_dt = float(np.median(d["dts"][5:]))
            n_steps = int(d["meta"].get("num_steps", len(s)))
            ax.text(0.98, 0.98, f"dt={mean_dt:.0f}ms  N={n_steps}",
                    ha="right", va="top", transform=ax.transAxes,
                    fontsize=7, color="#555")

    for ax in axes[len(valid):]:
        ax.set_visible(False)

    plt.tight_layout()
    _save(fig, "training_curves")


def plot_step_time_comparison():
    """Bar chart: median step time per model architecture."""
    paths = list(LOGS.glob("*.log")) if LOGS.exists() else []
    model_dts: dict[str, list[float]] = {}
    for p in paths[:60]:
        d = parse_log(p)
        if len(d["dts"]) < 10:
            continue
        # Infer model name from log file name
        name = p.stem.lower()
        if "rfno" in name:    model = "RFNO"
        elif "afno" in name:  model = "AFNO"
        elif "ffno" in name:  model = "FFNO"
        elif "fno" in name:   model = "FNO"
        elif "uno" in name:   model = "UNO"
        elif "wno" in name:   model = "WNO"
        elif "deeponet" in name: model = "DeepONet"
        else:                 model = "Other"
        model_dts.setdefault(model, []).append(
            float(np.median(d["dts"][5:]))   # skip compile spike
        )

    if not model_dts:
        return

    models = sorted(model_dts, key=lambda m: np.median(model_dts[m]))
    medians = [np.median(model_dts[m]) for m in models]
    p25     = [np.percentile(model_dts[m], 25) for m in models]
    p75     = [np.percentile(model_dts[m], 75) for m in models]
    errs    = [[med - lo for med, lo in zip(medians, p25)],
               [hi - med for med, hi in zip(medians, p75)]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(models, medians,
                  color=[MODEL_COLORS.get(m, "#aaa") for m in models],
                  alpha=0.85, edgecolor="white")
    ax.errorbar(models, medians, yerr=errs, fmt="none",
                color="#2c3e50", capsize=4, linewidth=1.5)

    # FNO reference line
    if "FNO" in model_dts:
        ref = np.median(model_dts["FNO"])
        ax.axhline(ref, color="#2980b9", linestyle="--", linewidth=1.5,
                   label=f"FNO baseline {ref:.0f}ms")
        ax.legend(fontsize=9)

    # Steps-in-budget annotation
    budget = 300_000   # ms
    for bar, m in zip(bars, models):
        dt  = bar.get_height()
        est = int(budget / max(dt, 1))
        ax.text(bar.get_x() + bar.get_width() / 2, dt + 0.5,
                f"~{est} steps", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Median step time (ms)", fontsize=11)
    ax.set_title("Step Time by Architecture (lower → more training steps in 5 min)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "step_time_comparison")


# ── 3. DATA SAMPLES ───────────────────────────────────────────────────────────

def _get_data_samples(benchmark: str, n: int = 6) -> tuple:
    """Load a few samples from the benchmark data generators."""
    try:
        sys.path.insert(0, str(REPO))
        from prepare import GRID_SIZE, make_dataloader
        from benchmarks_ext import EXT_BENCHMARKS, make_ext_dataloader
        import mlx.core as mx

        if benchmark in EXT_BENCHMARKS:
            loader = make_ext_dataloader(benchmark, "train", n)
        else:
            loader = make_dataloader(benchmark, "train", n)
        x, y = next(loader)
        return np.array(x), np.array(y)
    except Exception as e:
        print(f"  Could not load {benchmark} data: {e}")
        return None, None


def plot_data_samples(benchmark: Optional[str] = None):
    """Show sample input/output pairs for each benchmark."""
    try:
        from prepare import GRID_SIZE
        grid = np.linspace(0, 1, GRID_SIZE)
    except Exception:
        GRID_SIZE = 64
        grid = np.linspace(0, 1, GRID_SIZE)

    benchmarks = [benchmark] if benchmark else ["burgers_1d", "kdv_1d", "wave_1d"]
    n_samples  = 5

    for bm in benchmarks:
        inp, tgt = _get_data_samples(bm, n_samples)
        if inp is None:
            continue

        fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 6))
        fig.suptitle(f"{bm.replace('_', ' ')} — {n_samples} training samples",
                     fontsize=13, fontweight="bold")
        color = BM_COLORS.get(bm, "#333")

        for i in range(min(n_samples, len(inp))):
            x_i  = inp[i]
            y_i  = tgt[i]
            x_1d = x_i if x_i.ndim == 1 else x_i[:, 0]
            y_1d = y_i if y_i.ndim == 1 else y_i[:, 0]

            ax_in  = axes[0, i]
            ax_out = axes[1, i]

            ax_in.plot(grid[:len(x_1d)], x_1d, color=color, linewidth=1.5)
            ax_in.set_title(f"Input u(x,0) #{i+1}", fontsize=9)
            ax_in.set_xlabel("x")
            ax_in.grid(alpha=0.3)

            ax_out.plot(grid[:len(y_1d)], y_1d, color=color, linewidth=1.5,
                        linestyle="-")
            ax_out.set_title(f"Target u(x,T) #{i+1}", fontsize=9)
            ax_out.set_xlabel("x")
            ax_out.grid(alpha=0.3)

        for ax in axes.flatten():
            ax.spines[["top", "right"]].set_visible(False)

        axes[0, 0].set_ylabel("Amplitude", fontsize=10)
        axes[1, 0].set_ylabel("Amplitude", fontsize=10)
        plt.tight_layout()
        _save(fig, f"data_samples_{bm}")

    # If 2D benchmark
    bm2d_list = [benchmark] if benchmark in ("darcy_2d",) else (
        ["darcy_2d"] if not benchmark else []
    )
    for bm in bm2d_list:
        inp, tgt = _get_data_samples(bm, 4)
        if inp is None:
            continue
        try:
            side = int(inp.shape[1] ** 0.5)
            fig, axes = plt.subplots(2, 4, figsize=(16, 6))
            fig.suptitle(f"{bm.replace('_', ' ')} — 4 samples", fontsize=13,
                         fontweight="bold")
            for i in range(4):
                im0 = axes[0, i].imshow(inp[i].reshape(side, side), cmap="viridis")
                axes[0, i].set_title(f"Input #{i+1}", fontsize=9)
                plt.colorbar(im0, ax=axes[0, i], fraction=0.046)
                im1 = axes[1, i].imshow(tgt[i].reshape(side, side), cmap="inferno")
                axes[1, i].set_title(f"Target #{i+1}", fontsize=9)
                plt.colorbar(im1, ax=axes[1, i], fraction=0.046)
            plt.tight_layout()
            _save(fig, f"data_samples_{bm}")
        except Exception:
            pass


# ── 4. SOLVER VALIDATION ─────────────────────────────────────────────────────

def plot_solver_validation():
    """
    Compare numerical solvers against analytical solutions.

    Wave equation (ut0=0): d'Alembert → u(x,t) = [u0(x+ct) + u0(x-ct)] / 2
    KdV:  compare single-soliton numerical vs analytical
    """
    try:
        from prepare import GRID_SIZE, solve_wave_batch
        from benchmarks_ext import WAVE_C, WAVE_T, WAVE_NSTEPS
        import mlx.core as mx
    except Exception as e:
        print(f"  Solver validation unavailable: {e}")
        return

    N    = GRID_SIZE
    c    = WAVE_C
    T    = WAVE_T
    x    = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # ── Wave equation validation ──────────────────────────────────────────────
    rng  = np.random.RandomState(42)
    n_modes = 4
    u0   = np.zeros(N)
    for k in range(1, n_modes + 1):
        phi = rng.uniform(0, 2 * np.pi)
        u0 += np.sin(k * x + phi) / k

    ut0 = np.zeros_like(u0)

    # Numerical
    u_num = np.array(solve_wave_batch(u0[None], ut0[None], c=c, T=T,
                                      n_steps=WAVE_NSTEPS))[0]

    # Analytical: d'Alembert with periodic wrapping
    # u(x,t) = [u0(x-ct) + u0(x+ct)] / 2
    def dAlembert(u0_, x_, c_, t_):
        from scipy.interpolate import interp1d
        L    = 2 * np.pi
        xp   = np.mod(x_ + c_ * t_, L)
        xm   = np.mod(x_ - c_ * t_, L)
        f    = interp1d(x_, u0_, kind="cubic", assume_sorted=True,
                        fill_value="extrapolate")
        return 0.5 * (f(xp) + f(xm))

    try:
        u_ana = dAlembert(u0, x, c, T)
        err   = float(np.sqrt(np.mean((u_num - u_ana) ** 2)) /
                      (np.sqrt(np.mean(u_ana ** 2)) + 1e-8))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle("Wave Equation Solver Validation (ut₀=0)",
                     fontsize=13, fontweight="bold")

        axes[0].plot(x, u0,    color="#2980b9", linewidth=1.8, label="u₀(x)")
        axes[0].set_title("Initial condition u₀(x)", fontsize=11)
        axes[0].legend()

        axes[1].plot(x, u_ana, color="#27ae60", linewidth=2.0, label="Analytical")
        axes[1].plot(x, u_num, color="#e74c3c", linewidth=1.4,
                     linestyle="--", label="Numerical (Störmer-Verlet)")
        axes[1].set_title(f"Solution at t={T}  |  rel-L2 error={err:.2e}",
                          fontsize=10)
        axes[1].legend()

        axes[2].plot(x, u_num - u_ana, color="#e67e22", linewidth=1.5)
        axes[2].axhline(0, color="black", linewidth=0.5)
        axes[2].set_title(f"Pointwise error (max={float(np.max(np.abs(u_num-u_ana))):.2e})",
                          fontsize=10)
        axes[2].set_ylabel("Numerical − Analytical")

        for ax in axes:
            ax.set_xlabel("x"); ax.grid(alpha=0.3)
            ax.spines[["top","right"]].set_visible(False)

        # Flag concern
        concern = err > 0.01
        status  = "CONCERN — large solver error!" if concern else "OK — solver matches"
        fig.text(0.5, -0.02, f"Solver status: {status}  (rel-L2 = {err:.2e})",
                 ha="center", fontsize=11,
                 color="#e74c3c" if concern else "#27ae60",
                 fontweight="bold")

        plt.tight_layout()
        _save(fig, "solver_validation_wave")
        print(f"    Wave solver rel-L2 vs analytical: {err:.2e}  "
              + ("⚠ CONCERN" if concern else "✓ OK"))

    except Exception as e:
        print(f"  d'Alembert comparison failed: {e}")

    # ── KdV conservation law validation ──────────────────────────────────────
    # KdV form: u_t + u·u_x + u_xxx = 0.  Conserved quantities:
    #   I1 = ∫ u dx        (mass)
    #   I2 = ∫ u² dx       (L2 norm / momentum)
    #   I3 = ∫ (u³/3 - (∂u/∂x)²) dx  (Hamiltonian)
    # ETDRK4 is a Runge-Kutta exponential integrator that conserves these.
    try:
        from prepare import solve_kdv_batch
        from benchmarks_ext import KDV_T, KDV_NSTEPS

        rng = np.random.RandomState(7)
        n_samples = 8
        u0s = []
        for _ in range(n_samples):
            ic = np.zeros(N)
            for k in range(1, 5):
                phi = rng.uniform(0, 2 * np.pi)
                ic += np.sin(k * x + phi) / k
            u0s.append(ic)
        u0_batch = np.array(u0s, dtype=np.float32)
        uT_batch = np.array(solve_kdv_batch(u0_batch, T=KDV_T,
                                             n_steps=KDV_NSTEPS))

        dx = 2 * np.pi / N
        I1_in  = u0_batch.mean(axis=1) * 2 * np.pi
        I1_out = uT_batch.mean(axis=1) * 2 * np.pi
        I2_in  = (u0_batch ** 2).mean(axis=1) * 2 * np.pi
        I2_out = (uT_batch ** 2).mean(axis=1) * 2 * np.pi

        rel_I1 = np.abs((I1_out - I1_in) / (np.abs(I1_in) + 1e-8))
        rel_I2 = np.abs((I2_out - I2_in) / (np.abs(I2_in) + 1e-8))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle("KdV Solver Validation — Conservation Laws",
                     fontsize=13, fontweight="bold")

        # Plot sample trajectory
        axes[0].plot(x, u0_batch[0], color="#3498db", linewidth=1.8, label="u₀(x)")
        axes[0].plot(x, uT_batch[0], color="#e74c3c", linewidth=1.8,
                     linestyle="--", label=f"u(x,T={KDV_T})")
        axes[0].set_title("Sample: Initial → Final", fontsize=11)
        axes[0].legend(fontsize=8)

        # Conservation of mass I1
        axes[1].scatter(range(n_samples), rel_I1, color="#3498db", s=60, zorder=3)
        axes[1].axhline(0.01, color="#e74c3c", linestyle="--", linewidth=1.2,
                        label="1% threshold")
        axes[1].set_yscale("log")
        axes[1].set_title("Mass conservation |ΔI₁/I₁|", fontsize=11)
        axes[1].legend(fontsize=8)
        axes[1].set_xlabel("Sample")

        # Conservation of L2 norm I2
        axes[2].scatter(range(n_samples), rel_I2, color="#27ae60", s=60, zorder=3)
        axes[2].axhline(0.01, color="#e74c3c", linestyle="--", linewidth=1.2,
                        label="1% threshold")
        axes[2].set_yscale("log")
        axes[2].set_title("L2-norm conservation |ΔI₂/I₂|", fontsize=11)
        axes[2].legend(fontsize=8)
        axes[2].set_xlabel("Sample")

        for ax in axes:
            ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)

        max_I1_err = float(rel_I1.max())
        max_I2_err = float(rel_I2.max())
        concern = max_I1_err > 0.01 or max_I2_err > 0.05
        status  = "⚠ CONCERN — conservation violated" if concern else "✓ OK — conservation holds"
        fig.text(0.5, -0.02,
                 f"KdV solver: {status}  "
                 f"(mass err={max_I1_err:.2e}, L2 err={max_I2_err:.2e})",
                 ha="center", fontsize=11,
                 color="#e74c3c" if concern else "#27ae60", fontweight="bold")

        plt.tight_layout()
        _save(fig, "solver_validation_kdv")
        print(f"    KdV conservation: mass err={max_I1_err:.2e}, "
              f"L2 err={max_I2_err:.2e}  "
              + ("⚠ CONCERN" if concern else "✓ OK"))

    except Exception as e:
        print(f"  KdV conservation validation failed: {e}")


# ── 5. SPECTRAL ANALYSIS ─────────────────────────────────────────────────────

def plot_spectral_analysis():
    """Plot mean Fourier power spectrum of inputs and outputs per benchmark."""
    try:
        from prepare import GRID_SIZE
    except Exception:
        GRID_SIZE = 64
    N = GRID_SIZE

    benchmarks = ["burgers_1d", "kdv_1d", "wave_1d"]
    n_samples  = 64

    fig, axes = plt.subplots(1, len(benchmarks),
                              figsize=(6 * len(benchmarks), 5))
    fig.suptitle("Fourier Power Spectrum — Inputs vs Outputs",
                 fontsize=13, fontweight="bold")

    for ax, bm in zip(axes, benchmarks):
        inp, tgt = _get_data_samples(bm, n_samples)
        if inp is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        # 1-D signals only
        if inp.ndim == 3:   # [B, N, C]
            inp = inp[:, :, 0]
        if tgt.ndim == 3:
            tgt = tgt[:, :, 0]

        k  = np.arange(N // 2 + 1)

        inp_fft = np.abs(np.fft.rfft(inp, axis=1)) ** 2      # [B, N//2+1]
        tgt_fft = np.abs(np.fft.rfft(tgt, axis=1)) ** 2

        mean_in  = inp_fft.mean(axis=0)
        mean_out = tgt_fft.mean(axis=0)
        std_in   = inp_fft.std(axis=0)
        std_out  = tgt_fft.std(axis=0)

        ax.semilogy(k, mean_in,  color="#3498db", linewidth=1.8, label="Input u(x,0)")
        ax.fill_between(k, np.maximum(mean_in - std_in, 1e-8),
                        mean_in + std_in, alpha=0.2, color="#3498db")
        ax.semilogy(k, mean_out, color="#e74c3c", linewidth=1.8,
                    linestyle="--", label="Target u(x,T)")
        ax.fill_between(k, np.maximum(mean_out - std_out, 1e-8),
                        mean_out + std_out, alpha=0.2, color="#e74c3c")

        # Mark the m=24 cutoff (our best FNO mode count)
        ax.axvline(24, color="#27ae60", linestyle=":", linewidth=1.5,
                   label="m=24 cutoff (FNO best)")

        ax.set_xlabel("Fourier mode k", fontsize=10)
        ax.set_ylabel("Mean power |û(k)|²", fontsize=10)
        ax.set_title(bm.replace("_", " "), fontsize=11, fontweight="bold",
                     color=BM_COLORS.get(bm, "black"))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim([0, N // 2])

        # Annotation: fraction of power in m<24 modes
        p_in  = float(inp_fft[:, :24].sum() / inp_fft.sum())
        p_out = float(tgt_fft[:, :24].sum() / tgt_fft.sum())
        ax.text(0.97, 0.97, f"m<24 power: in={p_in:.1%}  out={p_out:.1%}",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=8, color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor="#ddd"))

    plt.tight_layout()
    _save(fig, "spectral_analysis")


# ── 6. CONCERN DASHBOARD ─────────────────────────────────────────────────────

def _check_concerns(rows: list[dict]) -> list[dict]:
    """
    Automatically scan results for known problem patterns.
    Returns list of concern dicts with severity, label, message.
    """
    concerns = []

    # Group by benchmark
    by_bm: dict[str, list] = {}
    for r in rows:
        by_bm.setdefault(r["benchmark"], []).append(r)

    for bm, bm_rows in by_bm.items():
        kept    = [r for r in bm_rows if r.get("status") == "keep"]
        all_val = [r["val"] for r in bm_rows if not math.isnan(r["val"])]
        sota    = SOTA.get(bm)

        # Concern: near-random performance
        if kept:
            best_val = min(r["val"] for r in kept)
            if best_val > 0.9:
                concerns.append({
                    "severity": "critical",
                    "benchmark": bm,
                    "label": "Near-random performance",
                    "message": f"{bm}: best={best_val:.4f} ≈ 1.0 (random baseline). "
                               "Data may be ill-posed, mislabelled, or solver is wrong.",
                })
            elif sota and best_val > sota * 5:
                concerns.append({
                    "severity": "warning",
                    "benchmark": bm,
                    "label": f"Far from SOTA ({best_val/sota:.1f}×)",
                    "message": f"{bm}: best={best_val:.4f}, SOTA={sota:.4f}. "
                               f"Gap={best_val/sota:.1f}×.",
                })
            elif sota and best_val < sota * 0.5:
                concerns.append({
                    "severity": "note",
                    "benchmark": bm,
                    "label": "Beating SOTA",
                    "message": f"{bm}: best={best_val:.4f} is {sota/best_val:.1f}× "
                               f"better than SOTA={sota:.4f}. Validate solver accuracy.",
                })

        # Concern: all experiments crashing
        crashes = [r for r in bm_rows if r.get("status") == "crash"]
        if len(crashes) > len(bm_rows) * 0.3:
            concerns.append({
                "severity": "warning",
                "benchmark": bm,
                "label": f"{len(crashes)}/{len(bm_rows)} crashes",
                "message": f"{bm}: {len(crashes)} out of {len(bm_rows)} experiments crashed.",
            })

        # Concern: no improvement trend
        if len(kept) >= 5:
            recent = sorted(kept, key=lambda r: r.get("commit", ""))[-5:]
            recent_vals = [r["val"] for r in recent]
            if all(v >= recent_vals[0] for v in recent_vals[1:]):
                concerns.append({
                    "severity": "warning",
                    "benchmark": bm,
                    "label": "Plateau — no recent improvement",
                    "message": f"{bm}: last 5 'keep' results show no improvement. "
                               "Consider new architectures or data augmentation.",
                })

    # Concern: AFNO/FFNO architectures consistently failing vs FNO
    for arch in ("AFNO", "FFNO"):
        arch_rows = [r for r in rows if r.get("model") == arch
                     and not math.isnan(r["val"])]
        fno_rows  = [r for r in rows if r.get("model") == "FNO"
                     and not math.isnan(r["val"])
                     and r["benchmark"] in {r2["benchmark"] for r2 in arch_rows}]
        if arch_rows and fno_rows:
            best_arch = min(r["val"] for r in arch_rows)
            best_fno  = min(r["val"] for r in fno_rows)
            if best_arch > best_fno * 1.5:
                concerns.append({
                    "severity": "note",
                    "benchmark": "cross-benchmark",
                    "label": f"{arch} underperforms FNO",
                    "message": f"{arch} best={best_arch:.4f} vs FNO best={best_fno:.4f}. "
                               "Step-time cost likely offset any accuracy benefit.",
                })

    return concerns


def plot_concern_dashboard():
    """Visual concern report: severity table + per-benchmark health bars."""
    rows     = load_results()
    concerns = _check_concerns(rows)
    if not rows:
        print("  No results to analyse.")
        return

    # Per-benchmark stats
    by_bm: dict[str, dict] = {}
    for r in rows:
        bm   = r["benchmark"]
        sota = SOTA.get(bm)
        if bm not in by_bm:
            by_bm[bm] = {"total": 0, "kept": 0, "crashed": 0,
                          "best": float("inf"), "sota": sota}
        by_bm[bm]["total"] += 1
        if r.get("status") == "keep" and not math.isnan(r["val"]):
            by_bm[bm]["kept"] += 1
            by_bm[bm]["best"] = min(by_bm[bm]["best"], r["val"])
        if r.get("status") == "crash":
            by_bm[bm]["crashed"] += 1

    benchmarks = sorted(by_bm)
    n_concerns = len(concerns)

    fig = plt.figure(figsize=(14, max(7, 2.5 * len(benchmarks) + 2 * n_concerns)))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[len(benchmarks), max(1, n_concerns)],
                             hspace=0.5)

    # ── Panel 1: benchmark health ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Benchmark Health Overview", fontsize=13, fontweight="bold",
                  pad=12)
    ax1.axis("off")

    col_labels = ["Benchmark", "Total\nRuns", "Kept", "Crashes",
                  "Best Val", "SOTA", "Gap",  "Health"]
    n_cols = len(col_labels)
    n_rows = len(benchmarks)
    cell_h = 1.0 / (n_rows + 1.5)
    col_widths = [0.22, 0.07, 0.07, 0.08, 0.10, 0.10, 0.09, 0.27]

    # Header
    xs = [sum(col_widths[:i]) + col_widths[i] / 2 for i in range(n_cols)]
    for x, lab in zip(xs, col_labels):
        ax1.text(x, 1 - cell_h * 0.5, lab, ha="center", va="center",
                 fontsize=9, fontweight="bold", color="white",
                 transform=ax1.transAxes)
    ax1.add_patch(FancyBboxPatch((0, 1 - cell_h), 1, cell_h,
                                  transform=ax1.transAxes, clip_on=False,
                                  facecolor="#2c3e50", edgecolor="none"))

    for row_i, bm in enumerate(benchmarks):
        info  = by_bm[bm]
        best  = info["best"] if info["best"] < float("inf") else None
        sota  = info["sota"]
        gap   = (best / sota) if (best and sota) else None
        y_top = 1 - cell_h * (row_i + 1.5)

        # Health classification
        if best is None:
            health, health_color = "No data", "#7f8c8d"
        elif best > 0.9:
            health, health_color = "CRITICAL — near-random", "#e74c3c"
        elif gap and gap < 0.5:
            health, health_color = "EXCELLENT — beats SOTA", "#27ae60"
        elif gap and gap < 2.0:
            health, health_color = "GOOD — near SOTA", "#2ecc71"
        elif gap and gap < 10.0:
            health, health_color = f"FAIR — {gap:.1f}× from SOTA", "#f39c12"
        else:
            health, health_color = f"POOR — {gap:.0f}× from SOTA", "#e74c3c"

        row_bg = "#f8f9fa" if row_i % 2 == 0 else "#ecf0f1"
        ax1.add_patch(FancyBboxPatch((0, y_top), 1, cell_h,
                                      transform=ax1.transAxes, clip_on=False,
                                      facecolor=row_bg, edgecolor="none"))

        vals_row = [
            bm.replace("_", " "),
            str(info["total"]),
            str(info["kept"]),
            str(info["crashed"]) if info["crashed"] > 0 else "0",
            f"{best:.4f}" if best else "—",
            f"{sota:.4f}" if sota else "—",
            f"{gap:.1f}×" if gap else "—",
            health,
        ]
        for col_i, (x, v) in enumerate(zip(xs, vals_row)):
            color = health_color if col_i == n_cols - 1 else "#2c3e50"
            weight= "bold" if col_i in (0, n_cols - 1) else "normal"
            ax1.text(x, y_top + cell_h / 2, v, ha="center", va="center",
                     fontsize=8.5, color=color, fontweight=weight,
                     transform=ax1.transAxes)

    # ── Panel 2: concern list ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_title(f"Automated Concerns ({n_concerns} found)",
                  fontsize=13, fontweight="bold", pad=12)

    severity_colors = {
        "critical": "#e74c3c", "warning": "#f39c12", "note": "#3498db",
    }
    severity_icons  = {"critical": "🔴", "warning": "⚠️", "note": "ℹ️"}

    if not concerns:
        ax2.text(0.5, 0.5, "No concerns detected ✓",
                 ha="center", va="center", fontsize=12, color="#27ae60",
                 transform=ax2.transAxes, fontweight="bold")
    else:
        y = 0.95
        dy = 0.9 / len(concerns)
        for c in concerns:
            sev    = c["severity"]
            color  = severity_colors.get(sev, "#555")
            prefix = f"[{sev.upper()}]  {c['label']}"
            ax2.text(0.01, y, prefix, transform=ax2.transAxes,
                     fontsize=9, color=color, fontweight="bold", va="top")
            ax2.text(0.01, y - 0.025, "    " + c["message"],
                     transform=ax2.transAxes, fontsize=8, color="#555", va="top",
                     wrap=True)
            y -= dy

    plt.tight_layout()
    _save(fig, "concern_dashboard")

    # Print text summary
    print(f"\n  {'='*60}")
    print(f"  CONCERN REPORT ({n_concerns} issues)")
    print(f"  {'='*60}")
    sev_order = {"critical": 0, "warning": 1, "note": 2}
    for c in sorted(concerns, key=lambda x: sev_order.get(x["severity"], 3)):
        icon = {"critical": "🔴", "warning": "⚠️", "note": "ℹ️"}.get(c["severity"],"  ")
        print(f"  {icon}  [{c['severity'].upper():8s}] {c['label']}")
        print(f"          {c['message']}")
    print()


# ── 7. ARCHITECTURE COMPARISON RADAR ─────────────────────────────────────────

def plot_architecture_comparison():
    """Radar chart comparing architectures across: accuracy, speed, stability."""
    rows   = load_results()
    if not rows:
        return

    # Build per-model stats across all benchmarks
    model_stats: dict[str, dict] = {}
    for r in rows:
        m   = r.get("model", "?")
        val = r["val"]
        if math.isnan(val) or m in ("?",):
            continue
        if m not in model_stats:
            model_stats[m] = {"vals": [], "crashes": 0, "total": 0, "dts": []}
        model_stats[m]["vals"].append(val)
        model_stats[m]["total"] += 1
        if r.get("status") == "crash":
            model_stats[m]["crashes"] += 1

    # Load step times from logs
    if LOGS.exists():
        for p in LOGS.glob("*.log"):
            d   = parse_log(p)
            if len(d["dts"]) < 5:
                continue
            n   = p.stem.lower()
            for m in model_stats:
                if m.lower() in n:
                    model_stats[m]["dts"].append(float(np.median(d["dts"][5:])))
                    break

    if len(model_stats) < 2:
        return

    # Normalise to [0,1] scores (higher = better)
    models = sorted(model_stats, key=lambda m: min(model_stats[m]["vals"]
                                                   if model_stats[m]["vals"]
                                                   else [1.0]))[:8]

    def safe_min(lst, default=1.0):
        return min(lst) if lst else default

    best_vals  = {m: safe_min(model_stats[m]["vals"])    for m in models}
    mean_dts   = {m: safe_min(model_stats[m]["dts"], 70) for m in models}
    crash_rate = {m: model_stats[m]["crashes"] /
                     max(model_stats[m]["total"], 1)      for m in models}

    global_best_val = min(best_vals.values())
    global_worst_dt = max(mean_dts.values())
    # accuracy score: 1 = best val, 0 = 10× worse
    acc_score  = {m: max(0, 1 - math.log10(max(best_vals[m] / global_best_val, 1)) / 1.0)
                  for m in models}
    # speed score: 1 = fastest, 0 = 2× slowest
    spd_score  = {m: 1 - (mean_dts[m] - min(mean_dts.values())) /
                           max(global_worst_dt - min(mean_dts.values()), 1)
                  for m in models}
    # stability: 1 = no crashes, 0 = all crash
    stab_score = {m: 1.0 - crash_rate[m] for m in models}
    # kept_ratio: fraction of runs that improved
    kept_score = {m: sum(1 for r in rows if r.get("model") == m
                         and r.get("status") == "keep") /
                     max(model_stats[m]["total"], 1)
                  for m in models}

    categories  = ["Accuracy", "Speed", "Stability", "Keep Rate"]
    N_cat       = len(categories)
    angles      = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
    angles     += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))
    fig.suptitle("Architecture Comparison Radar\n(higher = better in all axes)",
                 fontsize=12, fontweight="bold")

    for m in models:
        scores = [acc_score[m], spd_score[m], stab_score[m], kept_score[m]]
        scores += scores[:1]
        color  = MODEL_COLORS.get(m, "#aaa")
        ax.plot(angles, scores, color=color, linewidth=2.0, label=m)
        ax.fill(angles, scores, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    plt.tight_layout()
    _save(fig, "architecture_radar")


# ── 8. INPUT DISTRIBUTION ANALYSIS ───────────────────────────────────────────

def plot_input_statistics():
    """Show distribution of input and target amplitudes, smoothness, and dynamic range."""
    benchmarks = ["burgers_1d", "kdv_1d", "wave_1d"]
    n_samples  = 256

    stats_per_bm: dict[str, dict] = {}
    for bm in benchmarks:
        inp, tgt = _get_data_samples(bm, n_samples)
        if inp is None:
            continue
        if inp.ndim == 3:
            inp = inp[:, :, 0]
        if tgt.ndim == 3:
            tgt = tgt[:, :, 0]

        # Gradient magnitude as smoothness proxy
        inp_grad = np.abs(np.diff(inp, axis=1)).mean(axis=1)
        tgt_grad = np.abs(np.diff(tgt, axis=1)).mean(axis=1)

        stats_per_bm[bm] = {
            "inp_std":  inp.std(axis=1),
            "tgt_std":  tgt.std(axis=1),
            "inp_grad": inp_grad,
            "tgt_grad": tgt_grad,
            "inp_max":  np.abs(inp).max(axis=1),
            "tgt_max":  np.abs(tgt).max(axis=1),
        }

    if not stats_per_bm:
        return

    n_bms = len(stats_per_bm)
    fig, axes = plt.subplots(3, n_bms, figsize=(5 * n_bms, 11))
    if n_bms == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle("Input / Target Distribution Analysis",
                 fontsize=13, fontweight="bold")

    row_labels = ["Std dev (amplitude)", "Mean |∂u/∂x| (smoothness)",
                  "Max amplitude |u|∞"]

    for col_i, (bm, s) in enumerate(stats_per_bm.items()):
        for row_i, (k_in, k_out, ylabel) in enumerate(zip(
            ["inp_std",  "inp_grad", "inp_max"],
            ["tgt_std",  "tgt_grad", "tgt_max"],
            row_labels
        )):
            ax = axes[row_i, col_i]
            ax.hist(s[k_in],  bins=30, color="#3498db", alpha=0.7, label="Input",
                    density=True)
            ax.hist(s[k_out], bins=30, color="#e74c3c", alpha=0.7, label="Target",
                    density=True)
            if row_i == 0:
                ax.set_title(bm.replace("_", " "), fontsize=11, fontweight="bold",
                             color=BM_COLORS.get(bm, "black"))
            if col_i == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "input_statistics")


# ── Architecture sanity-check (merged from vis.py) ───────────────────────────

def plot_model_arch(benchmark: str = "burgers_1d", model_type: str = "FNO",
                   modes: int = 16, levels: int = 3,
                   hidden: int = 64, layers: int = 4, samples: int = 3):
    """Run an untrained model forward pass and plot prediction vs ground truth.

    Useful for verifying shape compatibility before a full training run.
    Saves to figs/arch_<benchmark>_<model_type>.png.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from prepare import make_dataloader, GRID_SIZE
    from research_plugins import MODEL_REGISTRY

    is_1d  = benchmark.endswith("_1d")
    mk     = ("FNO2D" if model_type == "FNO" and not is_1d else model_type)
    model  = MODEL_REGISTRY.build(mk, n_modes=modes, hidden_dim=hidden,
                                   n_layers=layers, n_levels=levels)
    mx.eval(model.parameters())

    loader       = make_dataloader(benchmark, "val", samples)
    x, y         = next(loader)
    if model_type == "DeepONet":
        B      = x.shape[0]
        coords = mx.linspace(0, 1, GRID_SIZE).reshape(1, GRID_SIZE, 1)
        coords = mx.broadcast_to(coords, (B, GRID_SIZE, 1))
        pred   = model(x, coords)
    else:
        pred = model(x)
    mx.eval(pred)

    x_np   = np.array(x)
    y_np   = np.array(y)
    p_np   = np.array(pred)

    if is_1d:
        fig, axes = plt.subplots(samples, 1, figsize=(10, 3 * samples))
        if samples == 1:
            axes = [axes]
        grid = np.linspace(0, 1, x_np.shape[1])
        for i in range(samples):
            axes[i].plot(grid, x_np[i], "k--", alpha=0.7, label="Input")
            axes[i].plot(grid, y_np[i], "b-",  linewidth=2, label="Target")
            axes[i].plot(grid, p_np[i], "r--", linewidth=2, label="Pred (untrained)")
            axes[i].legend(fontsize=8)
            axes[i].set_title(f"Sample {i+1} — untrained {model_type}")
    else:
        fig, axes = plt.subplots(samples, 3, figsize=(15, 4 * samples))
        if samples == 1:
            axes = axes[None, :]
        for i in range(samples):
            kw = dict(cmap="RdBu_r",
                      vmin=float(min(y_np[i].min(), p_np[i].min())),
                      vmax=float(max(y_np[i].max(), p_np[i].max())))
            for ax, img, title in zip(axes[i],
                                       [x_np[i], y_np[i], p_np[i]],
                                       ["Input", "Target", f"Pred ({model_type})"]):
                im = ax.imshow(img, **(dict(cmap="viridis") if title == "Input" else kw))
                ax.set_title(f"{title} {i+1}")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    _save(fig, f"arch_{benchmark}_{model_type}")


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_MODES = [
    "leaderboard", "timeline", "training", "step_time",
    "data", "validation", "spectral", "concerns",
    "architecture", "distribution",
]
# "arch" is not included in "all" — it requires a live model forward pass



def main():
    global SHOW
    p = argparse.ArgumentParser(description="SciML Visualization Suite")
    p.add_argument("--mode",      default="all",
                   help=f"One of: {', '.join(ALL_MODES+['all'])}")
    p.add_argument("--benchmark", default=None,
                   help="Filter data/spectral plots to this benchmark")
    p.add_argument("--log",       default=None,
                   help="Path to a specific .log file (for training mode)")
    p.add_argument("--show",      action="store_true",
                   help="Display plots interactively (requires display)")
    p.add_argument("--model",     default="FNO",
                   help="Model type for --mode arch (default: FNO)")
    p.add_argument("--hidden",    type=int, default=64,
                   help="Hidden dim for --mode arch")
    p.add_argument("--layers",    type=int, default=4,
                   help="Num layers for --mode arch")
    p.add_argument("--modes",     type=int, default=16,
                   help="Fourier modes for --mode arch")
    p.add_argument("--levels",    type=int, default=3,
                   help="Wavelet levels for --mode arch (WNO)")
    p.add_argument("--samples",   type=int, default=3,
                   help="Num samples to plot for --mode arch")
    args = p.parse_args()

    SHOW = args.show
    if SHOW:
        matplotlib.use("TkAgg")
        plt.switch_backend("TkAgg")

    FIGS.mkdir(exist_ok=True)
    mode = args.mode.lower()
    modes_to_run = ALL_MODES if mode == "all" else [mode]

    for m in modes_to_run:
        print(f"\n── {m} ──")
        if m == "leaderboard":
            plot_leaderboard()
            plot_experiment_timeline()
        elif m == "timeline":
            plot_experiment_timeline()
        elif m == "training":
            plot_training_curves(args.log)
        elif m == "step_time":
            plot_step_time_comparison()
        elif m == "data":
            plot_data_samples(args.benchmark)
        elif m == "validation":
            plot_solver_validation()
        elif m == "spectral":
            plot_spectral_analysis()
        elif m == "concerns":
            plot_concern_dashboard()
        elif m == "architecture":
            plot_architecture_comparison()
        elif m == "distribution":
            plot_input_statistics()
        elif m == "arch":
            plot_model_arch(
                benchmark=args.benchmark or "burgers_1d",
                model_type=args.model,
                modes=args.modes, levels=args.levels,
                hidden=args.hidden, layers=args.layers,
                samples=args.samples,
            )
        else:
            print(f"  Unknown mode: {m!r}")

    print(f"\nAll plots saved to {FIGS}/")


if __name__ == "__main__":
    main()
