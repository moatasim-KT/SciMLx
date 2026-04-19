"""Architecture visualization generator for SciML models.

Responsibilities:
  1. Rename existing figs/arch/ files to the canonical {RegistryKey}.png format.
  2. Generate matplotlib block-diagram PNGs for every MODEL_REGISTRY key that
     does not already have a visualization.

Run:
    uv run python scripts/gen_arch_viz.py
    uv run python scripts/gen_arch_viz.py --dry-run   # preview only, no writes
    uv run python scripts/gen_arch_viz.py --force      # regenerate script-made files only

SAFETY: A manifest at figs/arch/.generated_manifest.json tracks which PNGs were
created by this script.  --force ONLY regenerates files listed in the manifest.
User-provided architecture images (not in manifest) are NEVER overwritten.

Output: figs/arch/{RegistryKey}.png  (800×440 px, transparent-safe dark background)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT     = Path(__file__).parent.parent
ARCH_DIR      = REPO_ROOT / "figs" / "arch"
MANIFEST_PATH = ARCH_DIR / ".generated_manifest.json"
ARCH_DIR.mkdir(parents=True, exist_ok=True)


def _load_manifest() -> set[str]:
    """Return set of filenames (basename only) known to be script-generated."""
    if MANIFEST_PATH.exists():
        try:
            return set(json.loads(MANIFEST_PATH.read_text()))
        except Exception:
            pass
    return set()


def _save_manifest(manifest: set[str], dry: bool) -> None:
    if not dry:
        MANIFEST_PATH.write_text(json.dumps(sorted(manifest), indent=2))

# ── Canonical rename table ────────────────────────────────────────────────────
# Maps existing filename (without dir) → canonical registry-key name (without .png)
RENAME_MAP: dict[str, str] = {
    "deeponet_arch.png":   "DeepONet",
    "FEDONet.png":         "FEDONet2D",
    "fno_arch.png":        "FNO",
    "GNOT.png":            "GNOT",
    "HNN.png":             "HNN",
    "MambaNO.png":         "MambaNO",
    "MemNO.png":           "MemNO",
    "Neural ODE.png":      "NeuralODE",
    "PACMANN.png":         "PACMANN",
    "pinn_factor_arch.png":"PINO",
    "RFNO.png":            "RFNO",
    "S4D.png":             "S4NO",
    "sno_arch.png":        "SNO2D",
    "ssno_arch.png":       "SSNO",
    "TFNO.png":            "TFNO",
    "transolver_arch.png": "Transolver",
    "UNO.png":             "UNO",
    "vsmno_arch.png":      "VSMNO2D",
    "wno_arch.png":        "WNO",
}

# ── Secondary aliases (same arch as primary, create symlink/copy) ─────────────
# Maps secondary registry key → primary key whose PNG to reuse
ALIAS_MAP: dict[str, str] = {
    "FNO2D":        "FNO",
    "FNO_MC":       "FNO",
    "RFNO2D":       "RFNO",
    "GNOT2D":       "GNOT",
    "GNOT_FFNO":    "GNOT",
    "RTFNO":        "TFNO",
    "CPFNO":        "TFNO",
    "TFNO2D":       "TFNO",
    "AFNO":         "FFNO",       # same block-diag MLP idea; FFNO shares spirit
    "PODDeepONet":  "DeepONet",
    "Transolver2D": "Transolver",
    "WNO_GNOT":     "WNO",
    "PINN":         "PINO",
    "UDE":          "NeuralODE",
    "LatentODE":    "NeuralODE",
    "DualDeepONet": "TimeDeepONet",
    "EnergyFNO":    "HNN",
    "HybridFNODeepONet2D": "HybridDecoderDeepONet2D",
    "KAN_FNO":      "cPIKAN_FNO",
    "ModifiedKAN_FNO": "cPIKAN_FNO",
}

# ── Block diagram specs ───────────────────────────────────────────────────────
# Each spec is a list of "rows". A row is a list of Block dicts.
# Blocks on the same row are laid out left-to-right.
# Connector arrows flow: row[i] → row[i+1] (between row centers).
# Special block keys:
#   "label"  : display text (may contain \n)
#   "color"  : fill hex
#   "w"      : relative width (default 1.0)
#   "skip"   : if True, draw a bypass arrow around this row group
#   "branch" : list of parallel sub-rows (drawn side-by-side, merged after)

C_INPUT  = "#22D3EE"   # cyan   — input / output
C_LIFT   = "#7C3AED"   # indigo — linear lift / projection
C_SPEC   = "#A78BFA"   # purple — spectral / Fourier layers
C_ATTN   = "#FBBF24"   # amber  — attention
C_MLP    = "#4ADE80"   # green  — MLP / dense
C_ODE    = "#F87171"   # red    — ODE solver
C_SKIP   = "#484F58"   # gray   — residual / skip

SPECS: dict[str, list[dict[str, Any]]] = {

    # ── FNO family ────────────────────────────────────────────────────────────
    "FNO": [
        {"label": "u(x)\nInput",         "color": C_INPUT},
        {"label": "Linear Lift\n(P)",     "color": C_LIFT},
        {"label": "Spectral Conv\n(top) + W·x (skip)\n+ GELU  × L",
                                          "color": C_SPEC},
        {"label": "Linear Projection\n(Q)","color": C_LIFT},
        {"label": "û(x)\nOutput",         "color": C_INPUT},
    ],

    "RFNO": [
        {"label": "u(x)\nInput",          "color": C_INPUT},
        {"label": "Linear Lift",          "color": C_LIFT},
        {"label": "Real Spectral Conv\n(rFFT basis) + W·x\n+ GELU  × L",
                                          "color": C_SPEC},
        {"label": "Linear Projection",    "color": C_LIFT},
        {"label": "û(x)\nOutput",         "color": C_INPUT},
    ],

    "UNO": [
        {"label": "u(x)\nInput",          "color": C_INPUT},
        {"label": "Lift  P",              "color": C_LIFT},
        {"label": "SpectralConv (down)\n+ W·x  × L/2",
                                          "color": C_SPEC},
        {"label": "SpectralConv (up)\n+ W·x  × L/2",
                                          "color": C_SPEC},
        {"label": "Projection  Q",        "color": C_LIFT},
        {"label": "û(x)\nOutput",         "color": C_INPUT},
    ],

    # ── AFNO / FFNO ───────────────────────────────────────────────────────────
    "FFNO": [
        {"label": "u(x)\nInput",              "color": C_INPUT},
        {"label": "Linear Lift",              "color": C_LIFT},
        {"label": "rFFT\n(real+imag concat)", "color": C_SPEC},
        {"label": "Block-Diagonal MLP\n(shared across modes)", "color": C_MLP},
        {"label": "Soft-Shrink\n(sparsity λ)","color": C_SKIP},
        {"label": "iRFFT\n+ W·x skip",       "color": C_SPEC},
        {"label": "Repeat × L\n+ GELU",      "color": C_LIFT},
        {"label": "Linear Projection",        "color": C_LIFT},
        {"label": "û(x)\nOutput",             "color": C_INPUT},
    ],

    # ── Chebyshev KAN + FNO ───────────────────────────────────────────────────
    "cPIKAN_FNO": [
        {"label": "u(x)\nInput",                  "color": C_INPUT},
        {"label": "Linear Lift",                  "color": C_LIFT},
        {"label": "SpectralConv (top)",           "color": C_SPEC},
        {"label": "+ Chebyshev-KAN skip\n(B-spline basis, degree d)",
                                                  "color": C_MLP},
        {"label": "GELU  × L",                   "color": C_LIFT},
        {"label": "Linear Projection",            "color": C_LIFT},
        {"label": "û(x)\nOutput",                 "color": C_INPUT},
    ],

    # ── DeepONet ──────────────────────────────────────────────────────────────
    "DeepONet": [
        {"label": "u₀(sensor pts)\nInput field",  "color": C_INPUT,  "w": 1.4},
        {"label": "Branch Net\n(Dense MLP)",       "color": C_MLP,    "w": 1.4},
        {"label": "b(u₀) ∈ ℝᵖ\n⊙",               "color": C_SPEC,   "w": 0.7},
        {"label": "Trunk Net\n(coord MLP)",        "color": C_MLP,    "w": 1.4},
        {"label": "Σ bₖ · tₖ(x) + bias\nOutput û(x)",
                                                   "color": C_INPUT,  "w": 1.4},
    ],

    # ── TimeDeepONet ──────────────────────────────────────────────────────────
    "TimeDeepONet": [
        {"label": "u₀(x)\nIC field",           "color": C_INPUT,  "w": 1.2},
        {"label": "Branch 1\n(IC MLP)",         "color": C_MLP,    "w": 1.2},
        {"label": "b₁ ⊙ b₂\nGating",           "color": C_SPEC,   "w": 0.9},
        {"label": "t ∈ [0,T]\nTime scalar",     "color": C_INPUT,  "w": 1.0},
        {"label": "Branch 2\n(Time MLP)",       "color": C_MLP,    "w": 1.2},
        {"label": "dot(T)\nTrunk score",        "color": C_SPEC,   "w": 0.9},
        {"label": "x coords\nTrunk",            "color": C_INPUT,  "w": 1.0},
        {"label": "Trunk Net\n(coord MLP)",     "color": C_MLP,    "w": 1.2},
        {"label": "û(x,t)\nOutput",             "color": C_INPUT,  "w": 1.0},
    ],

    # ── NeuralODE / UDE / LatentODE ───────────────────────────────────────────
    "NeuralODE": [
        {"label": "u(t₀)\nInitial State",         "color": C_INPUT},
        {"label": "ODE-RHS Net\n(MLP: f_θ(u,t))", "color": C_MLP},
        {"label": "Fixed-step RK4\nODE Solver",   "color": C_ODE},
        {"label": "Integrate\n[t₀ → T]",          "color": C_SKIP},
        {"label": "û(T)\nOutput State",           "color": C_INPUT},
    ],

    # ── HNN / EnergyFNO ───────────────────────────────────────────────────────
    "HNN": [
        {"label": "(q,p)\nPhase Space Input",      "color": C_INPUT},
        {"label": "Encode\n(q,p) → (q̃,p̃)",       "color": C_MLP},
        {"label": "H(q̃,p̃)\nHamiltonian MLP",      "color": C_ODE},
        {"label": "∂H/∂p  −∂H/∂q\n(autodiff)",    "color": C_ATTN},
        {"label": "Hamiltonian Step\nq + α·(∂H/∂p)",
                                                    "color": C_SPEC},
        {"label": "û(T)\nOutput",                  "color": C_INPUT},
    ],

    # ── GNOT ──────────────────────────────────────────────────────────────────
    "GNOT": [
        {"label": "u(x)\nInput",                       "color": C_INPUT},
        {"label": "Geometry-aware\nLifting",            "color": C_LIFT},
        {"label": "Cross-Attn\n(query: coords\nkey/val: input)\n× L",
                                                        "color": C_ATTN},
        {"label": "Feed-Forward\nMLP",                  "color": C_MLP},
        {"label": "Projection\n(query grid)",           "color": C_LIFT},
        {"label": "û(x)\nOutput",                       "color": C_INPUT},
    ],

    # ── WNO ───────────────────────────────────────────────────────────────────
    "WNO": [
        {"label": "u(x)\nInput",                 "color": C_INPUT},
        {"label": "Linear Lift",                 "color": C_LIFT},
        {"label": "Haar DWT\n(multi-level)",     "color": C_SPEC},
        {"label": "Wavelet Conv\n(per-level W)", "color": C_SPEC},
        {"label": "IDWT\n+ W·x skip + GELU  × L",
                                                 "color": C_SPEC},
        {"label": "Linear Projection",           "color": C_LIFT},
        {"label": "û(x)\nOutput",                "color": C_INPUT},
    ],

    # ── S4NO ──────────────────────────────────────────────────────────────────
    "S4NO": [
        {"label": "u(x)\nInput sequence",        "color": C_INPUT},
        {"label": "Linear Lift",                 "color": C_LIFT},
        {"label": "S4D Layer\n(diag SSM: Ā,B̄,C̄,D̄)\nO(N log N) conv",
                                                 "color": C_ODE},
        {"label": "+ SpectralConv\nresidual  × L","color": C_SPEC},
        {"label": "GELU + LayerNorm",            "color": C_LIFT},
        {"label": "Linear Projection",           "color": C_LIFT},
        {"label": "û(x)\nOutput",                "color": C_INPUT},
    ],

    # ── SSNO ──────────────────────────────────────────────────────────────────
    "SSNO": [
        {"label": "u(x)\nInput",                 "color": C_INPUT},
        {"label": "Linear Lift",                 "color": C_LIFT},
        {"label": "S4D SSM branch\n(adaptive state)",
                                                 "color": C_ODE},
        {"label": "+ SpectralConv branch\n(frequency)",
                                                 "color": C_SPEC},
        {"label": "Merge + GELU  × L",           "color": C_MLP},
        {"label": "Linear Projection",           "color": C_LIFT},
        {"label": "û(x)\nOutput",                "color": C_INPUT},
    ],

    # ── TFNO / CPFNO ──────────────────────────────────────────────────────────
    "TFNO": [
        {"label": "u(x)\nInput",                          "color": C_INPUT},
        {"label": "Linear Lift",                          "color": C_LIFT},
        {"label": "Tucker SpectralConv\n(rank-r tensor decomp)\n+ W·x + GELU  × L",
                                                          "color": C_SPEC},
        {"label": "Linear Projection",                    "color": C_LIFT},
        {"label": "û(x)\nOutput",                         "color": C_INPUT},
    ],

    # ── Transolver ────────────────────────────────────────────────────────────
    "Transolver": [
        {"label": "u(x)\nInput",                       "color": C_INPUT},
        {"label": "Slice-based\nLifting (Physics Attn)","color": C_LIFT},
        {"label": "Physics-Aware Slice Attn\n(Q=slices, K/V=tokens)\n× L",
                                                       "color": C_ATTN},
        {"label": "Feed-Forward\nMLP + LayerNorm",     "color": C_MLP},
        {"label": "Inverse Slice\nDecoder",            "color": C_LIFT},
        {"label": "û(x)\nOutput",                      "color": C_INPUT},
    ],

    # ── MambaNO ───────────────────────────────────────────────────────────────
    "MambaNO": [
        {"label": "u(x)\nInput",                   "color": C_INPUT},
        {"label": "Linear Lift",                   "color": C_LIFT},
        {"label": "Selective SSM\n(Mamba: Δ,A,B,C)\n+ SpectralConv\n× L",
                                                   "color": C_ODE},
        {"label": "GELU + Norm",                   "color": C_MLP},
        {"label": "Linear Projection",             "color": C_LIFT},
        {"label": "û(x)\nOutput",                  "color": C_INPUT},
    ],

    # ── MemNO ─────────────────────────────────────────────────────────────────
    "MemNO": [
        {"label": "u(x)\nInput",                  "color": C_INPUT},
        {"label": "Linear Lift",                  "color": C_LIFT},
        {"label": "Memory Bank\n(persistent K,V pairs)",
                                                  "color": C_ATTN},
        {"label": "Cross-Attention\n(query ← hidden)\n× L",
                                                  "color": C_ATTN},
        {"label": "SpectralConv\n+ GELU",         "color": C_SPEC},
        {"label": "Linear Projection",            "color": C_LIFT},
        {"label": "û(x)\nOutput",                 "color": C_INPUT},
    ],

    # ── PACMANN ───────────────────────────────────────────────────────────────
    "PACMANN": [
        {"label": "u(x)\nInput",                   "color": C_INPUT},
        {"label": "Positional\nEncoding",          "color": C_LIFT},
        {"label": "Multi-Head\nCross-Attn  × L",   "color": C_ATTN},
        {"label": "Spectral\nMixing",              "color": C_SPEC},
        {"label": "FFN + Norm",                    "color": C_MLP},
        {"label": "Linear Projection",             "color": C_LIFT},
        {"label": "û(x)\nOutput",                  "color": C_INPUT},
    ],

    # ── PINO ──────────────────────────────────────────────────────────────────
    "PINO": [
        {"label": "u(x)\nInput",                    "color": C_INPUT},
        {"label": "FNO Backbone\n(SpectralConv × L)","color": C_SPEC},
        {"label": "û(x) Prediction",               "color": C_MLP},
        {"label": "Physics Residual\nL_PDE(û)",     "color": C_ODE},
        {"label": "L_data + λ·L_PDE\nJoint Loss",  "color": C_ATTN},
        {"label": "û(x)\nOutput",                   "color": C_INPUT},
    ],

    # ── SNO2D ────────────────────────────────────────────────────────────────
    "SNO2D": [
        {"label": "u(x,y)\nInput",                 "color": C_INPUT},
        {"label": "Linear Lift",                   "color": C_LIFT},
        {"label": "2D SpectralConv\n(n_modes × n_modes)\n× L",
                                                   "color": C_SPEC},
        {"label": "+ Bilinear skip\n+ GELU",       "color": C_MLP},
        {"label": "Linear Projection",             "color": C_LIFT},
        {"label": "û(x,y)\nOutput",                "color": C_INPUT},
    ],

    # ── VSMNO2D ───────────────────────────────────────────────────────────────
    "VSMNO2D": [
        {"label": "u(x,y)\nInput",                  "color": C_INPUT},
        {"label": "Lift + Grid Embed",              "color": C_LIFT},
        {"label": "Variable-Scale\nSpectral Mixing\n(multi-resolution)",
                                                    "color": C_SPEC},
        {"label": "+ MLP skip  × L",               "color": C_MLP},
        {"label": "Linear Projection",             "color": C_LIFT},
        {"label": "û(x,y)\nOutput",                "color": C_INPUT},
    ],

    # ── FEDONet2D ─────────────────────────────────────────────────────────────
    "FEDONet2D": [
        {"label": "u(x,y)\nInput",                 "color": C_INPUT},
        {"label": "FE-inspired\nDomain Lift",      "color": C_LIFT},
        {"label": "Local FE Basis\n(hat functions)",
                                                   "color": C_MLP},
        {"label": "Global DeepONet\nTrunk Decode", "color": C_SPEC},
        {"label": "Hybrid\nFE + NO output",        "color": C_MLP},
        {"label": "û(x,y)\nOutput",                "color": C_INPUT},
    ],

    # ── HybridDecoderDeepONet2D ───────────────────────────────────────────────
    "HybridDecoderDeepONet2D": [
        {"label": "u(x,y)\nInput",                      "color": C_INPUT,  "w": 1.2},
        {"label": "Branch: Lift\n→ FNO Blocks × L\n(spatial encoder)",
                                                         "color": C_SPEC,   "w": 1.5},
        {"label": "⊙  Element-wise\nProduct",           "color": C_ATTN,   "w": 0.8},
        {"label": "Coords (x,y)\nTrunk Input",          "color": C_INPUT,  "w": 1.0},
        {"label": "Trunk: MLP\n(coord embedding)",      "color": C_MLP,    "w": 1.3},
        {"label": "Linear Projection\n→ Output",        "color": C_LIFT,   "w": 1.2},
        {"label": "û(x,y)\nOutput",                     "color": C_INPUT,  "w": 1.0},
    ],

    # ── AttentionEnhancedFNO2D ────────────────────────────────────────────────
    "AttentionEnhancedFNO2D": [
        {"label": "u(x,y)\nInput",                          "color": C_INPUT},
        {"label": "Lift + Grid Embed",                      "color": C_LIFT},
        {"label": "SpectralConv2D ──┐\nAxial Attention ──┤  + GELU\nMLP skip ──────┘  × L",
                                                            "color": C_ATTN},
        {"label": "Linear Projection",                      "color": C_LIFT},
        {"label": "û(x,y)\nOutput",                         "color": C_INPUT},
    ],

    # ── HANO2D ────────────────────────────────────────────────────────────────
    "HANO2D": [
        {"label": "u(x,y)\nInput",                             "color": C_INPUT},
        {"label": "Lift + Grid Embed",                         "color": C_LIFT},
        {"label": "SpectralConv2D\n(frequency path)",          "color": C_SPEC},
        {"label": "+ Axial Attention\n(row then col)\n× L",    "color": C_ATTN},
        {"label": "+ Linear skip\n+ GELU",                     "color": C_MLP},
        {"label": "Linear Projection",                         "color": C_LIFT},
        {"label": "û(x,y)\nOutput",                            "color": C_INPUT},
    ],
}

# ── Drawing engine ────────────────────────────────────────────────────────────

def draw_arch(blocks: list[dict], title: str, out_path: Path) -> None:
    """Draw a horizontal flow-diagram and save to out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    BG      = "#0D1117"
    FG      = "#E6EDF3"
    FG2     = "#C9D1D9"
    ARROW   = "#484F58"

    n = len(blocks)
    TOTAL_W = 10.0
    FIG_H   = 3.8

    # Compute block widths proportionally
    weights  = [b.get("w", 1.0) for b in blocks]
    gaps     = 0.28 * (n - 1)
    margins  = 0.28
    avail    = TOTAL_W - gaps - 2 * margins
    tot_w    = sum(weights)
    bw       = [avail * wt / tot_w for wt in weights]

    fig, ax = plt.subplots(figsize=(TOTAL_W, FIG_H), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, TOTAL_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # Title
    ax.text(TOTAL_W / 2, FIG_H - 0.18, title,
            ha="center", va="top", fontsize=11, fontweight="bold",
            color=FG, fontfamily="monospace")

    BH      = 1.6     # block height
    BY      = (FIG_H - BH) / 2 - 0.15  # y-center of all blocks

    x_cursor = margins
    centers  = []

    for i, (blk, w) in enumerate(zip(blocks, bw)):
        cx  = x_cursor + w / 2
        cy  = BY + BH / 2
        centers.append((cx, cy))

        # Shadow
        shadow = FancyBboxPatch(
            (x_cursor + 0.03, BY - 0.04), w, BH,
            boxstyle="round,pad=0.06", linewidth=0,
            facecolor="#000000", alpha=0.35, zorder=1,
        )
        ax.add_patch(shadow)

        # Main block
        rect = FancyBboxPatch(
            (x_cursor, BY), w, BH,
            boxstyle="round,pad=0.06", linewidth=1.5,
            edgecolor=blk["color"], facecolor=blk["color"] + "28",
            zorder=2,
        )
        ax.add_patch(rect)

        # Top accent bar
        accent = FancyBboxPatch(
            (x_cursor, BY + BH - 0.12), w, 0.12,
            boxstyle="round,pad=0.00", linewidth=0,
            facecolor=blk["color"], alpha=0.75, zorder=3,
        )
        ax.add_patch(accent)

        # Label text
        label = blk["label"]
        ax.text(cx, cy, label,
                ha="center", va="center", fontsize=7.5,
                color=FG2, fontfamily="monospace",
                linespacing=1.4, zorder=4,
                multialignment="center")

        x_cursor += w + 0.28

    # Arrows between blocks
    for i in range(n - 1):
        x0 = centers[i][0]   + bw[i]   / 2
        x1 = centers[i+1][0] - bw[i+1] / 2
        mid_y = centers[i][1]
        ax.annotate(
            "", xy=(x1, mid_y), xytext=(x0, mid_y),
            arrowprops=dict(arrowstyle="-|>", color=ARROW,
                            lw=1.3, mutation_scale=14),
            zorder=5,
        )

    plt.tight_layout(pad=0.2)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SciML architecture visualizations")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    ap.add_argument("--force",   action="store_true",
                    help="Regenerate script-made PNGs (manifest entries only — never touches user images)")
    args = ap.parse_args()

    dry      = args.dry_run
    manifest = _load_manifest()   # filenames this script has previously written

    # ── Step 1: Rename existing files ─────────────────────────────────────────
    print("\n=== Step 1: Renaming existing files ===")
    for old_name, new_key in RENAME_MAP.items():
        old_path = ARCH_DIR / old_name
        new_path = ARCH_DIR / f"{new_key}.png"
        if not old_path.exists():
            print(f"  SKIP  {old_name}  (not found)")
            continue
        if old_path == new_path:
            print(f"  OK    {old_name}  (already canonical)")
            continue
        if new_path.exists() and not args.force:
            print(f"  SKIP  {old_name} → {new_key}.png  (target exists, use --force)")
            continue
        # Only allow overwrite of target if it was script-generated
        if new_path.exists() and args.force and new_path.name not in manifest:
            print(f"  PROTECT  {new_key}.png  (user-provided, not overwriting)")
            continue
        print(f"  RENAME  {old_name}  →  {new_key}.png")
        if not dry:
            shutil.move(str(old_path), str(new_path))
            manifest.discard(old_name)
            manifest.add(new_path.name)

    # ── Step 2: Create aliases (copy) ─────────────────────────────────────────
    print("\n=== Step 2: Creating alias copies ===")
    for alias_key, primary_key in ALIAS_MAP.items():
        src  = ARCH_DIR / f"{primary_key}.png"
        dest = ARCH_DIR / f"{alias_key}.png"
        if not src.exists():
            print(f"  SKIP  {alias_key} → {primary_key}  (source missing)")
            continue
        if dest.exists() and not args.force:
            print(f"  OK    {alias_key}.png  (alias exists)")
            continue
        # Only allow overwrite if script-generated
        if dest.exists() and args.force and dest.name not in manifest:
            print(f"  PROTECT  {alias_key}.png  (user-provided, not overwriting)")
            continue
        print(f"  COPY  {primary_key}.png  →  {alias_key}.png")
        if not dry:
            shutil.copy2(str(src), str(dest))
            manifest.add(dest.name)

    # ── Step 3: Generate missing diagrams ─────────────────────────────────────
    print("\n=== Step 3: Generating missing diagrams ===")
    for key, spec in SPECS.items():
        out_path = ARCH_DIR / f"{key}.png"
        if out_path.exists() and not args.force:
            print(f"  OK    {key}.png  (exists)")
            continue
        # With --force, only regenerate if we made it; skip user-provided images
        if out_path.exists() and args.force and out_path.name not in manifest:
            print(f"  PROTECT  {key}.png  (user-provided, not overwriting)")
            continue
        print(f"  GEN   {key}.png")
        if not dry:
            draw_arch(spec, key, out_path)
            manifest.add(out_path.name)

    # ── Persist manifest ──────────────────────────────────────────────────────
    _save_manifest(manifest, dry)

    # ── Summary ───────────────────────────────────────────────────────────────
    existing   = sorted(p.name for p in ARCH_DIR.glob("*.png"))
    user_imgs  = [n for n in existing if n not in manifest]
    gen_imgs   = [n for n in existing if n in manifest]
    print(f"\n=== Done — {len(existing)} PNGs in {ARCH_DIR.relative_to(REPO_ROOT)} ===")
    print(f"  {len(user_imgs)} user-provided (protected)   {len(gen_imgs)} script-generated (manifest)")
    for name in existing:
        tag = "  " if name in manifest else "★ "   # ★ = user-provided / protected
        print(f"  {tag}{name}")


if __name__ == "__main__":
    main()
