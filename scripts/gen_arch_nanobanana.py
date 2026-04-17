"""Generate architecture diagrams for SciML models using Nanobanana (Gemini 3.1 Flash Image).

Usage:
    uv run python scripts/gen_arch_nanobanana.py             # generate all missing
    uv run python scripts/gen_arch_nanobanana.py --force     # regenerate all (skips user-provided)
    uv run python scripts/gen_arch_nanobanana.py --model FNO2D AFNO  # specific models

Requires: infsh logged in with credits  (infsh login)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parent.parent
ARCH_DIR      = REPO_ROOT / "figs" / "arch"
MANIFEST_PATH = ARCH_DIR / ".generated_manifest.json"
ARCH_DIR.mkdir(parents=True, exist_ok=True)

# ── Prompts per model ─────────────────────────────────────────────────────────
BASE_STYLE = (
    "Dark navy background (#0D1117). "
    "Clean technical scientific architecture diagram. "
    "Teal/cyan labeled rectangular boxes, white text labels, gray arrows between stages. "
    "16:9 aspect ratio. No decorative elements, no people, no photorealism."
)

PROMPTS: dict[str, str] = {
    "FNO2D": (
        "Technical architecture diagram for FNO2D (2D Fourier Neural Operator). "
        "Horizontal left-to-right flow: [u(x,y) Input] → [Lifting Layer P: pointwise linear] → "
        "[FNO Block ×4 (each block: SpectralConv2D branch (2D FFT → complex weight matrix multiply → 2D IFFT) "
        "PLUS pointwise Linear skip branch → add → GELU)] → [Projection Layer Q] → [û(x,y) Output]. "
        + BASE_STYLE
    ),
    "AFNO": (
        "Technical architecture diagram for AFNO (Adaptive Fourier Neural Operator). "
        "Flow: [Input tokens (H×W×C)] → [2D FFT → frequency domain tokens] → "
        "[Adaptive Spectral Filter: learned per-frequency 2×2 weight blocks mixing channels in freq domain] → "
        "[2D IFFT → spatial domain] → [MLP with SoftShrink sparsity] → [Output tokens]. "
        "Inset showing frequency grid with highlighted low-freq modes. " + BASE_STYLE
    ),
    "FFNO": (
        "Technical architecture diagram for FFNO (Factorized Fourier Neural Operator). "
        "Flow: [u(x,y) Input] → [Lifting] → [Factorized FNO Block ×L: "
        "1D SpectralConv along x-axis PLUS 1D SpectralConv along y-axis → pointwise add → Linear skip → GELU] → "
        "[Projection] → [û Output]. "
        "Emphasize axis-factorized spectral convolution (x-pass then y-pass separately). " + BASE_STYLE
    ),
    "CPFNO": (
        "Technical architecture diagram for CPFNO (CP-decomposed Fourier Neural Operator). "
        "Flow: [Input] → [Lifting] → [CPFNO Block ×L: SpectralConv with CP-rank-r decomposed weight tensor "
        "W≈Σ a_r⊗b_r⊗c_r (sum of outer products, rank r) → Linear skip → GELU] → [Projection] → [Output]. "
        "Inset: CP decomposition diagram showing W as sum of r rank-1 tensors. " + BASE_STYLE
    ),
    "TFNO2D": (
        "Technical architecture diagram for TFNO2D (Tucker-decomposed Tensor FNO 2D). "
        "Flow: [u(x,y) Input] → [Lifting P] → [TFNO2D Block ×L: SpectralConv2D with Tucker-factorized "
        "complex weight tensor (core tensor G + factor matrices U1, U2, U3) → Linear skip → GELU] → "
        "[Projection Q] → [û(x,y) Output]. "
        "Inset: Tucker decomposition tensor diagram G×U1×U2×U3. " + BASE_STYLE
    ),
    "RTFNO": (
        "Technical architecture diagram for RTFNO (Residual Tensor Fourier Neural Operator). "
        "Flow: [Input] → [Lifting] → [RTFNO Block ×L: SpectralConv2D + "
        "long residual skip from block input → pointwise Linear → LayerNorm → GELU] → [Projection] → [Output]. "
        "Show orange residual/skip connection arcs bypassing each block. " + BASE_STYLE
    ),
    "NeuralODE": (
        "Technical architecture diagram for Neural ODE. "
        "LEFT panel: [z(t0) initial state] → [ODE Solver (Dopri5/RK4) integrating dz/dt = f_θ(z,t)] → [z(t1) output], "
        "with f_θ shown as a neural network MLP inside the solver. "
        "RIGHT panel: comparison trajectory plot showing ResNet discrete steps vs Neural ODE continuous curve in latent space. "
        "Teal/blue color scheme. " + BASE_STYLE
    ),
    "LatentODE": (
        "Technical architecture diagram for Latent Neural ODE (VAE + Neural ODE). "
        "Three stages: "
        "ENCODE: [Observations x(t1..tn)] → [RNN or Recognition ODE → μ, σ] → [sample z(t0) ~ N(μ,σ)]. "
        "DYNAMICS: [z(t0)] → [Neural ODE: dz/dt = f_θ(z,t), ODE solver] → [z(t1),...,z(tN)]. "
        "DECODE: [z(ti)] → [MLP Decoder → x̂(ti)]. "
        "Show smooth latent trajectory curve. Teal encoder, purple ODE dynamics, blue decoder. " + BASE_STYLE
    ),
    "UDE": (
        "Technical architecture diagram for UDE (Universal Differential Equation). "
        "Core diagram: [x(t0) initial condition] → "
        "[Hybrid ODE Solver: dx/dt = f_known(x,t) + U_θ(x,t) where f_known is known physics (shown in teal) "
        "and U_θ is neural network correction (shown in orange)] → [x(t) trajectory output]. "
        "Show f_known as a simple equation block and U_θ as an MLP. "
        "Emphasize the hybrid physics+ML structure with two additive terms. " + BASE_STYLE
    ),
    "PODDeepONet": (
        "Technical architecture diagram for PODDeepONet (Proper Orthogonal Decomposition DeepONet). "
        "Two-branch architecture: "
        "BRANCH (left): [u(x) sensor readings] → [Branch MLP → modal coefficients c_1,...,c_r]. "
        "POD BASIS (right): [Precomputed POD modes φ_1(y),...,φ_r(y) from SVD of training snapshots]. "
        "OUTPUT: [Σ c_k · φ_k(y) → û(y)]. "
        "Show wavy sinusoidal POD mode shapes in gold. Teal branch, gold basis. " + BASE_STYLE
    ),
    "PINO": (
        "Technical architecture diagram for PINO (Physics-Informed Neural Operator). "
        "Flow: [u(x) Input] → [FNO Backbone: Lifting → SpectralConv blocks → Projection → û(x)]. "
        "Two loss streams diverging from û: "
        "(1) DATA LOSS: ||û - u_data||² (teal arrow). "
        "(2) PDE RESIDUAL LOSS: ||L[û] - f||² computed via automatic differentiation (red dashed arrow). "
        "Show ∂/∂x operator and PDE constraint box in red. Gradient flow arrows back through both losses. " + BASE_STYLE
    ),
    "GNOT2D": (
        "Technical architecture diagram for GNOT2D (General Neural Operator Transformer, 2D). "
        "Flow: [2D irregular mesh input (x,y,f)] → [Linear node embedding] → "
        "[L× Heterogeneous Cross-Attention blocks: query tokens from output query points, "
        "key+value tokens from input mesh nodes, multi-head attention with geometric encoding] → "
        "[MLP decoder per query point] → [û(x,y)]. "
        "Show 2D mesh as dots with attention edges radiating from query point. Teal/amber. " + BASE_STYLE
    ),
    "Transolver2D": (
        "Technical architecture diagram for Transolver2D (Transformer PDE solver, 2D). "
        "Flow: [2D PDE field u(x,y)] → [Physics-aware Tokenizer: slice field into P physics-tokens "
        "each capturing local structure] → [L× Transformer blocks with Physics-Cross-Attention: "
        "queries from output coords, keys/values from physics tokens] → [Token aggregation MLP] → [û(x,y)]. "
        "Show tokenization slicing diagram. Teal/amber. " + BASE_STYLE
    ),
    "HybridDecoderDeepONet2D": (
        "Technical architecture diagram for HybridDecoderDeepONet2D (FNO encoder + DeepONet decoder, 2D). "
        "Two-stage: "
        "STAGE 1 (left, teal): [u(x,y) Input] → [FNO2D Encoder: Lifting → 4× SpectralConv2D → latent h(x,y)]. "
        "STAGE 2 (right, purple): [h(x,y) features → Branch MLP] + [query coords (y) → Trunk MLP] → "
        "[dot product ⊙ → û(y)]. "
        "Show handoff arrow between FNO encoder and DeepONet decoder. " + BASE_STYLE
    ),
    "HybridFNODeepONet2D": (
        "Technical architecture diagram for HybridFNODeepONet2D (parallel FNO + DeepONet, 2D). "
        "Parallel two-path architecture: "
        "PATH A (top, teal): [u(x,y)] → [FNO2D: 4× SpectralConv2D blocks → spectral features f_FNO]. "
        "PATH B (bottom, purple): [u sensor values → Branch MLP] + [coords y → Trunk MLP] → [DeepONet output f_DON]. "
        "FUSION: [f_FNO + f_DON → Learned linear fusion → û(x,y)]. " + BASE_STYLE
    ),
    "DualDeepONet": (
        "Technical architecture diagram for DualDeepONet (two-branch DeepONet). "
        "Three-input architecture: "
        "BRANCH 1 (teal): [u(x) primary input function sensors] → [Deep MLP b1 → embedding e1]. "
        "BRANCH 2 (blue): [v(x) auxiliary/physics features] → [Deep MLP b2 → embedding e2]. "
        "TRUNK (purple): [query coords y] → [Deep MLP t → embedding t]. "
        "OUTPUT: [(e1 + e2) ⊙ t + bias → û(y)]. " + BASE_STYLE
    ),
    "EnergyFNO": (
        "Technical architecture diagram for EnergyFNO (energy-conserving FNO). "
        "Flow: [u(x,t) Input] → [Lifting] → [EnergyFNO Blocks ×L: SpectralConv + Linear skip + GELU, "
        "with energy normalization: each block output normalized so total energy E=∫|û|²dx is conserved] → "
        "[Projection] → [û(x,t)]. "
        "Show energy monitor inset: E(t) flat line indicating conservation. Green energy path. " + BASE_STYLE
    ),
    "FNO_MC": (
        "Technical architecture diagram for FNO_MC (FNO with Monte Carlo Dropout uncertainty). "
        "Flow: [u(x) Input] → [Lifting] → [FNO Blocks with MC Dropout layers (p=0.1, active at inference)] → "
        "[Projection] → [N=50 stochastic forward passes → ensemble outputs û_1,...,û_N] → "
        "[Mean μ(x) ± Std σ(x) with uncertainty band]. "
        "Show N parallel stochastic paths as light lines converging to mean with orange uncertainty shading. " + BASE_STYLE
    ),
    "GNOT_FFNO": (
        "Technical architecture diagram for GNOT_FFNO (GNOT + Factorized FNO hybrid). "
        "Parallel fusion: "
        "TOP PATH (teal): [Input field] → [FFNO: factorized x-axis + y-axis spectral convolution blocks → spectral features]. "
        "BOTTOM PATH (amber): [Input mesh nodes] → [GNOT cross-attention blocks → graph/attention features]. "
        "FUSION: [Spectral features ⊕ Graph features → MLP projection → û output]. " + BASE_STYLE
    ),
    "WNO_GNOT": (
        "Technical architecture diagram for WNO_GNOT (Wavelet Neural Operator + GNOT hybrid). "
        "Parallel two-path: "
        "LEFT PATH (green): [Input] → [WNO: multi-level Haar wavelet decomp → "
        "learned transforms on wavelet coefficients → inverse wavelet → wavelet features]. "
        "RIGHT PATH (amber): [Input mesh] → [GNOT: cross-attention on irregular grid → graph features]. "
        "MERGE: [Wavelet features ⊕ Graph features → MLP] → [û]. " + BASE_STYLE
    ),
    "TimeDeepONet": (
        "Technical architecture diagram for TimeDeepONet (time-conditioned DeepONet). "
        "Modified DeepONet with temporal conditioning: "
        "BRANCH (teal): [u(x, t₀) initial condition sensors] → [Branch MLP → b(u)]. "
        "TRUNK (purple+orange): [spatial coords x concatenated with time t → "
        "Time-conditioned Trunk MLP with sinusoidal temporal encoding → t(x,t)]. "
        "OUTPUT: [b ⊙ t → û(x,t)]. "
        "Show time axis as orange running thread through trunk. " + BASE_STYLE
    ),
    "KAN_FNO": (
        "Technical architecture diagram for KAN_FNO (Kolmogorov-Arnold Network + FNO). "
        "Flow: [u(x) Input] → [Lifting] → [KAN-FNO Block ×L: "
        "SpectralConv branch (FFT→W(k)→IFFT) PLUS KAN skip branch (learnable B-spline edge functions φ_ij(x) "
        "instead of fixed activations) → add] → [Projection] → [û(x)]. "
        "Inset showing KAN layer: nodes connected by curved B-spline activation φ(x) on each edge in gold. " + BASE_STYLE
    ),
    "ModifiedKAN_FNO": (
        "Technical architecture diagram for ModifiedKAN_FNO (Modified KAN Fourier Neural Operator). "
        "Flow: [u(x) Input] → [Lifting] → [ModKAN-FNO Block ×L: SpectralConv + "
        "Modified KAN layer (simplified B-spline grid with adaptive knot placement, fewer parameters) + LayerNorm] → "
        "[Projection] → [û(x)]. "
        "Inset: adaptive KAN layer showing sparse knot positions adapting to data distribution in gold. " + BASE_STYLE
    ),
    "cPIKAN_FNO": (
        "Technical architecture diagram for cPIKAN_FNO (continuous Physics-Informed KAN FNO). "
        "Flow: [u(x) Input] → [Lifting] → [cPIKAN Blocks ×L: SpectralConv + KAN skip (B-spline activations)] → "
        "[Projection → û(x)]. "
        "Two loss streams: DATA LOSS ||û-u_true||² (teal) AND PDE LOSS ||L[û]-f||² "
        "(red dashed, computed via automatic differentiation through the continuous KAN). "
        "Show ∂/∂x and PDE constraint in red. Gold KAN edges, red physics path. " + BASE_STYLE
    ),
    "AttentionEnhancedFNO2D": (
        "Technical architecture diagram for AttentionEnhancedFNO2D (FNO2D with axial attention). "
        "Flow: [u(x,y) Input] → [Lifting + Grid Embedding] → "
        "[L× Enhanced Blocks: three parallel paths inside each block: "
        "(1, teal) SpectralConv2D, "
        "(2, amber) Row-wise 1D self-attention across x, "
        "(3, orange) Column-wise 1D self-attention across y → "
        "sum all three → MLP + GELU] → [Projection] → [û(x,y)]. "
        "Show the 3-way split and merge clearly inside one block. " + BASE_STYLE
    ),
    "HANO2D": (
        "Technical architecture diagram for HANO2D (Hierarchical Attention Neural Operator 2D). "
        "Flow: [u(x,y) Input] → [Lifting + Grid Embedding] → "
        "[L× HANO Blocks: SpectralConv2D (global low-freq path, teal) + "
        "Axial Attention (row self-attention then column self-attention, purple) + "
        "Linear skip → GELU] → [Projection] → [û(x,y)]. "
        "Show hierarchical multi-scale: coarse spectral features at top, fine local attention below, merging. " + BASE_STYLE
    ),
    "RFNO2D": (
        "Technical architecture diagram for RFNO2D (Real-valued FNO 2D). "
        "Flow: [u(x,y) Input] → [Lifting P] → [RFNO2D Block ×L: "
        "real-valued SpectralConv2D using rfft2 (exploiting Hermitian symmetry, half the frequencies) → "
        "real weights on real/imag parts separately → irfft2 → Linear skip → GELU] → "
        "[Projection Q] → [û(x,y)]. "
        "Highlight the rfft2 Hermitian symmetry saving with a frequency-domain diagram. " + BASE_STYLE
    ),
}


def _load_manifest() -> set[str]:
    if MANIFEST_PATH.exists():
        try:
            return set(json.loads(MANIFEST_PATH.read_text()))
        except Exception:
            pass
    return set()


def _save_manifest(manifest: set[str]) -> None:
    MANIFEST_PATH.write_text(json.dumps(sorted(manifest), indent=2))


def generate_image(model_key: str, prompt: str, out_path: Path) -> bool:
    """Run infsh and save the result PNG to out_path. Returns True on success."""
    payload = json.dumps({
        "prompt": prompt,
        "aspect_ratio": "16:9",
        "resolution": "2K",
    })

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        result_file = Path(tf.name)

    try:
        result = subprocess.run(
            ["infsh", "app", "run", "google/gemini-3-1-flash-image-preview",
             "--input", payload, "--save", str(result_file)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  ERROR  {model_key}: {result.stderr.strip()}")
            return False

        data = json.loads(result_file.read_text())

        # infsh saves images as base64 or file paths in output.images
        images = data.get("output", {}).get("images", []) or data.get("images", [])
        if not images:
            print(f"  ERROR  {model_key}: no images in response. Keys: {list(data.keys())}")
            return False

        img_data = images[0]
        # Handle base64 or URL
        if isinstance(img_data, str) and img_data.startswith("data:image"):
            import base64
            header, b64 = img_data.split(",", 1)
            out_path.write_bytes(base64.b64decode(b64))
        elif isinstance(img_data, str) and (img_data.startswith("http") or img_data.startswith("/")):
            # File path written by infsh
            shutil.copy2(img_data, out_path)
        elif isinstance(img_data, dict) and "url" in img_data:
            import urllib.request
            urllib.request.urlretrieve(img_data["url"], str(out_path))
        else:
            print(f"  ERROR  {model_key}: unrecognized image format: {str(img_data)[:80]}")
            return False

        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT  {model_key}")
        return False
    except Exception as e:
        print(f"  ERROR  {model_key}: {e}")
        return False
    finally:
        result_file.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force",  action="store_true",
                    help="Regenerate even if PNG exists (only manifest entries)")
    ap.add_argument("--model", nargs="+", metavar="KEY",
                    help="Generate only these models (default: all missing)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest    = _load_manifest()
    target_keys = args.model if args.model else list(PROMPTS.keys())

    ok_count   = 0
    skip_count = 0
    err_count  = 0

    for key in target_keys:
        if key not in PROMPTS:
            print(f"  UNKNOWN  {key} — no prompt defined, skipping")
            continue

        out_path = ARCH_DIR / f"{key}.png"

        if out_path.exists() and not args.force:
            print(f"  OK    {key}.png  (exists)")
            skip_count += 1
            continue

        # Safety: never overwrite user-provided images
        if out_path.exists() and args.force and out_path.name not in manifest:
            print(f"  PROTECT  {key}.png  (user-provided, skipping)")
            skip_count += 1
            continue

        print(f"  GEN   {key}.png ...", end="", flush=True)
        if args.dry_run:
            print(" [dry-run]")
            continue

        success = generate_image(key, PROMPTS[key], out_path)
        if success:
            manifest.add(out_path.name)
            _save_manifest(manifest)
            size_kb = out_path.stat().st_size // 1024
            print(f" done ({size_kb} KB)")
            ok_count += 1
            time.sleep(0.5)   # small pause between requests
        else:
            err_count += 1

    print(f"\nDone — generated: {ok_count}  skipped: {skip_count}  errors: {err_count}")


if __name__ == "__main__":
    main()
