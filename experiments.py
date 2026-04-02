"""Declarative experiment registry for autonomous SciML research.

Every entry is an ExperimentConfig.  autorun.py works through this list in
priority order, skips experiments already recorded in results.tsv, and logs
each outcome.

To add a new experiment: append an ExperimentConfig to EXPERIMENTS.
Priorities: 1 = run first, 10 = run last.

Research directions (cross-reference with program.md):
  P1 – FNO width/depth/modes sweep     (quick wins around baseline)
  P2 – UNO architecture                (multi-scale FNO)
  P3 – WNO architecture                (wavelet operator)
  P4 – PINO physics loss               (data + PDE residual)
  P5 – LR / schedule tuning
  P6 – DeepONet variants
  P7 – Combined best-found configs
"""

from dataclasses import dataclass, field
from typing import List


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    name:        str              # unique key (used for dedup against results.tsv)
    benchmark:   str              # "burgers_1d" | "darcy_2d"
    model:       str              # "FNO" | "UNO" | "WNO" | "DeepONet" | "PODDeepONet"
    hidden_dim:  int              # channel width
    n_layers:    int              # depth (FNO blocks per level for UNO)
    n_modes:     int  = 16        # Fourier modes (FNO / UNO)
    n_levels:    int  = 3         # Haar levels (WNO)
    lr:          float = 1e-3     # learning rate
    batch_size:  int  = 32        # training batch size
    grad_clip:   float = 1.0      # gradient clipping (0 = disabled)
    pino_lambda: float = 0.0      # PINO physics-loss weight
    priority:    int  = 5         # 1 = highest; run in ascending order
    rationale:   str  = ""        # why this experiment?
    expected:    str  = ""        # expected val_l2_rel range or direction

    def to_cli_args(self) -> List[str]:
        return [
            "--benchmark",   self.benchmark,
            "--model",       self.model,
            "--hidden",      str(self.hidden_dim),
            "--layers",      str(self.n_layers),
            "--modes",       str(self.n_modes),
            "--levels",      str(self.n_levels),
            "--lr",          str(self.lr),
            "--batch_size",  str(self.batch_size),
            "--grad_clip",   str(self.grad_clip),
            "--pino_lambda", str(self.pino_lambda),
        ]

    def short(self) -> str:
        """One-line summary for logging."""
        parts = [f"{self.model}", f"h={self.hidden_dim}", f"l={self.n_layers}"]
        if self.model in ("FNO", "UNO"):
            parts.append(f"m={self.n_modes}")
        if self.model == "WNO":
            parts.append(f"lvl={self.n_levels}")
        if self.pino_lambda > 0:
            parts.append(f"pino={self.pino_lambda}")
        if self.lr != 1e-3:
            parts.append(f"lr={self.lr:.0e}")
        if self.grad_clip != 1.0:
            parts.append(f"clip={self.grad_clip}")
        return "  ".join(parts)


# ── Experiment queue ──────────────────────────────────────────────────────────

EXPERIMENTS: List[ExperimentConfig] = [

    # ── P1 · FNO width sweep ─────────────────────────────────────────────────
    ExperimentConfig(
        name="fno_h128_m16_l4",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=4, n_modes=16,
        priority=1,
        rationale="Doubling hidden width from 64→128; more expressive feature space.",
        expected="~0.18–0.20 (10–15% better than h=64 baseline 0.2307)",
    ),
    ExperimentConfig(
        name="fno_h256_m16_l4",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=256, n_layers=4, n_modes=16,
        priority=1,
        rationale="Maximum practical width on Apple Silicon for 5-min budget.",
        expected="May match or exceed h=128 if not step-time-limited.",
    ),
    ExperimentConfig(
        name="fno_h64_m16_l6",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=64, n_layers=6, n_modes=16,
        priority=1,
        rationale="Deeper model, same width; tests whether depth helps more than width.",
        expected="~0.19–0.21",
    ),
    ExperimentConfig(
        name="fno_h64_m16_l8",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=64, n_layers=8, n_modes=16,
        priority=2,
        rationale="Very deep; may suffer from vanishing gradients at this width.",
        expected="Unknown — useful data point regardless.",
    ),
    ExperimentConfig(
        name="fno_h128_m16_l6",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=16,
        priority=2,
        rationale="Wide AND deep — best of both if memory allows.",
        expected="~0.15–0.18 if 5-min budget is enough to converge.",
    ),
    ExperimentConfig(
        name="fno_h64_m24_l4",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=64, n_layers=4, n_modes=24,
        priority=2,
        rationale="More Fourier modes; captures more frequencies in Burgers shock.",
        expected="~0.20–0.22; modes alone rarely dominate over width.",
    ),
    ExperimentConfig(
        name="fno_h64_m32_l4",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=64, n_layers=4, n_modes=32,
        priority=2,
        rationale="Maximum modes (= GRID_SIZE//2); includes all spatial frequencies.",
        expected="~0.19–0.22",
    ),
    ExperimentConfig(
        name="fno_h128_m24_l6",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=24,
        priority=3,
        rationale="Combined wide+deep+more-modes if earlier sweeps confirm these help.",
        expected="~0.14–0.17 (targets SOTA ~0.01–0.05 range with good training)",
    ),

    # ── P2 · UNO architecture ────────────────────────────────────────────────
    ExperimentConfig(
        name="uno_h64_l1",
        benchmark="burgers_1d", model="UNO",
        hidden_dim=64, n_layers=1, n_modes=16,
        priority=1,
        rationale="Shallow UNO; fast per-step → more training iterations in 5 min.",
        expected="~0.18–0.22; U-Net structure may help even with 1 block/level.",
    ),
    ExperimentConfig(
        name="uno_h64_l2",
        benchmark="burgers_1d", model="UNO",
        hidden_dim=64, n_layers=2, n_modes=16,
        priority=1,
        rationale="Standard UNO depth; matches Rahman et al. (2022) paper config.",
        expected="~0.17–0.20; reference paper reports ~20% over flat FNO.",
    ),
    ExperimentConfig(
        name="uno_h128_l2",
        benchmark="burgers_1d", model="UNO",
        hidden_dim=128, n_layers=2, n_modes=16,
        priority=2,
        rationale="Wider UNO; bottleneck uses 4×128=512 channels.",
        expected="~0.15–0.18 if not memory/time constrained.",
    ),
    ExperimentConfig(
        name="uno_h64_l3",
        benchmark="burgers_1d", model="UNO",
        hidden_dim=64, n_layers=3, n_modes=16,
        priority=2,
        rationale="Deeper UNO; 9 total FNO blocks (3 per level).",
        expected="~0.17–0.20; may be slower per step.",
    ),
    ExperimentConfig(
        name="uno_h128_l1_m24",
        benchmark="burgers_1d", model="UNO",
        hidden_dim=128, n_layers=1, n_modes=24,
        priority=3,
        rationale="Wide UNO with more modes; tests if UNO benefits from extra freq info.",
        expected="~0.14–0.18",
    ),

    # ── P3 · WNO architecture ────────────────────────────────────────────────
    ExperimentConfig(
        name="wno_h64_lvl3_l4",
        benchmark="burgers_1d", model="WNO",
        hidden_dim=64, n_layers=4, n_levels=3,
        priority=1,
        rationale="WNO baseline; 3 Haar levels decomposes N=64 to 8-pt approx scale.",
        expected="~0.20–0.25; may match FNO for Burgers shock.",
    ),
    ExperimentConfig(
        name="wno_h128_lvl3_l4",
        benchmark="burgers_1d", model="WNO",
        hidden_dim=128, n_layers=4, n_levels=3,
        priority=2,
        rationale="Wider WNO.",
        expected="~0.17–0.22",
    ),
    ExperimentConfig(
        name="wno_h64_lvl4_l4",
        benchmark="burgers_1d", model="WNO",
        hidden_dim=64, n_layers=4, n_levels=4,
        priority=2,
        rationale="4 Haar levels → 4-pt approx scale; more multi-scale info.",
        expected="~0.18–0.23; risk of losing spatial resolution.",
    ),
    ExperimentConfig(
        name="wno_h64_lvl2_l6",
        benchmark="burgers_1d", model="WNO",
        hidden_dim=64, n_layers=6, n_levels=2,
        priority=3,
        rationale="Fewer decomp levels but deeper; focuses on fine-scale features.",
        expected="~0.20–0.25",
    ),

    # ── P4 · PINO physics loss ────────────────────────────────────────────────
    ExperimentConfig(
        name="fno_h128_pino1e3",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=4, n_modes=16,
        pino_lambda=1e-3,
        priority=2,
        rationale="PINO with very small lambda; physics regularises without dominating.",
        expected="Should improve over pure-data if λ is well-tuned.",
    ),
    ExperimentConfig(
        name="fno_h128_pino1e2",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=4, n_modes=16,
        pino_lambda=1e-2,
        priority=2,
        rationale="PINO λ=0.01; standard starting point in Li et al. PINO paper.",
        expected="~0.15–0.20 if residual loss is well-scaled.",
    ),
    ExperimentConfig(
        name="fno_h128_pino1e1",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=4, n_modes=16,
        pino_lambda=1e-1,
        priority=3,
        rationale="PINO λ=0.1; physics loss may start to dominate data loss.",
        expected="May worsen if physics loss gradient magnitudes are large.",
    ),
    ExperimentConfig(
        name="uno_h64_pino1e2",
        benchmark="burgers_1d", model="UNO",
        hidden_dim=64, n_layers=2, n_modes=16,
        pino_lambda=1e-2,
        priority=3,
        rationale="Combine UNO architecture with PINO loss.",
        expected="Potentially best result if both improvements stack.",
    ),
    ExperimentConfig(
        name="uno_h128_pino1e2",
        benchmark="burgers_1d", model="UNO",
        hidden_dim=128, n_layers=2, n_modes=16,
        pino_lambda=1e-2,
        priority=4,
        rationale="Wide UNO + PINO; tests whether physics loss helps a wide model.",
        expected="~0.13–0.17 (optimistic target)",
    ),

    # ── P5 · LR & schedule tuning ────────────────────────────────────────────
    ExperimentConfig(
        name="fno_h128_lr3e4",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=4, n_modes=16,
        lr=3e-4,
        priority=3,
        rationale="Lower LR; more stable but may not fully converge in 5 min.",
        expected="Useful if h=128 training is noisy at lr=1e-3.",
    ),
    ExperimentConfig(
        name="fno_h128_lr3e3",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=4, n_modes=16,
        lr=3e-3,
        priority=3,
        rationale="Higher LR; converges faster but needs stable gradients.",
        expected="Could be best if GRAD_CLIP=1.0 is sufficient to stabilise.",
    ),
    ExperimentConfig(
        name="fno_h64_no_clip",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=64, n_layers=4, n_modes=16,
        grad_clip=0.0,
        priority=4,
        rationale="Disable gradient clipping; baseline check if clip hurts speed.",
        expected="Similar to h=64 baseline; useful for calibrating GRAD_CLIP.",
    ),
    ExperimentConfig(
        name="fno_h128_tight_clip",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=4, n_modes=16,
        grad_clip=0.3,
        priority=4,
        rationale="Tighter gradient clipping; more conservative updates.",
        expected="May help if h=128 training is unstable.",
    ),

    # ── P6 · DeepONet variants ────────────────────────────────────────────────
    ExperimentConfig(
        name="deeponet_h128_l4",
        benchmark="burgers_1d", model="DeepONet",
        hidden_dim=128, n_layers=4,
        priority=3,
        rationale="Wider DeepONet; prior run (h=64 old arch) got 0.808.",
        expected="New LayerNorm DeepONet should get ~0.10–0.15.",
    ),
    ExperimentConfig(
        name="deeponet_h256_l4",
        benchmark="burgers_1d", model="DeepONet",
        hidden_dim=256, n_layers=4,
        priority=4,
        rationale="Very wide DeepONet branch/trunk.",
        expected="~0.08–0.12 if convergence is fast enough.",
    ),
    ExperimentConfig(
        name="pod_deeponet_h64",
        benchmark="burgers_1d", model="PODDeepONet",
        hidden_dim=64, n_layers=3,
        priority=4,
        rationale="POD basis + branch coefficient learning; low-rank approximation.",
        expected="~0.10–0.20 depending on how well the shared basis is learned.",
    ),
    ExperimentConfig(
        name="pod_deeponet_h128",
        benchmark="burgers_1d", model="PODDeepONet",
        hidden_dim=128, n_layers=3,
        priority=5,
        rationale="Wider POD-DeepONet.",
        expected="~0.08–0.15",
    ),

    # ── P7 · Darcy 2D sweep ──────────────────────────────────────────────────
    ExperimentConfig(
        name="darcy_fno_h64_m12_l4",
        benchmark="darcy_2d", model="FNO",
        hidden_dim=64, n_layers=4, n_modes=12,
        priority=5,
        rationale="FNO on Darcy; current baseline is 0.9986 (near random). "
                  "This tests if the solver produces learnable data.",
        expected="If >0.9, solver data is likely not meaningful.",
    ),
    ExperimentConfig(
        name="darcy_fno_h128_m12_l4",
        benchmark="darcy_2d", model="FNO",
        hidden_dim=128, n_layers=4, n_modes=12,
        priority=6,
        rationale="Wider FNO for Darcy; baseline check.",
        expected="Similar to above.",
    ),

    # ── P8 · Batch size sensitivity ──────────────────────────────────────────
    ExperimentConfig(
        name="fno_h128_bs16",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=4, n_modes=16,
        batch_size=16,
        priority=5,
        rationale="Smaller batch → more gradient steps in 5 min; noisier gradients "
                  "but better epoch coverage.",
        expected="May help convergence if bandwidth is the bottleneck.",
    ),
    ExperimentConfig(
        name="fno_h128_bs64",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=4, n_modes=16,
        batch_size=64,
        priority=5,
        rationale="Larger batch → smoother gradients, fewer steps per epoch.",
        expected="May hurt convergence in fixed-time budget.",
    ),

    # ── P9 · Follow-ups from session-1/2 findings ────────────────────────────
    # Key empirical result: FNO h=128, l=6 = 0.1852 (best so far).
    # Width (h=256) helped more than raw depth (l=6 at h=64).
    # h=128 + l=6 beat both: best combo in the budget.
    # WNO & PINO failed — both fixed; re-run with fixed code.

    ExperimentConfig(
        name="fno_h128_m16_l8",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=8, n_modes=16,
        priority=1,
        rationale="Extend best config (h=128,l=6→0.1852): does l=8 add more?",
        expected="~0.16–0.18; step time ~35ms so still ~8500 steps.",
    ),
    ExperimentConfig(
        name="fno_h256_m16_l6",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=256, n_layers=6, n_modes=16,
        priority=1,
        rationale="Wide AND deep: h=256 (2nd best) + l=6 (best depth).",
        expected="~0.16–0.18; slower per step (~45ms) but still meaningful.",
    ),
    ExperimentConfig(
        name="fno_h128_m24_l6",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=24,
        priority=1,
        rationale="Best depth config + more Fourier modes (24 vs 16).",
        expected="~0.17–0.19; more modes capture higher-freq Burgers features.",
    ),
    ExperimentConfig(
        name="fno_h128_m32_l6",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=32,
        priority=2,
        rationale="Max modes (N//2=32) at best depth.",
        expected="~0.17–0.19; all spatial frequencies included.",
    ),
    ExperimentConfig(
        name="fno_h128_l6_lr3e4",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=16,
        lr=3e-4,
        priority=2,
        rationale="Best arch (h=128,l=6) with lower LR: smoother late convergence.",
        expected="~0.17–0.19; may help in the warmdown phase.",
    ),
    ExperimentConfig(
        name="fno_h128_l6_lr3e3",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=16,
        lr=3e-3,
        priority=2,
        rationale="Best arch (h=128,l=6) with higher LR: faster early convergence.",
        expected="~0.16–0.18 if GRAD_CLIP=1.0 keeps it stable.",
    ),
    ExperimentConfig(
        name="fno_h128_l6_pino_fixed",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=16,
        pino_lambda=0.05,
        priority=2,
        rationale="PINO on best arch with FIXED normalised physics loss. "
                  "Previous runs (λ=0.01/0.001) failed because residual was "
                  "unnormalised; now fixed to relative-L2 scaling.",
        expected="~0.15–0.18 if physics loss genuinely guides training.",
    ),
    ExperimentConfig(
        name="fno_h128_l6_pino_small",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=16,
        pino_lambda=0.01,
        priority=2,
        rationale="PINO with fixed normalisation, λ=0.01 (conservative).",
        expected="~0.15–0.18.",
    ),
    ExperimentConfig(
        name="wno_h128_l6_fixed",
        benchmark="burgers_1d", model="WNO",
        hidden_dim=128, n_layers=6, n_levels=3,
        priority=2,
        rationale="WNO with FIXED Haar transform (reshape-based instead of "
                  "stride-2 slicing, which produced near-zero gradients). "
                  "Wider and deeper than original failed run.",
        expected="~0.20–0.25 if gradient fix resolves the learning failure.",
    ),
    ExperimentConfig(
        name="uno_h128_l2_wide",
        benchmark="burgers_1d", model="UNO",
        hidden_dim=128, n_layers=2, n_modes=16,
        priority=2,
        rationale="h=64 UNO lost to FNO (0.222 vs 0.185). "
                  "h=128 matches the winning FNO width — bottleneck has 512ch.",
        expected="~0.17–0.20; UNO paper claims ~20% over FNO at matched capacity.",
    ),
    ExperimentConfig(
        name="fno_h128_l6_bs16",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=16,
        batch_size=16,
        priority=3,
        rationale="Best arch + small batch → ~2× more gradient steps per minute.",
        expected="~0.16–0.18; higher gradient noise may help escape local optima.",
    ),

    # ── P10 · Mode fine-sweep around m=24 (session-3 finding) ────────────────
    # m=24 beat m=16 significantly (0.1648 vs 0.1852) and m=32 exploded (0.631).
    # Goal: find the exact mode sweet spot and combine with best depth/width.

    ExperimentConfig(
        name="fno_h128_m20_l6",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=20,
        priority=1,
        rationale="Mode sweep: m=20 between 16 (0.185) and 24 (0.165).",
        expected="~0.170–0.185; fine-tuning the mode cutoff.",
    ),
    ExperimentConfig(
        name="fno_h128_m22_l6",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=22,
        priority=1,
        rationale="Mode sweep: m=22.",
        expected="~0.165–0.180.",
    ),
    ExperimentConfig(
        name="fno_h128_m26_l6",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=26,
        priority=1,
        rationale="Mode sweep: m=26 (just above best m=24).",
        expected="~0.160–0.170 if m=24 was not yet the true peak.",
    ),
    ExperimentConfig(
        name="fno_h128_m28_l6",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=28,
        priority=1,
        rationale="Mode sweep: m=28 (approaching m=32 failure zone).",
        expected="~0.165–0.180; approaching aliasing instability.",
    ),
    ExperimentConfig(
        name="fno_h256_m24_l6",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=256, n_layers=6, n_modes=24,
        priority=1,
        rationale="Best modes (m=24) + widest model (h=256) + best depth (l=6). "
                  "h=256+m=16 got 0.185; m=24 boosted h=128 by 11% → may do same here.",
        expected="~0.145–0.165; most likely path to <0.15.",
    ),
    ExperimentConfig(
        name="fno_h128_m24_l8",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        priority=1,
        rationale="Best modes + deeper (l=8). l=8 alone gave 0.187; with m=24 may compound.",
        expected="~0.155–0.165.",
    ),
    ExperimentConfig(
        name="fno_h128_m24_bs16",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=24,
        batch_size=16,
        priority=2,
        rationale="Best config + smaller batch → ~2× more gradient steps in 5 min.",
        expected="~0.150–0.165; noise may help or hinder.",
    ),
    ExperimentConfig(
        name="fno_h128_m24_lr2e3",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=24,
        lr=2e-3,
        priority=2,
        rationale="Best config + moderately higher LR (2e-3 between 1e-3 and 3e-3).",
        expected="~0.155–0.170.",
    ),
    ExperimentConfig(
        name="fno_h256_m24_l8",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=256, n_layers=8, n_modes=24,
        priority=2,
        rationale="Maximum capacity: wide + deep + best modes. Step time ~60ms "
                  "so ~5000 steps; may be step-time-limited.",
        expected="~0.140–0.160 if not compute-limited.",
    ),
    ExperimentConfig(
        name="fno_h128_m24_l6_wd1e3",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=6, n_modes=24,
        priority=2,
        rationale="Best config + 10× higher weight decay (1e-3 vs default 1e-4). "
                  "More regularisation may reduce overfitting to training noise.",
        expected="~0.155–0.170.",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_experiments(
    benchmark: str | None = None,
    model: str | None = None,
    max_priority: int = 10,
) -> List[ExperimentConfig]:
    """Return filtered experiment list sorted by priority then name."""
    exps = EXPERIMENTS
    if benchmark:
        exps = [e for e in exps if e.benchmark == benchmark]
    if model:
        exps = [e for e in exps if e.model == model]
    exps = [e for e in exps if e.priority <= max_priority]
    return sorted(exps, key=lambda e: (e.priority, e.name))


if __name__ == "__main__":
    # Print all experiments in run order
    exps = get_experiments()
    print(f"Total experiments: {len(exps)}")
    print(f"Estimated wall-clock time: ~{len(exps) * 7 / 60:.1f} hours\n")
    print(f"{'#':>3}  {'Pri':>3}  {'Name':<30}  {'Config'}")
    print("-" * 90)
    for i, e in enumerate(exps, 1):
        print(f"{i:>3}  {e.priority:>3}  {e.name:<30}  {e.short()}")
