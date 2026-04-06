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
    benchmark:   str              # "burgers_1d" | "darcy_2d" | "kdv_1d" | "wave_1d"
    model:       str              # "FNO" | "RFNO" | "AFNO" | "FFNO" | "UNO" | "WNO" | "DeepONet" | "PODDeepONet"
    hidden_dim:  int              # channel width
    n_layers:    int              # depth (FNO blocks per level for UNO)
    n_modes:     int  = 16        # Fourier modes (FNO / UNO / RFNO / AFNO)
    n_levels:    int  = 3         # Haar levels (WNO)
    lr:          float = 1e-3     # learning rate
    batch_size:  int  = 32        # training batch size
    grad_clip:   float = 1.0      # gradient clipping (0 = disabled)
    pino_lambda: float = 0.0      # PINO physics-loss weight
    loss_type:   str  = "l2_rel"  # loss function: "l2_rel" | "h1" | "h1_strong" | "spectral"
    h1_alpha:    float = 0.1      # H1 loss derivative weight (used when loss_type="h1")
    augment:     bool  = False    # spatial-shift augmentation (periodic BCs only)
    save_ckpt:   bool  = False    # save model checkpoint after training
    budget_s:    int  = 300       # training time budget in seconds (default 5 min)
    parent_name: str  = ""        # name of parent experiment this branches from (for DAG lineage)
    priority:    int  = 5         # 1 = highest; run in ascending order
    rationale:   str  = ""        # why this experiment?
    expected:    str  = ""        # expected val_l2_rel range or direction
    paper_ref:   str  = ""        # paper ID from papers/*.yaml

    def to_cli_args(self) -> List[str]:
        args = [
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
            "--loss",        self.loss_type,
        ]
        if self.loss_type.startswith("h1"):
            args += ["--h1_alpha", str(self.h1_alpha)]
        if self.augment:
            args += ["--augment"]
        if self.save_ckpt:
            args += ["--save_ckpt"]
        if self.budget_s != 300:
            args += ["--budget", str(self.budget_s)]
        return args

    def short(self) -> str:
        """One-line summary for logging."""
        parts = [f"{self.model}", f"h={self.hidden_dim}", f"l={self.n_layers}"]
        if self.model in ("FNO", "RFNO", "AFNO", "FFNO", "UNO"):
            parts.append(f"m={self.n_modes}")
        if self.model == "WNO":
            parts.append(f"lvl={self.n_levels}")
        if self.pino_lambda > 0:
            parts.append(f"pino={self.pino_lambda}")
        if self.lr != 1e-3:
            parts.append(f"lr={self.lr:.0e}")
        if self.grad_clip != 1.0:
            parts.append(f"clip={self.grad_clip}")
        if self.loss_type != "l2_rel":
            parts.append(f"loss={self.loss_type}")
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
        name="fno_h128_m24_l6_v2",
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

    # ── P12 · Residual FNO (RFNO) — Pre-LN residual blocks ───────────────────
    # FNO saturates at l=8 (step-time-limited). Pre-LN residual connections
    # should allow deeper stacks without gradient degradation — same param count.
    ExperimentConfig(
        name="rfno_h128_m24_l8",
        benchmark="burgers_1d", model="RFNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        priority=1,
        rationale="RFNO at same config as FNO best (h=128,m=24,l=8). "
                  "Tests whether Pre-LN residuals alone improve over FNO 0.1553.",
        expected="~0.140–0.155 if normalisation helps; may match FNO.",
    ),
    ExperimentConfig(
        name="rfno_h128_m24_l10",
        benchmark="burgers_1d", model="RFNO",
        hidden_dim=128, n_layers=10, n_modes=24,
        priority=1,
        rationale="RFNO at l=10 — FNO degraded to 0.169 here due to fewer steps. "
                  "Residuals may allow l=10 to actually converge better.",
        expected="~0.135–0.150 if residuals unlock deeper models.",
    ),
    ExperimentConfig(
        name="rfno_h128_m24_l12",
        benchmark="burgers_1d", model="RFNO",
        hidden_dim=128, n_layers=12, n_modes=24,
        priority=1,
        rationale="RFNO at l=12 — FNO collapsed to 0.217. Residuals are the fix.",
        expected="~0.130–0.150 if depth scaling is unlocked.",
    ),
    ExperimentConfig(
        name="rfno_h128_m24_l6",
        benchmark="burgers_1d", model="RFNO",
        hidden_dim=128, n_layers=6, n_modes=24,
        priority=2,
        rationale="RFNO at shallower l=6 — ablation to isolate the Pre-LN effect "
                  "vs depth. FNO l=6 got 0.1648.",
        expected="~0.150–0.165.",
    ),
    ExperimentConfig(
        name="rfno_h128_m24_l16",
        benchmark="burgers_1d", model="RFNO",
        hidden_dim=128, n_layers=16, n_modes=24,
        priority=2,
        rationale="Push RFNO to l=16. Each block ~35ms → ~8500 steps in 5min "
                  "but only ~5300 iterations. Residuals may compensate.",
        expected="~0.125–0.145 if depth keeps scaling.",
    ),

    # ── P13 · AFNO (Adaptive FNO) — non-linear Fourier mixing ───────────────
    # Block-diagonal MLP + softshrink in spectral space.  Guibas et al. 2022.
    ExperimentConfig(
        name="afno_h128_m24_l8",
        benchmark="burgers_1d", model="AFNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        priority=1,
        rationale="AFNO at same config as FNO best. Non-linear Fourier mixing "
                  "+ softshrink sparsity. Hypothesis: MLP in Fourier space captures "
                  "more complex shock frequency interactions than linear maps.",
        expected="~0.13–0.15",
        paper_ref="afno-2022",
    ),
    ExperimentConfig(
        name="afno_h128_m24_l10",
        benchmark="burgers_1d", model="AFNO",
        hidden_dim=128, n_layers=10, n_modes=24,
        priority=1,
        rationale="AFNO with Pre-LN residuals should scale to l=10 (FNO degraded here). "
                  "Non-linear mixing + depth should compound benefits.",
        expected="~0.12–0.14",
        paper_ref="afno-2022",
    ),
    ExperimentConfig(
        name="afno_h128_m24_l12",
        benchmark="burgers_1d", model="AFNO",
        hidden_dim=128, n_layers=12, n_modes=24,
        priority=2,
        rationale="AFNO deep: l=12 with Pre-LN residuals. FNO collapsed at l=12 (0.217).",
        expected="~0.12–0.14",
        paper_ref="afno-2022",
    ),
    ExperimentConfig(
        name="afno_h128_m24_l8_sp0",
        benchmark="burgers_1d", model="AFNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        priority=2,
        rationale="AFNO without softshrink (sparsity=0) — ablation to isolate MLP "
                  "contribution from sparsity regularisation.",
        expected="~0.13–0.16 — isolates the MLP benefit",
        paper_ref="afno-2022",
    ),

    # ── P14 · H1 / Sobolev loss — frequency-weighted error ───────────────────
    # U-FNO (Wen et al. 2022) reports 10% improvement with H1 loss on flow problems.
    # H1 penalises gradient errors → directly targets Burgers shock fronts.
    ExperimentConfig(
        name="fno_h128_m24_l8_h1",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        loss_type="h1", h1_alpha=0.1,
        priority=1,
        rationale="H1 Sobolev loss on best FNO config. Shock fronts have large |∂u/∂x|; "
                  "H1 loss directly penalises gradient errors. α=0.1 is U-FNO default.",
        expected="~0.13–0.15 if H1 targets shock well",
        paper_ref="h1-sobolev-loss",
    ),
    ExperimentConfig(
        name="fno_h128_m24_l8_h1_a001",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        loss_type="h1", h1_alpha=0.01,
        priority=2,
        rationale="H1 with mild α=0.01 — avoids over-regularising smooth background.",
        expected="~0.14–0.16",
        paper_ref="h1-sobolev-loss",
    ),
    ExperimentConfig(
        name="rfno_h128_m24_l8_h1",
        benchmark="burgers_1d", model="RFNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        loss_type="h1", h1_alpha=0.1,
        priority=2,
        rationale="RFNO (Pre-LN residual) + H1 loss. Both independently may help; "
                  "combination might compound the gains.",
        expected="~0.12–0.14",
        paper_ref="h1-sobolev-loss",
    ),
    ExperimentConfig(
        name="afno_h128_m24_l8_h1",
        benchmark="burgers_1d", model="AFNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        loss_type="h1", h1_alpha=0.1,
        priority=2,
        rationale="AFNO + H1 loss: non-linear Fourier mixing + frequency-weighted "
                  "error. Best combination of architecture and loss innovations.",
        expected="~0.11–0.13",
        paper_ref="h1-sobolev-loss",
    ),

    # ── P15 · KdV benchmark (extended, uses ETDRK4 solver) ───────────────────
    ExperimentConfig(
        name="fno_kdv_h128_m24_l8",
        benchmark="kdv_1d", model="FNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        priority=2,
        rationale="Establish FNO baseline on KdV. Soliton dynamics are quasi-periodic "
                  "so FNO's Fourier basis is appropriate. SOTA: ~0.01.",
        expected="~0.02–0.08 (FNO good match for periodic KdV)",
        paper_ref="ffno-2023",
    ),
    ExperimentConfig(
        name="rfno_kdv_h128_m24_l8",
        benchmark="kdv_1d", model="RFNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        priority=3,
        rationale="RFNO on KdV — Pre-LN residuals may help capture multi-soliton dynamics.",
        expected="~0.01–0.05",
        paper_ref="rfno-2024",
    ),

    # ── P16 · Wave equation benchmark ────────────────────────────────────────
    ExperimentConfig(
        name="fno_wave_h64_m24_l4",
        benchmark="wave_1d", model="FNO",
        hidden_dim=64, n_layers=4, n_modes=24,
        priority=3,
        rationale="Establish FNO baseline on linear wave equation. "
                  "This is easy for FNO (linear PDE, periodic). SOTA: ~0.005.",
        expected="~0.005–0.02 (FNO near-exact for linear PDEs)",
    ),

    # ── P11 · Depth extension around fno_h128_m24_l8 (new best 0.1553) ────────
    # Key finding: depth l=8 beats width h=256; next question is how far depth scales.
    # Also probe whether the optimal mode count shifts at greater depth.
    ExperimentConfig(
        name="fno_h128_m24_l10",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=10, n_modes=24,
        priority=1,
        rationale="Depth trend: l=4(0.208)→l=6(0.165)→l=8(0.155). Does l=10 continue?",
        expected="~0.140–0.155 if depth trend continues.",
    ),
    ExperimentConfig(
        name="fno_h128_m24_l12",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=12, n_modes=24,
        priority=1,
        rationale="Push depth further. Each FNOBlock is ~40ms; l=12 ~4800 steps in 5 min.",
        expected="~0.135–0.155 if depth keeps helping; may saturate.",
    ),
    ExperimentConfig(
        name="fno_h128_m22_l8",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=8, n_modes=22,
        priority=1,
        rationale="m=22 slightly beat m=24 at l=6 (0.1643 vs 0.1648). Test if that "
                  "advantage persists or disappears at l=8.",
        expected="~0.150–0.158.",
    ),
    ExperimentConfig(
        name="fno_h128_m24_l8_bs16",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        batch_size=16,
        priority=1,
        rationale="Best depth config + half batch → ~2× gradient steps. "
                  "bs=16 gave 0.160 at l=6; may help more at l=8.",
        expected="~0.145–0.158.",
    ),
    ExperimentConfig(
        name="fno_h128_m24_l10_lr5e4",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=10, n_modes=24,
        lr=5e-4,
        priority=2,
        rationale="Deeper models often need lower LR to converge. "
                  "Pair l=10 with lr=5e-4 for more stable optimisation.",
        expected="~0.135–0.150.",
    ),
    ExperimentConfig(
        name="fno_h128_m24_l8_lr5e4",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        lr=5e-4,
        priority=2,
        rationale="Best config + lower LR. lr=1e-3 may be slightly too aggressive "
                  "for 5-min budget — 5e-4 gives more careful convergence.",
        expected="~0.148–0.158.",
    ),
    ExperimentConfig(
        name="fno_h128_m24_l12_bs16",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=12, n_modes=24,
        batch_size=16,
        priority=2,
        rationale="l=12 + bs=16 trades step size for step count — may help very deep model "
                  "escape early plateau.",
        expected="~0.130–0.150.",
    ),

    # ── P17 · AFNO v2 (fixed) + FFNO (factorized diagonal) ──────────────────
    # Previous AFNO results (0.56–0.72) were caused by a critical bug: real and
    # imaginary Fourier components were processed INDEPENDENTLY by the MLP.
    # The fix (v2): concatenate [xr; xi] → 2C → MLP(2C) → split.
    # This respects the complex convolution requirement: out_r must depend on
    # both xr and xi (analogous to FNO: out_r = xr*wr - xi*wi).
    #
    # FFNO uses diagonal spectral weights W[m,C] instead of full W[m,in,out].
    # 128× fewer spectral params → can afford h=256/512 or m=48 in same budget.
    ExperimentConfig(
        name="afno_v2_h128_m24_l8",
        benchmark="burgers_1d", model="AFNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        priority=1,
        rationale="AFNO v2 (fixed complex coupling) at FNO best config. Previous "
                  "AFNO failed (0.56–0.72) due to independent real/imag MLP — now "
                  "fixed by concatenating [xr;xi] → 2C input.",
        expected="~0.12–0.15 (should now match or beat FNO)",
        paper_ref="afno-2022",
    ),
    ExperimentConfig(
        name="afno_v2_h128_m24_l10",
        benchmark="burgers_1d", model="AFNO",
        hidden_dim=128, n_layers=10, n_modes=24,
        priority=1,
        rationale="AFNO v2 at l=10. Pre-LN residuals in AFNOBlock should unlock "
                  "depth that FNO cannot reach (FNO degraded at l=10: 0.169).",
        expected="~0.11–0.14",
        paper_ref="afno-2022",
    ),
    ExperimentConfig(
        name="ffno_h128_m32_l8",
        benchmark="burgers_1d", model="FFNO",
        hidden_dim=128, n_layers=8, n_modes=32,
        priority=1,
        rationale="FFNO: diagonal per-mode-per-channel weights → 128× cheaper "
                  "spectral params than FNO. Enables m=32 (max modes) with same "
                  "param budget. Key hypothesis: more modes + diagonal = better "
                  "frequency coverage without over-parameterisation.",
        expected="~0.13–0.15",
        paper_ref="ffno-2023",
    ),
    ExperimentConfig(
        name="ffno_h256_m32_l8",
        benchmark="burgers_1d", model="FFNO",
        hidden_dim=256, n_layers=8, n_modes=32,
        priority=1,
        rationale="FFNO with wide channels (h=256). FNO h=256 was step-time-limited "
                  "(slow spectral conv); FFNO's diagonal weights are much cheaper so "
                  "h=256 is feasible. Full channel mixing comes from pointwise Linear.",
        expected="~0.11–0.14 — wider FFNO should beat narrow FNO",
        paper_ref="ffno-2023",
    ),
    ExperimentConfig(
        name="ffno_h256_m24_l8",
        benchmark="burgers_1d", model="FFNO",
        hidden_dim=256, n_layers=8, n_modes=24,
        priority=1,
        rationale="FFNO h=256 at FNO optimal m=24. Combines width advantage of "
                  "diagonal weights with proven mode count.",
        expected="~0.12–0.14",
        paper_ref="ffno-2023",
    ),
    ExperimentConfig(
        name="ffno_h512_m24_l8",
        benchmark="burgers_1d", model="FFNO",
        hidden_dim=512, n_layers=8, n_modes=24,
        priority=2,
        rationale="FFNO at h=512 — only feasible because diagonal spectral weights "
                  "add O(m*C) not O(m*C²) params. Tests extreme width on Apple Silicon.",
        expected="~0.10–0.13 if width keeps helping",
        paper_ref="ffno-2023",
    ),
    ExperimentConfig(
        name="ffno_h256_m32_l10",
        benchmark="burgers_1d", model="FFNO",
        hidden_dim=256, n_layers=10, n_modes=32,
        priority=2,
        rationale="FFNO wide+deep+max-modes. FFNO blocks are faster than FNO blocks "
                  "(no einsum over in_c×out_c) so l=10 may still get enough steps.",
        expected="~0.10–0.13",
        paper_ref="ffno-2023",
    ),
    ExperimentConfig(
        name="ffno_h128_m48_l8",
        benchmark="burgers_1d", model="FFNO",
        hidden_dim=128, n_layers=8, n_modes=48,
        priority=2,
        rationale="FFNO at m=48 (3× FNO max modes). Only feasible with diagonal "
                  "weights — FNO's full spectral matrix at m=48 would be 12× larger.",
        expected="~0.12–0.15 — very high-frequency coverage test",
        paper_ref="ffno-2023",
    ),
    ExperimentConfig(
        name="afno_v2_h256_m24_l8",
        benchmark="burgers_1d", model="AFNO",
        hidden_dim=256, n_layers=8, n_modes=24,
        priority=2,
        rationale="AFNO v2 wider channels (h=256). BlockDiagMLP operates on 2C=512 "
                  "features — tests whether AFNO benefits from wider hidden dim.",
        expected="~0.11–0.14",
        paper_ref="afno-2022",
    ),
    ExperimentConfig(
        name="ffno_kdv_h256_m32_l8",
        benchmark="kdv_1d", model="FFNO",
        hidden_dim=256, n_layers=8, n_modes=32,
        priority=3,
        rationale="FFNO on KdV. Soliton dynamics involve many harmonics → high "
                  "mode count is critical. FFNO enables m=32 at h=256 cheaply.",
        expected="~0.01–0.04",
        paper_ref="ffno-2023",
    ),

    # ── P18 · KdV follow-ups (best = RFNO 0.002023, both FNO/RFNO beat SOTA) ──
    # Goal: push KdV closer to zero. Baseline is 4.9× better than SOTA already.
    ExperimentConfig(
        name="rfno_kdv_h128_m24_l10",
        benchmark="kdv_1d", model="RFNO",
        hidden_dim=128, n_layers=10, n_modes=24,
        priority=1,
        rationale="RFNO beat FNO on KdV at l=8. RFNO at l=10: Pre-LN residuals "
                  "enable deeper stacks. KdV is smoother than Burgers → less "
                  "step-time concern; l=10 should still converge well.",
        expected="~0.001–0.002 (continued improvement)",
        paper_ref="rfno-2024",
    ),
    ExperimentConfig(
        name="rfno_kdv_h256_m24_l8",
        benchmark="kdv_1d", model="RFNO",
        hidden_dim=256, n_layers=8, n_modes=24,
        priority=1,
        rationale="RFNO wider (h=256) on KdV. FNO h=256 was step-time-limited on "
                  "Burgers; KdV step times are shorter (~37ms) so h=256 is feasible.",
        expected="~0.001–0.002",
        paper_ref="rfno-2024",
    ),
    ExperimentConfig(
        name="rfno_kdv_h128_m32_l8",
        benchmark="kdv_1d", model="RFNO",
        hidden_dim=128, n_layers=8, n_modes=32,
        priority=1,
        rationale="RFNO with more modes (m=32) on KdV. Solitons involve many "
                  "harmonics — m=32 captures full spectrum up to Nyquist.",
        expected="~0.001–0.002",
        paper_ref="rfno-2024",
    ),
    ExperimentConfig(
        name="fno_kdv_h256_m32_l8",
        benchmark="kdv_1d", model="FNO",
        hidden_dim=256, n_layers=8, n_modes=32,
        priority=2,
        rationale="FNO wide+max-modes on KdV. Tests whether FNO can match RFNO "
                  "with wider channels (h=256) instead of residual connections.",
        expected="~0.001–0.003",
    ),
    ExperimentConfig(
        name="rfno_kdv_h128_m24_l12",
        benchmark="kdv_1d", model="RFNO",
        hidden_dim=128, n_layers=12, n_modes=24,
        priority=2,
        rationale="Deep RFNO on KdV (l=12). FNO collapsed at l=12 on Burgers but "
                  "KdV is smoother. Pre-LN residuals should enable this depth.",
        expected="~0.0008–0.0015",
        paper_ref="rfno-2024",
    ),

    # ── P19 · Wave equation benchmark ────────────────────────────────────────
    # Linear wave PDE: u_tt = c² u_xx.  FNO should do very well (linear, periodic).
    ExperimentConfig(
        name="fno_wave_h128_m24_l8",
        benchmark="wave_1d", model="FNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        priority=1,
        rationale="Full-capacity FNO on wave equation. Linear PDE → FNO's linear "
                  "spectral conv is theoretically exact. SOTA: ~0.005.",
        expected="<0.001 (linear PDE should be near-exact for FNO)",
    ),
    ExperimentConfig(
        name="rfno_wave_h128_m24_l8",
        benchmark="wave_1d", model="RFNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        priority=1,
        rationale="RFNO on wave equation. Residuals + Pre-LN may help at l=8.",
        expected="<0.001",
    ),
    ExperimentConfig(
        name="fno_wave_h64_m16_l4",
        benchmark="wave_1d", model="FNO",
        hidden_dim=64, n_layers=4, n_modes=16,
        priority=2,
        rationale="Lightweight FNO baseline on wave: quick result to confirm "
                  "benchmark is working and data is learnable.",
        expected="~0.003–0.01",
    ),
    ExperimentConfig(
        name="fno_wave_h128_m24_l8_v2",
        benchmark="wave_1d", model="FNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        priority=2,
        rationale="Re-run of fno_wave_h128 with fixed wave ICs (ut0=0). "
                  "Previous run used random ut0 making the problem ill-posed.",
        expected="<0.001 — should be near-exact for FNO on linear wave PDE",
    ),

    # ── P20 · Data augmentation (spatial shift) ─────────────────────────────
    # Spatial shift on periodic 1D benchmarks is a zero-cost augmentation:
    # u_aug(x) = u(x + δ) for random δ.  Exploits translation symmetry.
    # No extra solver calls; just np.roll on each batch.
    ExperimentConfig(
        name="fno_h128_m24_l8_aug",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        augment=True,
        priority=1,
        rationale="Spatial shift augmentation on best FNO config. "
                  "Random roll of u0+uT exploits periodic translation symmetry. "
                  "Effective dataset size ∞ at zero solver cost.",
        expected="~0.12–0.14 (10-20% over baseline 0.155)",
        paper_ref="augmentation-2023",
    ),
    ExperimentConfig(
        name="rfno_h128_m24_l8_aug",
        benchmark="burgers_1d", model="RFNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        augment=True,
        priority=1,
        rationale="RFNO + spatial shift augmentation. Tests whether augmentation "
                  "compounds with residual connections.",
        expected="~0.12–0.14",
        paper_ref="augmentation-2023",
    ),
    ExperimentConfig(
        name="rfno_kdv_h128_m24_l8_aug",
        benchmark="kdv_1d", model="RFNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        augment=True,
        priority=2,
        rationale="KdV RFNO + augmentation. KdV is Galilean-invariant so spatial "
                  "shifts are physically valid. Already near-SOTA — push lower.",
        expected="~0.001–0.0015",
        paper_ref="augmentation-2023",
    ),
    ExperimentConfig(
        name="fno_h128_m24_l8_aug_ckpt",
        benchmark="burgers_1d", model="FNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        augment=True, save_ckpt=True,
        priority=2,
        rationale="Best augmented config + checkpoint saving. Enables ensemble "
                  "inference (run N times, average predictions → UQ).",
        expected="~0.12–0.14",
        paper_ref="ensemble-uq-2023",
    ),

    # ── P21 · Corrected 2D benchmarks ────────────────────────────────────────
    # prepare.py's darcy_2d and navier_stokes_2d benchmarks are broken at the
    # data level (read-only file).  These use the fixed solvers in benchmarks_ext.py.
    ExperimentConfig(
        name="fno_darcy2d_fix_h32_m8_l4",
        benchmark="darcy_2d_fix", model="FNO",
        hidden_dim=32, n_layers=4, n_modes=8,
        budget_s=480,
        priority=1,
        rationale="Baseline FNO on corrected Darcy 2D benchmark. "
                  "prepare.py solver used mean(a) only and fixed-seed f → val≈0.998. "
                  "This fix: Richardson iteration with full a(x,y); f tied to data seed.",
        expected="~0.05–0.15 (SOTA 0.0108 with FNO h=32 on proper Darcy)",
    ),
    ExperimentConfig(
        name="fno_darcy2d_fix_h64_m12_l4",
        benchmark="darcy_2d_fix", model="FNO",
        hidden_dim=64, n_layers=4, n_modes=12,
        budget_s=480,
        priority=2,
        rationale="Larger FNO on corrected Darcy 2D to approach SOTA 0.0108.",
        expected="~0.02–0.08",
    ),
    ExperimentConfig(
        name="fno_ns2d_fix_h32_m8_l4",
        benchmark="ns_2d_fix", model="FNO",
        hidden_dim=32, n_layers=4, n_modes=8,
        budget_s=480,
        priority=1,
        rationale="Baseline FNO on corrected NS 2D benchmark. "
                  "prepare.py solver: IC scale=1.0 → CFL≈61 → NaN. "
                  "This fix: IC scale=0.1 → CFL≈0.6, n_steps=1000.",
        expected="~0.05–0.20 (NS is harder than Darcy at T=1)",
    ),
    ExperimentConfig(
        name="fno_ns2d_fix_h64_m12_l4",
        benchmark="ns_2d_fix", model="FNO",
        hidden_dim=64, n_layers=4, n_modes=12,
        budget_s=480,
        priority=2,
        rationale="Larger FNO on corrected NS 2D benchmark.",
        expected="~0.03–0.10",
    ),

    # ── P22 · State Space Models (S4D) ──────────────────────────────────────
    ExperimentConfig(
        name="s4no_h64_l4_burgers",
        benchmark="burgers_1d", model="S4NO",
        hidden_dim=64, n_layers=4,
        priority=2,
        rationale="S4D (Diagonal S4) on Burgers. State-space model captures "
                  "global interactions with O(N log N) complexity.",
        expected="~0.15–0.25",
        paper_ref="ssm-s4-2022",
    ),
    ExperimentConfig(
        name="s4no_h128_l6_burgers",
        benchmark="burgers_1d", model="S4NO",
        hidden_dim=128, n_layers=6,
        priority=3,
        rationale="Wider/deeper S4D on Burgers.",
        expected="~0.12–0.20",
        paper_ref="ssm-s4-2022",
    ),

    # ── P23 · Neural Operator Transformer (GNOT) ─────────────────────────────
    ExperimentConfig(
        name="gnot_h64_l3_burgers",
        benchmark="burgers_1d", model="GNOT",
        hidden_dim=64, n_layers=3,
        priority=3,
        rationale="Simplified GNOT (self-attention) on Burgers. Transformer "
                  "architecture for operator learning.",
        expected="~0.15–0.25",
        paper_ref="gnot-2023",
    ),

    # ── P24 · Physics-Informed Neural Operator (PINO - MLP Surrogate) ────────
    # NOTE: original PINO entries commented out — fundamentally broken for
    # endpoint-only formulation. Only "fixed" variants below are active.
    # ExperimentConfig(
    #     name="pino_mlp_h128_l6_burgers",
    #     benchmark="burgers_1d", model="PINO",
    #     hidden_dim=128, n_layers=6,
    #     priority=2,
    #     rationale="BROKEN — endpoint-only PINO. Physics residual not applicable.",
    #     paper_ref="pino-2021",
    # ),
    # ExperimentConfig(
    #     name="pino_mlp_h128_l6_pino01",
    #     benchmark="burgers_1d", model="PINO",
    #     hidden_dim=128, n_layers=6,
    #     pino_lambda=0.01,
    #     priority=3,
    #     rationale="BROKEN — endpoint-only PINO with lambda.",
    #     paper_ref="pino-2021",
    # ),

    # ── New benchmark baselines and near-SOTA pushes ──────────────────────────

    # ns_2d_fix — currently 1.2x SOTA (0.0152 vs 0.0128) with only 1 run
    ExperimentConfig(
        name="ns2d_fno_h64_l4_m12",
        benchmark="ns_2d_fix",
        model="FNO",
        hidden_dim=64, n_layers=4, n_modes=12,
        budget_s=480,
        priority=1,
        rationale="Smaller FNO gets more steps in budget; wave_1d showed h=64 l=4 wins over h=128 l=8",
        expected="0.010–0.013",
    ),
    ExperimentConfig(
        name="ns2d_rfno_h128_l8_m24",
        benchmark="ns_2d_fix",
        model="RFNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        budget_s=480,
        priority=1,
        rationale="RFNO pre-LN stability; KdV best config transferred to 2D NS",
        expected="0.010–0.014",
    ),
    ExperimentConfig(
        name="ns2d_fno_h128_l6_m16",
        benchmark="ns_2d_fix",
        model="FNO",
        hidden_dim=128, n_layers=6, n_modes=16,
        budget_s=480,
        priority=1,
        rationale="Medium-size FNO; balance between capacity and training steps for 2D",
        expected="0.010–0.015",
    ),

    # darcy_2d_fix — currently 13.6x SOTA; h=32 too small, h=128 ran out of time
    ExperimentConfig(
        name="darcy2d_rfno_h64_l8_m12",
        benchmark="darcy_2d_fix",
        model="RFNO",
        hidden_dim=64, n_layers=8, n_modes=12,
        budget_s=480,
        priority=1,
        rationale="RFNO h=64 fits 2D budget; pre-LN stability for deeper Darcy net",
        expected="0.05–0.10",
    ),
    ExperimentConfig(
        name="darcy2d_fno_h64_l6_m16",
        benchmark="darcy_2d_fix",
        model="FNO",
        hidden_dim=64, n_layers=6, n_modes=16,
        budget_s=480,
        priority=1,
        rationale="FNO h=64 — goldilocks size for 2D budget constraint",
        expected="0.05–0.12",
    ),

    # Untested high-fidelity simulation baselines
    ExperimentConfig(
        name="euler1d_fno_h64_l4_m16",
        benchmark="euler_1d",
        model="FNO",
        hidden_dim=64, n_layers=4, n_modes=16,
        priority=2,
        rationale="First baseline on compressible Euler 1D; multi-channel FNO auto-routed to FNO_MC",
        expected="0.05–0.20",
    ),
    ExperimentConfig(
        name="euler1d_rfno_h128_l8_m24",
        benchmark="euler_1d",
        model="RFNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        priority=2,
        rationale="RFNO for multi-channel Euler; best KdV arch transferred",
        expected="0.02–0.10",
    ),
    ExperimentConfig(
        name="swe2d_fno_h64_l4_m12",
        benchmark="swe_2d",
        model="FNO",
        hidden_dim=64, n_layers=4, n_modes=12,
        budget_s=480,
        priority=2,
        rationale="First baseline on 2D shallow water; analytic solver makes data generation instant",
        expected="0.005–0.05",
    ),
    ExperimentConfig(
        name="allen_cahn_fno_h64_l4_m12",
        benchmark="allen_cahn_2d",
        model="FNO",
        hidden_dim=64, n_layers=4, n_modes=12,
        budget_s=480,
        priority=2,
        rationale="First baseline on Allen-Cahn phase field; ETDRK2 solver generates data in ~16s",
        expected="0.02–0.10",
    ),

    # Cross-benchmark transfer: RFNO+aug on Burgers (KdV insight)
    ExperimentConfig(
        name="burgers_rfno_h128_l8_m24_aug",
        benchmark="burgers_1d",
        model="RFNO",
        hidden_dim=128, n_layers=8, n_modes=24,
        augment=True,
        priority=2,
        rationale="RFNO + augmentation combo; aug alone gave best Burgers (0.1468), RFNO alone 0.1618",
        expected="0.13–0.15",
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
