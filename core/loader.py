"""Declarative experiment registry for autonomous SciML research.

This module now loads experiment configurations from 'experiments.yaml'.
To add a new experiment, append it to 'experiments.yaml'.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    name:        str              # unique key (used for dedup against results.json)
    benchmark:   str              # "burgers_1d" | "darcy_2d" | "kdv_1d" | "wave_1d"
    model:       str              # "FNO" | "RFNO" | "AFNO" | "FFNO" | "UNO" | "WNO" | "DeepONet" | "PODDeepONet"
    hidden_dim:  int              # channel width
    n_layers:    int              # depth (FNO blocks per level for UNO)
    n_modes:     int  = 16        # Fourier modes (FNO / UNO / RFNO / AFNO)
    n_levels:    int  = 3         # Haar levels (WNO)
    n_head:      int  = 4         # Attention heads (Transolver / GNOT / Transformer)
    slice_num:   int  = 32        # Physics slices (Transolver)
    lr:          float = 1e-3     # learning rate
    batch_size:  int  = 64         # training batch size
    grad_clip:   float = 1.0      # gradient clipping (0 = disabled)
    pino_lambda: float = 0.0      # PINO physics-loss weight
    loss_type:   str  = "l2_rel"  # loss function
    h1_alpha:    float = 0.1      # H1 loss derivative weight
    augment:     bool  = False    # spatial-shift augmentation
    curriculum:  bool  = False    # training curriculum
    curriculum_epochs: int = 0    # Phase 12: Curriculum ramp epochs
    save_ckpt:   bool  = False    # save model checkpoint after training
    resume:      bool  = False    # resume from best checkpoint if exists
    resume_from: str   = ""       # resume from specific checkpoint name/path
    budget_s:    int  = 1200      # training time budget in seconds
    parent_name: str  = ""        # name of parent experiment
    priority:    int  = 5         # 1 = highest
    rationale:   str  = ""        # why this experiment?
    expected:    str  = ""        # expected val_l2_rel range
    paper_ref:   str  = ""        # paper ID from papers/*.yaml
    refine_grid: bool = False      # Phase 11: Adaptive grid refinement
    cheb_degree: int  = 5         # Phase 11: Degree for Chebyshev KAN
    seed:        int  = 42        # Global RNG seed for reproducibility
    lr_schedule: str  = "warmup_cosine"  # LR schedule: warmup_cosine|cosine|onecycle|none
    ema_decay:   float = 0.0      # EMA decay for model weights (0=disabled, 0.999 recommended)
    patience:    int  = 5         # early-stop patience: halt after this many consecutive non-improving evals (0=off)
    snapshot_ensemble: int = 0    # Phase 12: Snapshot ensembling (number of snapshots to average)

    def to_cli_args(self) -> List[str]:
        args = [
            "--benchmark",   self.benchmark,
            "--model",       self.model,
            "--name",        self.name,
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
        if self.curriculum:
            args += ["--curriculum"]
        if self.curriculum_epochs > 0:
            args += ["--curriculum_epochs", str(self.curriculum_epochs)]
        if self.model in ("Transolver", "Transolver2D", "Transolver2d", "GNOT", "GNOT_Axial2d"):
            args += ["--n_head", str(self.n_head)]
            args += ["--slice_num", str(self.slice_num)]
        if self.save_ckpt:
            args += ["--save_ckpt"]
        if self.resume:
            args += ["--resume"]
        if self.resume_from:
            args += ["--resume_from", self.resume_from]
        if self.refine_grid:
            args += ["--refine_grid"]
        if self.model == "cPIKAN_FNO":
            args += ["--degree", str(self.cheb_degree)]
        args += ["--budget", str(self.budget_s)]
        args += ["--seed", str(self.seed)]
        if self.lr_schedule != "warmup_cosine":
            args += ["--lr_schedule", self.lr_schedule]
        if self.ema_decay > 0:
            args += ["--ema_decay", str(self.ema_decay)]
        if self.patience != 5:
            args += ["--patience", str(self.patience)]
        if self.snapshot_ensemble > 0:
            args += ["--snapshot_ensemble", str(self.snapshot_ensemble)]
        return args

    def short(self) -> str:
        """One-line summary for logging."""
        parts = [f"{self.model}", f"h={self.hidden_dim}", f"l={self.n_layers}"]
        if self.model in ("FNO", "RFNO", "AFNO", "FFNO", "UNO", "UNO2d", "WNO2d", "Transolver", "Transolver2D"):
            parts.append(f"m={self.n_modes}")
        if self.model in ("Transolver", "Transolver2D", "GNOT", "GNOT_Axial2d"):
            parts.append(f"h={self.n_head}")
            parts.append(f"s={self.slice_num}")
        if self.model in ("WNO", "WNO2d"):
            parts.append(f"lvl={self.n_levels}")
        if self.pino_lambda > 0:
            parts.append(f"pino={self.pino_lambda}")
        if self.lr != 1e-3:
            parts.append(f"lr={self.lr:.0e}")
        if self.grad_clip != 1.0:
            parts.append(f"clip={self.grad_clip}")
        if self.loss_type != "l2_rel":
            parts.append(f"loss={self.loss_type}")
        if self.curriculum:
            parts.append("curric")
        if self.curriculum_epochs > 0:
            parts.append(f"cur_ep={self.curriculum_epochs}")
        return "  ".join(parts)

# ── Loader Logic ─────────────────────────────────────────────────────────────

def load_experiments(yaml_path: Path) -> List[ExperimentConfig]:
    if not yaml_path.exists():
        return []
    
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    if not data:
        return []
        
    return [ExperimentConfig(**d) for d in data]

# Load default set
REPO_ROOT = Path(__file__).parent.parent
YAML_PATH = REPO_ROOT / "experiments.yaml"
EXPERIMENTS = load_experiments(YAML_PATH)

def get_experiments(benchmark: Optional[str] = None, model: Optional[str] = None, priority: Optional[int] = None) -> List[ExperimentConfig]:
    """Return the experiments list, optionally filtered by benchmark, model, or priority."""
    queue = EXPERIMENTS
    if benchmark:
        queue = [e for e in queue if e.benchmark == benchmark]
    if model:
        queue = [e for e in queue if e.model == model]
    if priority:
        queue = [e for e in queue if e.priority <= priority]
    return queue

if __name__ == "__main__":
    # Smoke test
    print(f"Loaded {len(EXPERIMENTS)} experiments.")
    if EXPERIMENTS:
        print(f"First experiment: {EXPERIMENTS[0].name} ({EXPERIMENTS[0].benchmark})")
