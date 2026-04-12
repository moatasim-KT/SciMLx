"""Declarative experiment registry for autonomous SciML research.

This module now loads experiment configurations from 'experiments.yaml'.
To add a new experiment, append it to 'experiments.yaml'.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

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
    save_ckpt:   bool  = False    # save model checkpoint after training
    resume:      bool  = False    # resume from best checkpoint if exists
    resume_from: str   = ""       # resume from specific checkpoint name/path
    budget_s:    int  = 1200      # training time budget in seconds
    parent_name: str  = ""        # name of parent experiment
    priority:    int  = 5         # 1 = highest
    rationale:   str  = ""        # why this experiment?
    expected:    str  = ""        # expected val_l2_rel range
    paper_ref:   str  = ""        # paper ID from papers/*.yaml

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
        if self.model in ("Transolver", "Transolver2D", "GNOT"):
            args += ["--n_head", str(self.n_head)]
            args += ["--slice_num", str(self.slice_num)]
        if self.save_ckpt:
            args += ["--save_ckpt"]
        if self.resume:
            args += ["--resume"]
        if self.resume_from:
            args += ["--resume_from", self.resume_from]
        args += ["--budget", str(self.budget_s)]
        return args

    def short(self) -> str:
        """One-line summary for logging."""
        parts = [f"{self.model}", f"h={self.hidden_dim}", f"l={self.n_layers}"]
        if self.model in ("FNO", "RFNO", "AFNO", "FFNO", "UNO", "Transolver", "Transolver2D"):
            parts.append(f"m={self.n_modes}")
        if self.model in ("Transolver", "Transolver2D", "GNOT"):
            parts.append(f"h={self.n_head}")
            parts.append(f"s={self.slice_num}")
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
        if self.curriculum:
            parts.append("curric")
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
REPO_ROOT = Path(__file__).parent
YAML_PATH = REPO_ROOT / "experiments.yaml"
EXPERIMENTS = load_experiments(YAML_PATH)

if __name__ == "__main__":
    # Smoke test
    print(f"Loaded {len(EXPERIMENTS)} experiments.")
    if EXPERIMENTS:
        print(f"First experiment: {EXPERIMENTS[0].name} ({EXPERIMENTS[0].benchmark})")
