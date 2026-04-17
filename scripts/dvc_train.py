"""DVC training wrapper — reads params.yaml and invokes train.py.

This script exists because DVC cmd templates don't support conditional
expressions (e.g. ${augment == true && "--augment" || ""}).  The wrapper
reads params.yaml, builds the correct argv, and execs train.py.

Called by dvc.yaml stages.train.  Do not call directly.
"""

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
PARAMS_FILE = REPO_ROOT / "params.yaml"
TRAIN_SCRIPT = REPO_ROOT / "train.py"


def main() -> None:
    with open(PARAMS_FILE) as f:
        p = yaml.safe_load(f)["train"]

    cmd = [sys.executable, str(TRAIN_SCRIPT)]

    # Required string / numeric flags
    cmd += ["--benchmark",   str(p["benchmark"])]
    cmd += ["--model",       str(p["model"])]
    cmd += ["--loss",        str(p["loss"])]
    cmd += ["--h1_alpha",    str(p["h1_alpha"])]
    cmd += ["--modes",       str(p["n_modes"])]
    cmd += ["--hidden",      str(p["hidden_dim"])]
    cmd += ["--layers",      str(p["n_layers"])]
    cmd += ["--batch_size",  str(p["batch_size"])]
    cmd += ["--lr",          str(p["lr"])]
    cmd += ["--grad_clip",   str(p["grad_clip"])]
    cmd += ["--budget",      str(p["budget_s"])]

    # Optional boolean flags (action="store_true" in train.py)
    if p.get("augment"):
        cmd.append("--augment")
    if p.get("curriculum"):
        cmd.append("--curriculum")
    if p.get("save_ckpt"):
        cmd.append("--save_ckpt")

    # Optional name
    name = str(p.get("name", "")).strip()
    if name:
        cmd += ["--name", name]

    print(f"[dvc_train] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
