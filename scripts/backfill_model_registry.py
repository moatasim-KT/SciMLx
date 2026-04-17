"""Backfill model_registry.json from existing results.json + checkpoints/.

Finds the best val_l2_rel "keep" result per (benchmark, model) pair, matches
it to a checkpoint file in checkpoints/, and registers it.  Safe to re-run:
already-registered versions are skipped unless --force is passed.

Usage:
    uv run python scripts/backfill_model_registry.py
    uv run python scripts/backfill_model_registry.py --dry-run
    uv run python scripts/backfill_model_registry.py --force   # re-register all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
RESULTS_FILE = REPO_ROOT / "results.json"
CKPT_DIR    = REPO_ROOT / "checkpoints"


def find_checkpoint(exp_name: str, model: str, benchmark: str) -> Path | None:
    """Try several naming conventions to locate a checkpoint for this run."""
    candidates = []

    # 1. Primary: {exp_name}_best.npz  (written by trainer.py mid-run)
    if exp_name:
        candidates.append(CKPT_DIR / f"{exp_name}_best.npz")
        # Lowercase variant (some older runs used lower)
        candidates.append(CKPT_DIR / f"{exp_name.lower()}_best.npz")

    # 2. Fallback pattern: {model}_{benchmark}_val*.npz  (--save_ckpt path)
    if model and benchmark:
        pattern = f"{model}_{benchmark}_val*.npz"
        matches = sorted(CKPT_DIR.glob(pattern),
                         key=lambda p: float(p.stem.split("val")[-1]))
        candidates.extend(matches)

    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force",   action="store_true",
                    help="Re-register even if already in registry")
    args = ap.parse_args()

    if not RESULTS_FILE.exists():
        print("results.json not found — nothing to backfill.")
        return

    with open(RESULTS_FILE) as f:
        all_results = json.load(f)

    # Only consider "keep" entries with a valid score
    keeps = [
        e for e in all_results
        if e.get("status") == "keep"
        and e.get("val_l2_rel") and 0 < e["val_l2_rel"] < 10.0
    ]

    # Best result per (benchmark, model)
    best: dict[tuple, dict] = {}
    for e in keeps:
        key = (e["benchmark"], e["model"])
        if key not in best or e["val_l2_rel"] < best[key]["val_l2_rel"]:
            best[key] = e

    print(f"Found {len(best)} unique (benchmark, model) pairs from {len(keeps)} keep entries.\n")

    if not args.dry_run:
        # Import after path is set
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from core.model_versioning import get_registry
        reg = get_registry()

        # Build set of already-registered (benchmark, model, exp_name) to skip
        registered = {
            (v.benchmark, v.model, v.exp_name)
            for v in reg._versions
        }

    registered_count = 0
    skipped_count    = 0
    no_ckpt_count    = 0

    for (benchmark, model), e in sorted(best.items()):
        cfg      = e.get("config") or {}
        exp_name = cfg.get("name") or e.get("description", "").split()[0]
        val      = e["val_l2_rel"]

        key = (benchmark, model, exp_name)
        if not args.dry_run and not args.force and key in registered:
            print(f"  SKIP   {benchmark:28s} {model:30s} {val:.6f}  (already registered)")
            skipped_count += 1
            continue

        ckpt = find_checkpoint(exp_name, model, benchmark)
        ckpt_display = ckpt.name if ckpt else "—"

        if args.dry_run:
            tag = "FOUND " if ckpt else "NO_CKPT"
            print(f"  {tag:8s} {benchmark:28s} {model:30s} {val:.6f}  {ckpt_display}")
            if not ckpt:
                no_ckpt_count += 1
            else:
                registered_count += 1
            continue

        if ckpt:
            version_id = reg.register(
                ckpt_path     = ckpt,
                benchmark     = benchmark,
                model         = model,
                exp_name      = exp_name,
                val_l2_rel    = val,
                config        = cfg,
                mlflow_run_id = None,
            )
            print(f"  REG    {benchmark:28s} {model:30s} {val:.6f}  {ckpt.name}")
            registered_count += 1
        else:
            print(f"  NO_CKPT {benchmark:28s} {model:30s} {val:.6f}  (checkpoint not found)")
            no_ckpt_count += 1

    print(f"\nDone — registered: {registered_count}  skipped: {skipped_count}  no-ckpt: {no_ckpt_count}")


if __name__ == "__main__":
    main()
