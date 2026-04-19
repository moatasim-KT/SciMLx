"""Closed-Loop Reasoner for SciML AutoResearch.

After every non-keep experiment (discard or crash), this module reads the
result record + diagnostic stats + log context and generates an improved
follow-up ExperimentConfig. The improved config is appended to
experiments.yaml so the next autorun iteration picks it up automatically.

This creates the closed loop:
    run → discard/crash → reason → improved config in queue → run → ...

Usage (autorun.py integration):
    from core.closed_loop_reasoner import ClosedLoopReasoner
    reasoner = ClosedLoopReasoner()
    follow_up = reasoner.reason(exp, result_record, log_path)
    if follow_up:
        reasoner.enqueue(follow_up)
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from core.utils import REPO_ROOT, BENCHMARKS_2D, LOGS_DIR, load_results

EXPERIMENTS_YAML = REPO_ROOT / "experiments.yaml"
REFLECTIONS_FILE = LOGS_DIR / "reflections.json"

# Minimum number of revision cycles that must complete on a benchmark
# before new (never-tried) experiment configs may be introduced.
MIN_REVISIONS_BEFORE_NEW = 3

# Hard ceilings for Apple Silicon
MAX_2D_HIDDEN = 32
MAX_2D_LAYERS = 4
MAX_2D_MODES  = 12
MAX_1D_HIDDEN = 128
MAX_1D_LAYERS = 8

# Models known to be broken / never queue
BLACKLISTED_MODELS = {"PINO", "AFNO", "RFNO2D_broken"}
# Models that need a code fix before requeuing (2D shape contract broken)
NEEDS_CODE_FIX = {"MambaNO", "MemNO", "GNOT2D", "UNO2d", "WNO2d"}

# How many times we'll generate follow-ups for the same base experiment
MAX_FOLLOWUPS = 3

# ── Utility ───────────────────────────────────────────────────────────────────

def _is_2d(benchmark: str) -> bool:
    return benchmark in BENCHMARKS_2D

def _followup_name(base_name: str, n: int) -> str:
    """Generate collision-safe follow-up name."""
    # Strip existing _fn suffix if present
    base = re.sub(r"_f\d+$", "", base_name)
    return f"{base}_f{n}"

def _count_followups(base_name: str) -> int:
    """Count how many follow-ups already exist in experiments.yaml."""
    try:
        with open(EXPERIMENTS_YAML) as f:
            content = f.read()
        base = re.sub(r"_f\d+$", "", base_name)
        return len(re.findall(rf"name: {re.escape(base)}_f\d+", content))
    except Exception:
        return 0

def _name_exists(name: str) -> bool:
    try:
        with open(EXPERIMENTS_YAML) as f:
            return f"name: {name}" in f.read()
    except Exception:
        return False


# ── Revision chain helpers ─────────────────────────────────────────────────────

def _revision_chains(benchmark: str) -> Dict[str, List[dict]]:
    """
    Return all revision chains for a benchmark from results.json.
    A chain is {base_name: [r0, r1_f1, r2_f2, ...]} ordered by revision index.
    Only includes chains where the base experiment is a non-keep result.
    """
    try:
        results = load_results()
    except Exception:
        return {}

    bm_results = [r for r in results if r.get("benchmark") == benchmark]

    chains: Dict[str, List[dict]] = {}
    for r in bm_results:
        cfg = r.get("config") or {}
        name = cfg.get("name") or r.get("description", "").split()[0]
        if not name:
            continue
        base = re.sub(r"_f\d+$", "", name)
        if base not in chains:
            chains[base] = []
        chains[base].append(r)

    # Sort each chain by revision index (f0 < f1 < f2 < f3)
    def _rev_idx(r: dict) -> int:
        n = (r.get("config") or {}).get("name") or r.get("description", "").split()[0]
        m = re.search(r"_f(\d+)$", n or "")
        return int(m.group(1)) if m else 0

    return {base: sorted(chain, key=_rev_idx) for base, chain in chains.items()}


def benchmark_ready_for_new(benchmark: str) -> bool:
    """
    Return True if the benchmark is ready for new (never-tried) experiment configs.
    Condition: every existing non-keep experiment chain has completed >= MIN_REVISIONS_BEFORE_NEW
    revision cycles, OR the chain has reached a keep result.
    Returns True when there are no chains at all (first experiments on a benchmark).
    """
    chains = _revision_chains(benchmark)
    if not chains:
        return True

    for base, chain in chains.items():
        # Skip chains that have a keep result — they're done
        if any(r.get("status") == "keep" for r in chain):
            continue
        # Count completed revision cycles (follow-ups that have a result)
        completed_revisions = sum(
            1 for r in chain
            if re.search(r"_f\d+$", (r.get("config") or {}).get("name") or "")
        )
        if completed_revisions < MIN_REVISIONS_BEFORE_NEW:
            return False
    return True


# ── Reflection engine ──────────────────────────────────────────────────────────

def generate_reflection(benchmark: str) -> Dict[str, Any]:
    """
    Analyse all revision chains for a benchmark and produce a structured reflection:
    - what_worked: configs/changes that reduced val_l2_rel
    - what_didnt: configs that failed or made things worse
    - further_steps: prioritised list of recommended next actions
    - chain_summaries: per-chain val progression
    """
    chains = _revision_chains(benchmark)

    what_worked: List[str] = []
    what_didnt: List[str] = []
    further_steps: List[str] = []
    chain_summaries: List[Dict] = []

    all_kept = []
    all_vals = []

    for base, chain in chains.items():
        vals = []
        statuses = []
        critiques = []
        for r in chain:
            v = r.get("val_l2_rel")
            s = r.get("status", "unknown")
            cfg = r.get("config") or {}
            c = cfg.get("critique") or r.get("rationale") or ""
            vals.append(v)
            statuses.append(s)
            critiques.append(c)
            if v is not None:
                all_vals.append(v)
            if s == "keep":
                all_kept.append(r)

        # Determine if chain improved
        valid_vals = [v for v in vals if v is not None]
        improved = len(valid_vals) >= 2 and valid_vals[-1] < valid_vals[0]
        best_val = min(valid_vals) if valid_vals else None
        final_status = statuses[-1] if statuses else "unknown"

        chain_summaries.append({
            "base": base,
            "revisions": len(chain),
            "val_progression": vals,
            "statuses": statuses,
            "best_val": best_val,
            "final_status": final_status,
            "improved": improved,
        })

        # Attribute what worked/didn't by comparing consecutive revisions
        for i in range(1, len(chain)):
            prev_v = vals[i - 1]
            curr_v = vals[i]
            curr_cfg = chain[i].get("config") or {}
            curr_loss = curr_cfg.get("loss_type", "l2_rel")
            curr_ema = curr_cfg.get("ema_decay", 0)
            curr_cur = curr_cfg.get("curriculum", False)
            curr_ens = curr_cfg.get("snapshot_ensemble", 0)

            if prev_v is not None and curr_v is not None:
                delta = prev_v - curr_v
                changes = []
                prev_cfg = chain[i - 1].get("config") or {}
                if curr_cfg.get("loss_type") != prev_cfg.get("loss_type"):
                    changes.append(f"loss {prev_cfg.get('loss_type','l2_rel')}→{curr_loss}")
                if curr_ema and not prev_cfg.get("ema_decay"):
                    changes.append("added EMA=0.999")
                if curr_cur and not prev_cfg.get("curriculum"):
                    changes.append("added curriculum")
                if curr_ens and not prev_cfg.get("snapshot_ensemble"):
                    changes.append("added snapshot_ensemble")
                if curr_cfg.get("lr", 1e-3) < prev_cfg.get("lr", 1e-3) * 0.6:
                    changes.append(f"lr↓ {prev_cfg.get('lr',1e-3):.1e}→{curr_cfg.get('lr',1e-3):.1e}")

                tag = f"{base}: " + (", ".join(changes) if changes else "config tweak")
                if delta > 0.005:
                    what_worked.append(f"{tag} → val {prev_v:.4f}→{curr_v:.4f} (Δ-{delta:.4f})")
                elif delta < -0.005:
                    what_didnt.append(f"{tag} → val worsened {prev_v:.4f}→{curr_v:.4f}")
                elif statuses[i] == "crash":
                    what_didnt.append(f"{tag} → crashed")

    # Derive further steps from patterns
    crashed_models = {
        (r.get("model"), r.get("benchmark"))
        for base, chain in chains.items()
        for r in chain if r.get("status") == "crash"
    }
    poor_chains = [s for s in chain_summaries if s["best_val"] and s["best_val"] > 0.3]
    near_sota_chains = [s for s in chain_summaries if s["best_val"] and s["best_val"] < 0.05]

    if poor_chains:
        further_steps.append(
            f"{len(poor_chains)} chains still val>0.3 — try switching model family (FEDONet2D/HANO2D) "
            f"instead of hyperparameter tweaking"
        )
    if near_sota_chains:
        further_steps.append(
            f"{len(near_sota_chains)} chains near val<0.05 — push with longer budget, "
            f"snapshot_ensemble=5, lower lr=5e-4"
        )
    if crashed_models:
        further_steps.append(
            f"{len(crashed_models)} model/benchmark pairs crashed — verify data pipeline with "
            f"h=16 l=2 diagnostic probe before requeuing at full scale"
        )

    # Check if SOTA is beaten
    try:
        from core.utils import SOTA
        sota = SOTA.get(benchmark)
        our_best = min((v for v in all_vals if v is not None), default=None)
        if sota and our_best:
            ratio = our_best / sota
            if ratio < 1.0:
                further_steps.insert(0, f"SOTA BEATEN (our={our_best:.5f} < sota={sota:.5f}). "
                                        f"Focus on robustness: varied seeds, ensemble depth.")
            elif ratio < 1.5:
                further_steps.insert(0, f"Very close to SOTA ({ratio:.2f}×). "
                                        f"Try 2× budget + snapshot_ensemble=5.")
            else:
                further_steps.append(f"Still {ratio:.1f}× from SOTA ({our_best:.4f} vs {sota:.4f}). "
                                      f"Consider novel architecture or physics-informed loss.")
    except Exception:
        pass

    reflection = {
        "benchmark": benchmark,
        "generated_at": datetime.now().isoformat(),
        "total_chains": len(chains),
        "total_keep": len(all_kept),
        "best_val": min((v for v in all_vals if v is not None), default=None),
        "what_worked": what_worked,
        "what_didnt": what_didnt,
        "further_steps": further_steps,
        "chain_summaries": chain_summaries,
    }

    # Persist to logs/reflections.json
    _save_reflection(reflection)
    return reflection


def _save_reflection(reflection: Dict[str, Any]) -> None:
    """Append/update reflection in logs/reflections.json."""
    try:
        if REFLECTIONS_FILE.exists():
            data = json.loads(REFLECTIONS_FILE.read_text())
        else:
            data = {}
        data[reflection["benchmark"]] = reflection
        REFLECTIONS_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def load_reflections() -> Dict[str, Any]:
    """Load all saved reflections from logs/reflections.json."""
    try:
        if REFLECTIONS_FILE.exists():
            return json.loads(REFLECTIONS_FILE.read_text())
    except Exception:
        pass
    return {}


# ── Rule engine ───────────────────────────────────────────────────────────────

def _apply_rules(
    model: str,
    benchmark: str,
    val: Optional[float],
    crash_type: Optional[str],
    diag: Dict[str, float],
    config: Dict[str, Any],
    log_text: str,
) -> tuple[Dict[str, Any], str]:
    """
    Core reasoning rules. Returns (config_overrides, critique_text).

    Rules are applied in priority order — first matching rule wins for each
    field, multiple rules can contribute to the critique.
    """
    is_2d = _is_2d(benchmark)
    overrides: Dict[str, Any] = {}
    critique_parts: list[str] = []

    h = config.get("hidden_dim", 64)
    l = config.get("n_layers", 4)
    m = config.get("n_modes", 16)
    lr = config.get("lr", 1e-3)
    loss = config.get("loss_type", "l2_rel")
    ema = config.get("ema_decay", 0.0)
    clip = config.get("grad_clip", 1.0)
    budget = config.get("budget_s", 1800)
    ens = config.get("snapshot_ensemble", 0)
    cur = config.get("curriculum", False)
    slice_num = config.get("slice_num", 32)

    high_freq  = diag.get("diag_high_freq_error", 0.0)
    grad_norm  = diag.get("diag_grad_norm_max", 0.0)
    low_freq   = diag.get("diag_low_freq_error", 0.0)

    # ── CRASH HANDLING ────────────────────────────────────────────────────────

    if crash_type == "OOM":
        critique_parts.append(f"OOM crash: hidden_dim={h} exceeded memory ceiling.")
        if is_2d:
            overrides["hidden_dim"] = max(16, h // 2)
            overrides["n_layers"]   = min(l, 2)
            overrides["n_modes"]    = min(m, 8)
        else:
            overrides["hidden_dim"] = max(32, h // 2)
        critique_parts.append(f"Fix: hidden_dim → {overrides['hidden_dim']}, n_layers → {overrides.get('n_layers', l)}.")

    elif crash_type == "BroadcastingError":
        # Model's __call__ doesn't handle 2D [B,N1,N2,C] contract correctly
        critique_parts.append(
            f"BroadcastingError: {model} output shape mismatch on {benchmark}. "
            f"Model needs 2D __call__ fix before requeuing. "
            f"Workaround: use FNO2D or FEDONet2D which have verified 2D contracts."
        )
        overrides["model"]      = "FNO2D" if is_2d else "FNO"
        overrides["hidden_dim"] = MAX_2D_HIDDEN if is_2d else 64
        overrides["n_modes"]    = 8

    elif crash_type == "IncompatibleDimensions":
        critique_parts.append(f"IncompatibleDimensions: n_modes={m} too large for grid (N=64 → max 32). Fix: n_modes=8.")
        overrides["n_modes"] = 8

    elif crash_type in ("ValueError", "NaN/Inf"):
        critique_parts.append(f"NaN/Inf crash at lr={lr:.1e}. Gradients exploded.")
        new_lr = max(lr / 10, 1e-5)
        overrides["lr"]        = new_lr
        overrides["grad_clip"] = min(clip, 0.5) if clip > 0 else 0.5
        overrides["loss_type"] = "l2_rel"   # safest loss for first stable run
        critique_parts.append(f"Fix: lr → {new_lr:.1e}, grad_clip → 0.5, loss → l2_rel.")

    elif crash_type == "EarlyStop":
        critique_parts.append("Crashed via early stopping — diverged in first 30% of budget.")
        overrides["lr"]      = max(lr / 5, 5e-5)
        overrides["patience"] = 0   # disable early stop to see full trajectory
        critique_parts.append(f"Fix: lr → {overrides['lr']:.1e}, patience → 0.")

    elif crash_type == "NoOutput":
        critique_parts.append(
            f"NoOutput on {benchmark} — data-generation timeout or solver crash. "
            f"Reduce model size drastically and run diagnostic probe first."
        )
        overrides["hidden_dim"] = 16 if is_2d else 32
        overrides["n_layers"]   = 2
        overrides["n_modes"]    = 4

    # ── POOR RESULT HANDLING ──────────────────────────────────────────────────

    elif val is not None and val > 0.8:
        critique_parts.append(f"val={val:.4f} — complete convergence failure, near-random predictions.")
        if model in ("UDE", "NeuralODE", "LatentODE", "UDE"):
            critique_parts.append("ODE-family diverges on this PDE. Switch to FNO/RFNO family.")
            overrides["model"]       = "RFNO" if not is_2d else "FEDONet2D"
            overrides["hidden_dim"]  = MAX_1D_HIDDEN if not is_2d else MAX_2D_HIDDEN
            overrides["n_modes"]     = 24 if not is_2d else 8
            overrides["n_layers"]    = 8 if not is_2d else 4
        elif model in ("Transolver", "Transolver2D") and slice_num > 16:
            critique_parts.append(f"Transolver slice_num={slice_num} too large — use slice_num=8.")
            overrides["slice_num"] = 8
            overrides["n_head"]    = 4
        else:
            overrides["lr"]        = max(lr / 3, 3e-4)
            overrides["loss_type"] = "l2_rel"
            overrides["grad_clip"] = 1.0
            critique_parts.append(f"Fix: lr → {overrides['lr']:.1e}, loss → l2_rel, grad_clip=1.0.")

    elif val is not None and val > 0.3:
        critique_parts.append(f"val={val:.4f} — poor convergence (>2x baseline).")
        # Diagnose from spectral stats
        if high_freq > 0.3:
            critique_parts.append(f"diag_high_freq_error={high_freq:.2f} (>0.3) → switch to spectral loss.")
            overrides["loss_type"] = "spectral"
        elif high_freq > 0.1:
            critique_parts.append(f"diag_high_freq_error={high_freq:.2f} → try h1_adaptive loss.")
            overrides["loss_type"] = "h1_adaptive"
        elif grad_norm > 10:
            critique_parts.append(f"diag_grad_norm_max={grad_norm:.1f} (>10) → halve lr, tighten grad_clip.")
            overrides["lr"]        = max(lr / 2, 1e-4)
            overrides["grad_clip"] = 0.5
        else:
            critique_parts.append("No spectral anomaly — likely capacity-limited or LR schedule issue.")
            overrides["loss_type"] = "h1_adaptive"
            overrides["ema_decay"] = 0.999

        # Add EMA if not already using it
        if ema == 0.0:
            overrides["ema_decay"] = 0.999
            critique_parts.append("Adding EMA=0.999 (3-8% free improvement).")

    elif val is not None and val > 0.15:
        critique_parts.append(f"val={val:.4f} — below best, marginal run.")
        # Try a training trick not yet applied
        if ema == 0.0:
            overrides["ema_decay"] = 0.999
            critique_parts.append("EMA not used — add ema_decay=0.999.")
        if ens == 0:
            overrides["snapshot_ensemble"] = 3
            critique_parts.append("Add snapshot_ensemble=3 for inverse-val-error weighting.")
        if not cur:
            overrides["curriculum"] = True
            critique_parts.append("Add curriculum=True for spectral ramp-up.")
        if loss == "l2_rel":
            if high_freq > 0.1:
                overrides["loss_type"] = "h1_adaptive"
                critique_parts.append(f"diag_high_freq={high_freq:.2f} → h1_adaptive loss.")
            elif high_freq > 0.3:
                overrides["loss_type"] = "spectral"

    else:
        # val <= 0.15 — decent run but not best, try ensemble + EMA to push further
        critique_parts.append(f"val={val:.4f} — reasonable but below best. Try full training stack.")
        if ema == 0.0:
            overrides["ema_decay"] = 0.999
        if ens == 0:
            overrides["snapshot_ensemble"] = 3
        if not cur:
            overrides["curriculum"] = True

    # ── BUDGET: always apply floor and extend for follow-ups ─────────────────
    if is_2d:
        overrides["budget_s"] = max(budget, 3600)
    else:
        overrides["budget_s"] = max(budget, 1800)

    # ── HARD CONSTRAINTS ──────────────────────────────────────────────────────
    if is_2d:
        overrides["hidden_dim"] = min(overrides.get("hidden_dim", h), MAX_2D_HIDDEN)
        overrides["n_layers"]   = min(overrides.get("n_layers", l), MAX_2D_LAYERS)
        overrides["n_modes"]    = min(overrides.get("n_modes", m), MAX_2D_MODES)
    else:
        overrides["hidden_dim"] = min(overrides.get("hidden_dim", h), MAX_1D_HIDDEN)
        overrides["n_layers"]   = min(overrides.get("n_layers", l), MAX_1D_LAYERS)

    critique = " | ".join(critique_parts) if critique_parts else f"val={val} — follow-up with improved training stack."
    return overrides, critique


# ── Main class ────────────────────────────────────────────────────────────────

class ClosedLoopReasoner:
    """
    Generates improved follow-up ExperimentConfigs from failed/discarded runs.

    Call `reason(exp, result_dict, log_path)` after every non-keep run.
    Call `enqueue(follow_up_dict)` to append it to experiments.yaml.
    """

    def reason(
        self,
        exp_name: str,
        benchmark: str,
        model: str,
        val: Optional[float],
        crash_type: Optional[str],
        diag: Dict[str, float],
        config: Dict[str, Any],
        log_path: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyse a failed/discarded run and return an improved config dict,
        or None if the model is blacklisted or max follow-ups reached.
        """
        # Skip blacklisted / code-broken models
        if model in BLACKLISTED_MODELS:
            return None
        if model in NEEDS_CODE_FIX and crash_type == "BroadcastingError":
            # Still generate a follow-up but switch to a working model
            pass

        # Don't generate more than MAX_FOLLOWUPS follow-ups for the same base
        base_name = re.sub(r"(_r\d+|_adapt|_f\d+|_retry)$", "", exp_name)
        existing = _count_followups(base_name)
        if existing >= MAX_FOLLOWUPS:
            # Chain exhausted — generate a reflection for this benchmark
            try:
                generate_reflection(benchmark)
            except Exception:
                pass
            return None

        # Read log text for additional context
        log_text = ""
        if log_path and Path(log_path).exists():
            try:
                log_text = Path(log_path).read_text()[-4000:]  # last 4KB
            except Exception:
                pass

        overrides, critique = _apply_rules(
            model=model,
            benchmark=benchmark,
            val=val,
            crash_type=crash_type,
            diag=diag,
            config=config,
            log_text=log_text,
        )

        # Build follow-up name
        followup_n = existing + 1
        followup_name = _followup_name(base_name, followup_n)
        if _name_exists(followup_name):
            return None  # already queued

        # Build final config: start from original, apply overrides
        final = {
            "name":       followup_name,
            "benchmark":  benchmark,
            "model":      overrides.pop("model", model),
            "hidden_dim": overrides.pop("hidden_dim", config.get("hidden_dim", 64)),
            "n_layers":   overrides.pop("n_layers",  config.get("n_layers", 4)),
            "n_modes":    overrides.pop("n_modes",   config.get("n_modes", 16)),
            "lr":         overrides.pop("lr",        config.get("lr", 1e-3)),
            "loss_type":  overrides.pop("loss_type", config.get("loss_type", "l2_rel")),
            "ema_decay":  overrides.pop("ema_decay", config.get("ema_decay", 0.0)),
            "curriculum": overrides.pop("curriculum", config.get("curriculum", False)),
            "snapshot_ensemble": overrides.pop("snapshot_ensemble", config.get("snapshot_ensemble", 0)),
            "grad_clip":  overrides.pop("grad_clip",  config.get("grad_clip", 1.0)),
            "patience":   overrides.pop("patience",   config.get("patience", 5)),
            "budget_s":   overrides.pop("budget_s",   1800),
            "priority":   1,
            "parent_name": exp_name,
            "rationale":  f"[ClosedLoop] Follow-up to {exp_name} (val={f'{val:.4f}' if val else 'crash'}). {critique[:200]}",
            "critique":   critique,
        }
        # Apply any remaining overrides (e.g., slice_num, n_head)
        for k, v in overrides.items():
            final[k] = v

        # Add h1_alpha if h1 loss
        if final["loss_type"].startswith("h1") and "h1_alpha" not in final:
            final["h1_alpha"] = config.get("h1_alpha", 0.2)

        return final

    def enqueue(self, config_dict: Dict[str, Any]) -> None:
        """Append a follow-up config to experiments.yaml."""
        try:
            with open(EXPERIMENTS_YAML, "a") as f:
                f.write("\n")
                yaml.dump([config_dict], f, default_flow_style=False,
                          allow_unicode=True, sort_keys=False, width=120)
        except Exception as e:
            print(f"  [ClosedLoop] Failed to enqueue {config_dict.get('name')}: {e}")

    def reason_and_enqueue(
        self,
        exp_name: str,
        benchmark: str,
        model: str,
        val: Optional[float],
        crash_type: Optional[str],
        diag: Dict[str, float],
        config: Dict[str, Any],
        log_path: Optional[Path] = None,
    ) -> Optional[str]:
        """
        Convenience wrapper: reason + enqueue in one call.
        Returns the follow-up name if enqueued, None otherwise.
        """
        follow_up = self.reason(
            exp_name=exp_name, benchmark=benchmark, model=model,
            val=val, crash_type=crash_type, diag=diag,
            config=config, log_path=log_path,
        )
        if follow_up:
            self.enqueue(follow_up)
            return follow_up["name"]
        return None
