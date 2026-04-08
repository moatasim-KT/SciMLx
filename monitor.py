#!/usr/bin/env python3
"""Live experiment monitor. Run: uv run monitor.py"""

import os
import re
import sys
import time
from pathlib import Path
from utils import LOGS_DIR, load_results, done_names, SOTA

CLEAR = "\033[2J\033[H"
BOLD  = "\033[1m"
DIM   = "\033[2m"
GREEN = "\033[32m"
RED   = "\033[31m"
CYAN  = "\033[36m"
YELLOW= "\033[33m"
RESET = "\033[0m"

BAR_W = 30

def bar(frac, width=BAR_W):
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)

def color_ratio(val, sota):
    if sota is None or val is None:
        return f"{val:.6f}"
    ratio = val / sota
    s = f"{val:.6f} ({ratio:.2f}× SOTA)"
    if ratio < 1.0:
        return GREEN + BOLD + s + RESET
    elif ratio < 2.0:
        return YELLOW + s + RESET
    else:
        return RED + s + RESET

def latest_log():
    logs = sorted(Path(LOGS_DIR).glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for log in logs:
        if log.stat().st_size > 0:
            return log
    return logs[0] if logs else None

def parse_log(path):
    """Extract key info from a log file."""
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return {}

    info = {}

    m = re.search(r"Benchmark:\s+(\S+)", text)
    if m: info["benchmark"] = m.group(1)

    m = re.search(r"Model\s*:\s+(\S+)\s+layers=(\d+)\s+hidden=(\d+)", text)
    if m: info["model"], info["layers"], info["hidden"] = m.group(1), m.group(2), m.group(3)

    m = re.search(r"Params\s*:\s+([\d.]+M)", text)
    if m: info["params"] = m.group(1)

    m = re.search(r"budget (\d+)s", text)
    if m: info["budget"] = int(m.group(1))

    # Latest val
    vals = re.findall(r"val@(\d+)%:\s+([\d.]+)", text)
    if vals:
        pct, v = vals[-1]
        info["val_pct"] = int(pct)
        info["val"] = float(v)
        # track best
        best_val = min(float(v) for _, v in vals)
        info["best_val"] = best_val

    # Latest step line
    steps = re.findall(r"step\s+(\d+)\s+\((\d+\.\d+)%\).*?loss:\s+([\d.]+).*?remaining:\s+(\d+)s", text)
    if steps:
        step, pct, loss, rem = steps[-1]
        info["step"] = int(step)
        info["step_pct"] = float(pct)
        info["loss"] = float(loss)
        info["remaining"] = int(rem)

    # Final result
    m = re.search(r"val_l2_rel:\s+([\d.]+)", text)
    if m: info["final_val"] = float(m.group(1))

    return info

def leaderboard():
    results = load_results()
    bm_best = {}
    for r in results:
        bm = r.get("benchmark", "")
        if bm in ("burgers_1d", "darcy_2d", "navier_stokes_2d"):
            continue
        val = r.get("val_l2_rel")
        if val and (bm not in bm_best or val < bm_best[bm][0]):
            bm_best[bm] = (val, r.get("model", "?"))
    return bm_best

def render(log_path, log_info, board, n_done, n_pending):
    lines = []
    lines.append(f"{BOLD}{CYAN}═══ SciML Live Monitor ═══{RESET}  {DIM}{time.strftime('%H:%M:%S')}{RESET}")
    lines.append(f"{DIM}Done: {n_done}  |  Pending non-burgers: {n_pending}{RESET}")
    lines.append("")

    # Current experiment
    if log_path and log_info:
        name = log_path.stem
        bm    = log_info.get("benchmark", "?")
        model = log_info.get("model", "?")
        h     = log_info.get("hidden", "?")
        l     = log_info.get("layers", "?")
        params= log_info.get("params", "?")

        lines.append(f"{BOLD}▶  {name}{RESET}")
        lines.append(f"   {DIM}benchmark={bm}  model={model}  h={h}  l={l}  params={params}{RESET}")

        # Progress bar
        pct  = log_info.get("step_pct") or (log_info.get("val_pct") or 0)
        frac = pct / 100
        rem  = log_info.get("remaining", 0)
        loss = log_info.get("loss")

        rem_str = f"{rem//60}m {rem%60:02d}s" if rem >= 60 else f"{rem}s"
        lines.append(f"   [{bar(frac)}] {BOLD}{pct:.1f}%{RESET}")
        if loss is not None:
            lines.append(f"   {'Loss:':<14} {BOLD}{loss:.6f}{RESET}")
        if rem:
            lines.append(f"   {'Remaining:':<14} {CYAN}{rem_str}{RESET}")

        val      = log_info.get("val")
        best_val = log_info.get("best_val")
        sota_val = SOTA.get(bm)
        if val is not None:
            lines.append(f"   latest val : {color_ratio(val, sota_val)}")
        if best_val is not None and best_val != val:
            lines.append(f"   best so far: {color_ratio(best_val, sota_val)}")
        if "final_val" in log_info:
            lines.append(f"   {GREEN}FINAL: {log_info['final_val']:.6f}{RESET}")
    else:
        lines.append(f"{DIM}(waiting for experiment to start...){RESET}")

    lines.append("")
    lines.append(f"{BOLD}Leaderboard (non-burgers){RESET}")
    lines.append(f"  {'Benchmark':<20} {'Best':>10}  {'Model':<12}  {'vs SOTA'}")
    lines.append(f"  {'─'*20}  {'─'*10}  {'─'*12}  {'─'*15}")

    sota_targets = {
        "allen_cahn_2d": 0.020, "darcy_2d_fix": 0.0108, "euler_1d": 0.015,
        "kdv_1d": 0.010, "ns_2d_fix": 0.0128, "swe_2d": 0.002, "wave_1d": 0.005,
    }
    for bm in sorted(sota_targets):
        s = sota_targets[bm]
        if bm in board:
            val, model = board[bm]
            ratio = val / s
            tag = f"{ratio:.2f}×"
            col = GREEN if ratio < 1 else (YELLOW if ratio < 3 else RED)
            lines.append(f"  {bm:<20} {val:>10.6f}  {model:<12}  {col}{tag}{RESET}")
        else:
            lines.append(f"  {bm:<20} {'—':>10}   {'—':<12}  {DIM}not run{RESET}")

    return "\n".join(lines)

def main():
    refresh = 3  # seconds
    n_pending_cache = "?"
    try:
        from experiments import EXPERIMENTS
        n_pending_cache = sum(
            1 for e in EXPERIMENTS
            if e.name not in done_names() and e.benchmark != "burgers_1d"
        )
    except Exception:
        pass

    print("Starting monitor (Ctrl+C to quit)...")
    time.sleep(1)

    while True:
        try:
            log_path = latest_log()
            log_info = parse_log(log_path) if log_path else {}
            board    = leaderboard()
            n_done   = len(done_names())
            try:
                from experiments import EXPERIMENTS
                n_pending_cache = sum(
                    1 for e in EXPERIMENTS
                    if e.name not in done_names() and e.benchmark != "burgers_1d"
                )
            except Exception:
                pass

            output = render(log_path, log_info, board, n_done, n_pending_cache)
            sys.stdout.write(CLEAR + output + "\n")
            sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break
        except Exception as e:
            sys.stdout.write(f"\r[monitor error: {e}]")
            sys.stdout.flush()
        time.sleep(refresh)

if __name__ == "__main__":
    main()
