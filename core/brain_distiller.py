"""
Brain Distiller — Automated evolution of RESEARCH_BRAIN.md.

Parses trajectories.jsonl and results.json to extract scientific insights,
structural critiques, and confirmed patterns, then distills them into
the project's Living Brain.
"""

import json
import re
import os
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from core.arxiv_agent import ArXivAgent
from core.utils import REPO_ROOT, LOGS_DIR, PAPERS_DIR

BRAIN_PATH = REPO_ROOT / "RESEARCH_BRAIN.md"
TRAJECTORIES_FILE = LOGS_DIR / "trajectories.jsonl"

def update_paper_registry(query: str = "Neural Operator", generate_models: bool = False):
    """Fetch new papers, update the registry, and optionally generate model code."""
    agent = ArXivAgent()
    agent.update_registry(query=query, generate_models=generate_models)

def _replace_section(content: str, start_marker: str, end_marker: str, new_content: str) -> str:
    """Replace content between Markdown markers."""
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        return content
    return (
        content[:start_idx + len(start_marker)] +
        "\n" + new_content.strip() + "\n" +
        content[end_idx:]
    )

def distill():
    if not BRAIN_PATH.exists():
        print(f"Error: {BRAIN_PATH} not found.")
        return

    # 1. Load Trajectories
    entries = []
    if TRAJECTORIES_FILE.exists():
        with open(TRAJECTORIES_FILE) as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not entries:
        print("No trajectories found to distill.")
        return

    # 2. Extract Learned Lessons (confirmed outcomes)
    lessons = []
    # Filter for entries with an outcome
    completed = [e for e in entries if e.get("outcome") and "completed" in e.get("action", "")]
    
    # Take the last 10 completed experiments for lesson distillation
    recent_completed = completed[-10:]
    for e in recent_completed:
        outcome_str = e.get("outcome", "")
        # Clean up outcome: "val_l2_rel=0.1858 status=keep" -> "0.1858 (keep)"
        outcome_clean = outcome_str.replace("val_l2_rel=", "").replace("status=", "(") + ")"
        
        lesson = (
            f"| {e.get('timestamp', '')[:10]} "
            f"| {e.get('critique', 'N/A')[:60]}... "
            f"| {e.get('model', 'N/A')} on {e.get('benchmark', 'N/A')} "
            f"| **{outcome_clean}** |"
        )
        lessons.append(lesson)

    # 3. Extract Active Strategy (suggested mashups/interventions)
    # Get the latest critique that suggested a "Hybrid" or "Mashup"
    mashups = []
    for e in reversed(entries):
        critique = e.get("critique", "")
        if "Hybrid" in critique or "Mashup" in critique or "Synthesis" in critique:
            mashups.append(f"- **{e.get('benchmark')} Discovery**: {critique}")
        if len(mashups) >= 3:
            break
            
    # 4. Extract Architecture Evolution
    # Look for confirmed architectural wins (status=keep + best in benchmark)
    evolutions = []
    seen_evolutions = set()
    for e in reversed(completed):
        if "keep" in e.get("outcome", "").lower():
            insight = e.get("critique", "")
            if "Architecture" in insight or "block" in insight or "Hybrid" in insight:
                summary = f"- **{e.get('model')}**: {insight.split('.')[0]}."
                if summary not in seen_evolutions:
                    evolutions.append(summary)
                    seen_evolutions.add(summary)
        if len(evolutions) >= 3:
            break

    # 5. Build New Section Strings
    # Note: Keep the header row for lessons
    new_lessons = "\n".join(lessons)
    
    # Build strategy from live DB state
    try:
        from core.results_store import store
        from core.utils import SOTA
        bests = store.best_per_benchmark()
        beaten = sorted([(bm, v) for bm, v in bests.items() if SOTA.get(bm) and v < SOTA[bm]])
        gaps = sorted(
            [(bm, v / SOTA[bm]) for bm, v in bests.items() if SOTA.get(bm) and v >= SOTA[bm]],
            key=lambda x: x[1], reverse=True
        )
        beaten_str = ", ".join(f"`{bm}`" for bm, _ in beaten[:5])
        focus_bm, focus_gap = gaps[0] if gaps else ("burgers_1d", 58)
        new_strategy = (
            f"- **Current Score**: {len(beaten)}/14 SOTA benchmarks beaten: {beaten_str}\n"
            f"- **Current Focus**: Close `{focus_bm}` ({focus_gap:.1f}× gap).\n"
        )
    except Exception:
        new_strategy = "- **Current Focus**: Closing gaps in remaining benchmarks.\n"
    if mashups:
        new_strategy += "\n".join(mashups)
    else:
        new_strategy += "- **Priority Hypotheses**: High spectral modes for shocks; EMA for convergence stability."

    new_arch_evo = "\n".join(evolutions) if evolutions else "- No recent major architectural shifts."

    # 6. Apply Updates
    with open(BRAIN_PATH, "r") as f:
        content = f.read()

    content = _replace_section(content, "<!-- LESSONS_START -->", "<!-- LESSONS_END -->", new_lessons)
    content = _replace_section(content, "<!-- STRATEGY_START -->", "<!-- STRATEGY_END -->", new_strategy)
    content = _replace_section(content, "<!-- ARCH_EVO_START -->", "<!-- ARCH_EVO_END -->", new_arch_evo)

    with open(BRAIN_PATH, "w") as f:
        f.write(content)
    
    print(f"RESEARCH_BRAIN.md distilled successfully at {datetime.now().isoformat()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SciML Brain Distiller")
    parser.add_argument("--distill", action="store_true", help="Distill experiment trajectories into RESEARCH_BRAIN.md")
    parser.add_argument("--update-papers", type=str, help="Scan ArXiv for new papers with the given query")
    parser.add_argument("--generate-models", action="store_true", help="Generate model code for new papers")
    args = parser.parse_args()

    if args.update_papers:
        update_paper_registry(args.update_papers, generate_models=args.generate_models)
    
    if args.distill or (not args.update_papers):
        distill()
