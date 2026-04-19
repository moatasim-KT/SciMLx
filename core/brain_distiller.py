"""
Brain Distiller — Automated evolution of RESEARCH_BRAIN.md.

Parses trajectories.jsonl and results.json to extract scientific insights,
structural critiques, and confirmed patterns, then distills them into
the project's Living Brain.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from core.utils import REPO_ROOT, LOGS_DIR

BRAIN_PATH = REPO_ROOT / "RESEARCH_BRAIN.md"
TRAJECTORIES_FILE = LOGS_DIR / "trajectories.jsonl"

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
    
    new_strategy = "- **Current Focus**: Closing gaps in Darcy 2D and Burgers 1D.\n"
    if mashups:
        new_strategy += "\n".join(mashups)
    else:
        new_strategy += "- **Priority Hypotheses**: Stable depth ≥10 for spectral models; adaptive Sobolev weighting."

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
    distill()
