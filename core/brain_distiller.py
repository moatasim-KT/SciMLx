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

import google.generativeai as genai
from core.utils import REPO_ROOT, LOGS_DIR, PAPERS_DIR

BRAIN_PATH = REPO_ROOT / "RESEARCH_BRAIN.md"
TRAJECTORIES_FILE = LOGS_DIR / "trajectories.jsonl"

def _init_gemini():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro')

def fetch_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Fetch papers from ArXiv API."""
    base_url = "http://export.arxiv.org/api/query?"
    params = f"search_query=all:{query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    response = requests.get(base_url + params)
    if response.status_code != 200:
        return []
    
    # Simple XML parsing for entries
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.text)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
    papers = []
    for entry in root.findall('atom:entry', ns):
        papers.append({
            'id': entry.find('atom:id', ns).text.split('/')[-1],
            'title': entry.find('atom:title', ns).text.strip().replace('\n', ' '),
            'summary': entry.find('atom:summary', ns).text.strip(),
            'published': entry.find('atom:published', ns).text,
            'link': entry.find('atom:id', ns).text
        })
    return papers

def distill_paper(paper_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Use Gemini to distill a paper into a SciML implementation stub."""
    model = _init_gemini()
    if not model:
        print("Warning: GOOGLE_API_KEY not found. Skipping LLM distillation.")
        return None

    prompt = f"""
    You are a Scientific Intelligence Agent. Analyze this research paper summary and distill it into a structured YAML format for the SciMLx project.
    
    Paper Title: {paper_info['title']}
    Summary: {paper_info['summary']}
    
    Output ONLY valid YAML with these keys:
    id: (short-kebab-case-id)
    title: (full title)
    year: (int)
    model_class: (e.g., FNO, DeepONet, PINN, Transformer)
    status: pending
    benchmarks:
      benchmark_name: {{ reported_val_l2_rel: float }}
    suggested_experiments:
      - name: (experiment_name)
        rationale: (short explanation of why to run this)
        expected: (expected range of l2_rel)
    
    If the paper is not relevant to Neural Operators, PINNs, or SciML, return 'IRRELEVANT'.
    """
    
    response = model.generate_content(prompt)
    text = response.text.strip()
    
    if "IRRELEVANT" in text:
        return None
    
    # Clean up markdown code blocks if present
    text = re.sub(r"```yaml\n?|```", "", text).strip()
    
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception as e:
        print(f"Error parsing YAML from LLM: {e}")
        return None

def generate_model_code(paper_info: Dict[str, Any]) -> Optional[str]:
    """Use Gemini to generate PyTorch code for the model described in the paper."""
    model = _init_gemini()
    if not model:
        return None

    prompt = f"""
    You are a Senior SciML Engineer. Generate a production-grade PyTorch implementation of the model described in this paper.
    
    Paper Title: {paper_info['title']}
    Summary: {paper_info['summary']}
    
    The code MUST:
    1. Be a single Python file.
    2. Use 'torch' and 'torch.nn'.
    3. Implement a class that inherits from 'nn.Module'.
    4. Follow the SciMLx model interface (usually an __init__ with hyperparameters and a forward(self, u0) method).
    5. Include necessary sub-modules (e.g., SpectralConv, Attention).
    6. Be self-contained and ready to run.
    7. Include a docstring referencing the paper.
    
    Output ONLY the Python code. No explanations.
    """
    
    response = model.generate_content(prompt)
    code = response.text.strip()
    
    # Clean up markdown code blocks
    code = re.sub(r"```python\n?|```", "", code).strip()
    return code

def update_paper_registry(query: str = "Neural Operator", generate_models: bool = False):
    """Fetch new papers, update the registry, and optionally generate model code."""
    print(f"Scanning ArXiv for '{query}'...")
    papers = fetch_arxiv(query)
    new_count = 0
    
    for p in papers:
        paper_id = p['id']
        yaml_path = PAPERS_DIR / f"{paper_id}.yaml"
        if yaml_path.exists():
            continue
            
        print(f"Distilling: {p['title'][:60]}...")
        distilled = distill_paper(p)
        if distilled:
            with open(yaml_path, 'w') as f:
                import yaml
                yaml.dump(distilled, f, sort_keys=False)
            new_count += 1
            print(f"  ✓ Added to registry: {yaml_path.name}")
            
            if generate_models:
                model_name = distilled.get('id', paper_id).replace('-', '_')
                model_path = REPO_ROOT / "models" / f"{model_name}.py"
                if not model_path.exists():
                    print(f"  → Generating model code for {model_name}...")
                    code = generate_model_code(p)
                    if code:
                        model_path.write_text(code)
                        print(f"  ✓ Model created: models/{model_name}.py")
            
    print(f"Paper registry update complete. {new_count} new papers added.")

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
