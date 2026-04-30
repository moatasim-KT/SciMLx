"""
ASIL Ideate — The Synthesis Engine of the ASIL pipeline.

Autonomously cross-references RESEARCH_BRAIN.md with ArXiv/SOTA findings
to generate a novel Research Proposal in docs/proposals/.
"""

import argparse
import os
import re
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from core.arxiv_agent import ArXivAgent
from core.brain_distiller import BRAIN_PATH
from core.utils import REPO_ROOT, PAPERS_DIR, SOTA

PROPOSALS_DIR = REPO_ROOT / "docs" / "proposals"
TEMPLATE_PATH = PROPOSALS_DIR / "TEMPLATE.md"

def read_brain_gaps() -> str:
    """Read RESEARCH_BRAIN.md to extract current SOTA gaps and strategy."""
    if not BRAIN_PATH.exists():
        return "No RESEARCH_BRAIN.md found."
    
    content = BRAIN_PATH.read_text()
    
    # Extract Strategy and Roadmap sections
    strategy_match = re.search(r"<!-- STRATEGY_START -->(.*?)<!-- STRATEGY_END -->", content, re.DOTALL)
    strategy = strategy_match.group(1).strip() if strategy_match else "Strategy not found."
    
    # Extract the Roadmap table
    roadmap_match = re.search(r"## 9. Roadmap & SOTA Gaps(.*?)(?=\n##|$)", content, re.DOTALL)
    roadmap = roadmap_match.group(1).strip() if roadmap_match else "Roadmap not found."
    
    return f"### Current Strategy\n{strategy}\n\n### SOTA Gaps\n{roadmap}"

def _init_gemini():
    """Helper for tests and internal initialization."""
    agent = ArXivAgent()
    return agent._model

def synthesize_proposal(keywords: List[str], novelty: str, limit: int) -> str:
    """Use Gemini to synthesize a novel research proposal."""
    model = _init_gemini()
    if not model:
        raise RuntimeError("GOOGLE_API_KEY not found. LLM synthesis required.")

    agent = ArXivAgent()

    # 1. Update paper registry (Sync ArXiv)
    query = " ".join(keywords)
    print(f"Syncing ArXiv registry for: {query}...")
    agent.update_registry(query=query)

    # 2. Fetch relevant paper summaries for context
    # We use agent.search directly to get the summaries for the prompt
    papers = agent.search(query, max_results=limit)
    paper_context = "\n".join([f"- {p['title']} ({p['published'][:4]}): {p['summary'][:800]}..." for p in papers])

    # 3. Get project context from RESEARCH_BRAIN.md
    brain_context = read_brain_gaps()
    
    # 4. Read template
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Proposal template not found at {TEMPLATE_PATH}")
    template = TEMPLATE_PATH.read_text()

    prompt = f"""
    You are a Senior SciML Research Scientist. Your goal is to generate a novel Research Proposal for the SciMLx project.
    
    PROJECT REALITY (RESEARCH_BRAIN.md):
    {brain_context}
    
    LITERATURE CONTEXT (ArXiv):
    {paper_context}
    
    NOVELTY LEVEL: {novelty}
    - Low: Incremental improvement on existing architectures (e.g., adding EMA, tuning loss).
    - Medium: Architectural hybridization (e.g., FNO + Mamba, TFNO + KAN).
    - High: Paradigm shift or Novel PDE foundation (e.g., New spectral basis, Physics-Informed Latent SDEs).
    
    TEMPLATE TO FOLLOW:
    {template}
    
    INSTRUCTIONS:
    1. Cross-reference the ArXiv findings with the specific SOTA gaps in the project. 
       Look for benchmarks with large gaps (e.g., burgers_1d 58x gap, poisson_2d 4702x gap).
    2. Propose a NOVEL architecture or loss function that addresses a high-priority gap.
    3. If novelty is 'High', synthesize a hybrid of at least two distinct concepts.
    4. Ensure the proposal is technically grounded and specifically tailored for Apple Silicon (MLX) constraints.
       - 2D Hard Limits: hidden_dim < 64, n_layers < 8.
       - Model registry keys must be unique.
    5. Output the full Markdown content of the proposal, filling all placeholders like PROPOSAL_TITLE, TARGET_PDE_BENCHMARK, etc.
    6. Ensure the date in the YAML frontmatter and the content is updated to today's date (if applicable).
    7. Mathematical Rationale: Provide a brief but rigorous explanation of why the proposed changes improve the model's ability to solve the PDE.
    
    Output ONLY the final Markdown content. No conversational filler.
    """

    print(f"Synthesizing {novelty}-novelty proposal using Gemini 1.5 Pro...")
    response = model.generate_content(prompt)
    proposal_content = response.text.strip()
    
    # Clean up markdown code blocks if the LLM wrapped it
    proposal_content = re.sub(r"^```markdown\n?|```$", "", proposal_content, flags=re.MULTILINE).strip()
    
    return proposal_content

def save_proposal(content: str) -> Path:
    """Save the proposal with a slug-based filename."""
    # Extract title to generate slug
    title_match = re.search(r"title: \"(.*?)\"", content)
    if not title_match:
        title_match = re.search(r"# Research Proposal: (.*)", content)
    
    title = title_match.group(1) if title_match else "novel-research-proposal"
    slug = re.sub(r"[^\w\s-]", "", title).strip().lower()
    slug = re.sub(r"[-\s]+", "-", slug)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}-{slug}.md"
    file_path = PROPOSALS_DIR / filename
    
    # Update placeholders
    content = content.replace("YYYY-MM-DD", date_str)
    
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASIL Ideate — Research Synthesis Engine")
    parser.add_argument("--keywords", type=str, required=True, help="Comma-separated research keywords")
    parser.add_argument("--novelty", choices=["low", "medium", "high"], default="medium", help="Degree of architectural hybridization")
    parser.add_argument("--limit", type=int, default=5, help="Number of ArXiv papers to consider")
    
    args = parser.parse_args()
    keywords = [k.strip() for k in args.keywords.split(",")]
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        exit(1)

    try:
        proposal = synthesize_proposal(keywords, args.novelty, args.limit)
        saved_path = save_proposal(proposal)
        print(f"\n{'='*60}")
        print(f"SUCCESS: Novel Research Proposal Generated")
        print(f"Path: {saved_path}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"Error during ideation: {e}")
        exit(1)
