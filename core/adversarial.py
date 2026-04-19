"""Adversarial Reviewer for SciML architectures.

This module provides a Proposer-Reviewer pattern for vetting new PDE solvers.
It critiques architectures for spectral bias, capacity bottlenecks, and
MLX-specific efficiency issues.
"""

import os
import json
import urllib.request
from pathlib import Path
from typing import Optional
from core.utils import REPO_ROOT, load_results

class AdversarialReviewer:
    """Uses the Gemini API to critique a proposed model architecture."""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_endpoint = "gemini-2.0-flash"
        
    def _call_gemini(self, prompt: str) -> str:
        """Call the Gemini API with fallback to agent-driven reasoning."""
        if not self.api_key:
            return f"[AGENT_REQUEST] Please critique this architecture:\n{prompt}"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_endpoint}:generateContent?key={self.api_key}"
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        req = urllib.request.Request(url, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, data=json.dumps(data).encode("utf-8"), timeout=15) as f:
                resp = json.loads(f.read().decode("utf-8"))
                return resp["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            # Fall back to agent request on any API error (429, 404, etc)
            return f"[AGENT_REQUEST] (API Error: {e}) Please critique this architecture:\n{prompt}"


    def critique(self, model_code: str, rationale: str, benchmark: str) -> str:
        """Provide a detailed critique of a proposed architecture."""
        # 1. Gather context
        results = load_results(benchmark)
        # Filter for 'keep' status and valid score
        valid_results = [r["val_l2_rel"] for r in results if r.get("status") == "keep" and not (isinstance(r["val_l2_rel"], float) and (r["val_l2_rel"] is float("nan") or r["val_l2_rel"] > 1e6))]
        best_val = min(valid_results, default=1.0)
        
        # Load trajectories (last 5) for history awareness
        traj_path = REPO_ROOT / "logs" / "trajectories.jsonl"
        trajectories = []
        if traj_path.exists():
            try:
                with open(traj_path) as f:
                    lines = f.readlines()[-5:]
                    trajectories = [json.loads(l) for l in lines]
            except Exception:
                pass
        
        # 2. Build prompt
        prompt = f"""
You are a Senior SciML Architect specializing in Neural PDEs on Apple Silicon (MLX).
Critique the following architecture proposal for the benchmark: {benchmark}.

Current best val_l2_rel on this benchmark: {best_val:.6f}

--- Rationale ---
{rationale}

--- Model Implementation ---
{model_code}

--- Recent Trajectories ---
{json.dumps(trajectories, indent=2)}

--- Instructions ---
Provide a concise, skeptical review (max 200 words). 
1. Identify 2-3 specific architectural risks (e.g., spectral bias, lack of residuals, channel bottlenecks).
2. Check for MLX-specific issues (e.g., full attention is forbidden for N>64, use windowed/linear instead).
3. Veto recommendation: Should we RUN or REVISE this?
"""
        return self._call_gemini(prompt)

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Critique a SciML model")
    parser.add_argument("model_file", help="Path to the model source file")
    parser.add_argument("rationale", help="The rationale for this architecture")
    parser.add_argument("benchmark", help="The benchmark target")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_file):
        print(f"Error: File not found: {args.model_file}")
        sys.exit(1)
        
    reviewer = AdversarialReviewer()
    with open(args.model_file) as f:
        code = f.read()
    
    print("\n" + "="*70)
    print(f"  Adversarial Review for {args.benchmark}")
    print("="*70 + "\n")
    print(reviewer.critique(code, args.rationale, args.benchmark))
    print("\n" + "="*70)
