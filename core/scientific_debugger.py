"""Scientific Debugger for SciML failures.

This module implements the /debug protocol:
1. Load last safe checkpoint.
2. Run a 5-step probe with high-fidelity logging.
3. Analyze layer-wise stats to pinpoint explosion.
4. Propose a specific fix (e.g., 'reduce gating initialization' or 'add LayerNorm').
"""

import os
import json
import math
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any
from core.utils import REPO_ROOT, PROBE_LOG_DIR

class ScientificDebugger:
    """Diagnoses numerical failures (NaNs/Infs) using high-fidelity probe data."""
    
    def __init__(self, exp_name: str, benchmark: str):
        self.exp_name = exp_name
        self.benchmark = benchmark
        self._slug = exp_name.replace("/", "_").replace(" ", "_")
        self.probe_path = PROBE_LOG_DIR / f"probe_{self._slug}.jsonl"
        self.api_key = os.environ.get("GEMINI_API_KEY")

    def analyze_probe_log(self) -> Dict[str, Any]:
        """Analyze the probe log to pinpoint the layer where gradients exploded."""
        if not self.probe_path.exists():
            return {"error": "Probe log not found."}
            
        history = []
        with open(self.probe_path) as f:
            for line in f:
                try:
                    history.append(json.loads(line))
                except Exception:
                    continue
            
        if not history:
            return {"error": "Probe log is empty."}
            
        # 1. Detect where it went wrong
        offending_layer = None
        failure_step = None
        max_norm = 0.0
        
        for step_data in history:
            loss = step_data.get("loss", 0)
            if math.isnan(loss) or math.isinf(loss):
                failure_step = step_data["step"]
                # Look at the *previous* step to see the state before explosion
                break
            
            for layer, stats in step_data.get("layers", {}).items():
                norm = stats.get("grad_norm", 0.0)
                if norm > max_norm:
                    max_norm = norm
                    offending_layer = layer
                if norm > 1e10: # Extreme explosion
                    failure_step = step_data["step"]
                    offending_layer = layer
                    break
            if failure_step: break

        # If we didn't hit a NaN but the loop ended, take the last layer with highest norm
        if failure_step is None:
            failure_step = history[-1]["step"]

        return {
            "exp_name": self.exp_name,
            "benchmark": self.benchmark,
            "failure_step": failure_step,
            "offending_layer": offending_layer,
            "max_norm": max_norm,
            "history_summary": history[-3:] # Last 3 steps for context
        }

    def get_scientific_fix(self, analysis: Dict[str, Any]) -> str:
        """Propose a scientific fix using LLM reasoning (external API or local agent)."""
        prompt = f"""
        You are a Senior SciML Debugger. A Neural PDE solver failed with NaNs/Infs.
        Analyze this probe data and propose a scientific fix.

        --- Failure Context ---
        Experiment: {analysis['exp_name']}
        Benchmark: {analysis['benchmark']}
        Failure Step: {analysis['failure_step']}
        Offending Layer: {analysis['offending_layer']}
        Max Observed Grad Norm: {analysis['max_norm']:.2e}

        --- Last 3 Probe Steps ---
        {json.dumps(analysis['history_summary'], indent=2)}

        --- Instructions ---
        1. Identify the likely root cause (e.g., spectral instability, vanishing residuals, un-normalized activations).
        2. Propose a SPECIFIC code fix (e.g., "Add LayerNorm before {analysis['offending_layer']}" or "Initialize gating weights in {analysis['offending_layer']} to 1e-4").
        3. Keep it under 150 words.
        """
        prompt = textwrap.dedent(prompt)
        
        if not self.api_key:
            return f"[AGENT_REQUEST] Please diagnose this failure:\n{prompt}"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
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
            return f"[AGENT_REQUEST] (API Error: {e}) Please diagnose this failure:\n{prompt}"

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 core/scientific_debugger.py <exp_name> <benchmark>")
        sys.exit(1)
        
    debugger = ScientificDebugger(sys.argv[1], sys.argv[2])
    analysis = debugger.analyze_probe_log()
    if "error" in analysis:
        print(analysis["error"])
    else:
        print("\n" + "="*70)
        print(f"  Scientific Diagnosis for {sys.argv[1]}")
        print("="*70 + "\n")
        print(debugger.get_scientific_fix(analysis))
        print("\n" + "="*70)
