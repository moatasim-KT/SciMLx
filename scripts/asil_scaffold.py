#!/usr/bin/env python3
"""
ASIL Scaffolding Bridge
Automates the transition from research proposals to functional model code and experiments.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

# Mock torch if missing to allow ModelGate to at least load
try:
    import torch
except ImportError:
    from unittest.mock import MagicMock
    m = MagicMock()
    sys.modules["torch"] = m
    sys.modules["torch.nn"] = m
    sys.modules["torch.nn.functional"] = m
    print("Note: torch missing, using mocks for validation/registration.")

from core.scaffold import ModelGate, generate_stub
from core.utils import REPO_ROOT
from core.device import FRAMEWORK

def parse_proposal(proposal_path: Path):
    if not proposal_path.exists():
        print(f"Error: Proposal file not found at {proposal_path}")
        sys.exit(1)

    content = proposal_path.read_text()

    # Parse frontmatter
    fm_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL | re.MULTILINE)
    if not fm_match:
        print("Error: Could not find frontmatter in proposal.")
        sys.exit(1)
    
    fm = yaml.safe_load(fm_match.group(1))
    title = fm.get('title', 'Unknown Title')
    target_pde = fm.get('target_pde', 'burgers_1d')
    novelty_score = fm.get('novelty_score', 0)

    # Parse Implementation Specs
    specs_match = re.search(r'## Implementation Specs.*', content, re.DOTALL)
    if not specs_match:
        print("Error: Could not find '## Implementation Specs' in proposal.")
        sys.exit(1)
    
    specs = specs_match.group(0)
    registry_key_match = re.search(r'Registry Key\*\*:\s*`?(\w+)`?', specs)
    if not registry_key_match:
        print("Error: Could not find 'Registry Key' in Implementation Specs.")
        sys.exit(1)
    registry_key = registry_key_match.group(1)

    hard_limits_match = re.search(r'Hard Limits\*\*:\s*(.*)', specs)
    hard_limits = hard_limits_match.group(1).strip() if hard_limits_match else ""

    return {
        "title": title,
        "target_pde": target_pde,
        "novelty_score": novelty_score,
        "registry_key": registry_key,
        "hard_limits": hard_limits
    }

def update_research_brain(proposal_data: dict):
    brain_path = REPO_ROOT / "RESEARCH_BRAIN.md"
    if not brain_path.exists():
        print("Warning: RESEARCH_BRAIN.md not found. Skipping update.")
        return

    content = brain_path.read_text()
    
    header = "## 11. Hypothesis Tracking"
    if header not in content:
        content += f"\n\n{header}\n\n"
        content += "| Date | Proposal Title | Registry Key | Target PDE | Novelty | Status |\n"
        content += "|---|---|---|---|---|---|\n"
    
    import datetime
    date_str = datetime.date.today().isoformat()
    new_entry = f"| {date_str} | {proposal_data['title']} | {proposal_data['registry_key']} | {proposal_data['target_pde']} | {proposal_data['novelty_score']} | Scaffolded |\n"
    
    content += new_entry
    brain_path.write_text(content)
    print(f"Updated RESEARCH_BRAIN.md with hypothesis: {proposal_data['title']}")

def main():
    parser = argparse.ArgumentParser(description="Scaffold a model from a research proposal.")
    parser.add_argument("--proposal", required=True, help="Path to the proposal markdown file.")
    args = parser.parse_args()

    proposal_path = Path(args.proposal)
    data = parse_proposal(proposal_path)

    print(f"Scaffolding model '{data['registry_key']}' for proposal '{data['title']}' using {FRAMEWORK.upper()} backend...")

    # 1. Generate stubs
    is_2d = "_2d" in data['target_pde'].lower() or "2d" in data['target_pde'].lower()
    stubs = generate_stub(data['registry_key'], notes=data['hard_limits'], two_d=is_2d)
    
    gate = ModelGate()
    model_files = {}
    for fw, code in stubs.items():
        model_file = REPO_ROOT / "models" / f"{data['registry_key'].lower()}_{fw}.py"
        model_file.write_text(code)
        model_files[fw] = model_file
        print(f"Generated {fw} stub at {model_file}")

    # 2. Validate both, register once
    # Mock torch if missing to allow ModelGate to at least load
    try:
        import torch
    except ImportError:
        from unittest.mock import MagicMock
        m = MagicMock()
        sys.modules["torch"] = m
        sys.modules["torch.nn"] = m
        sys.modules["torch.nn.functional"] = m
        print("Note: torch missing, using mocks for validation/registration.")

    for fw, model_file in model_files.items():
        ok, report = gate.validate(data['registry_key'], str(model_file))
        status = "PASSED" if ok else "FAILED"
        print(f"Validation {status} for {fw} backend: {report.get('warning', report.get('error', 'OK'))}")

    # Register using the one that matches current framework (dynamic backend support handles the rest)
    preferred_fw = FRAMEWORK if FRAMEWORK in model_files else "torch"
    preferred_file = model_files[preferred_fw]
    gate.register_and_queue(data['registry_key'], str(preferred_file), benchmarks=[data['target_pde']])
    print(f"Registered {data['registry_key']} in registry and experiments.yaml (Dynamic Dual-Backend)")

    # 3. Update Brain
    update_research_brain(data)

if __name__ == "__main__":
    main()
