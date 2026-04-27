import os
import sys
import shutil
import tempfile
import yaml
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to sys.path
sys.path.insert(0, "/home/moatasimfarooque/SciMLx")

from core.utils import REPO_ROOT
import scripts.asil_ideate as asil_ideate
import scripts.asil_scaffold as asil_scaffold

@pytest.fixture
def mock_asil_env(monkeypatch):
    """Set up a temporary ASIL environment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # 1. Setup Directory Structure
        (tmp_path / "docs" / "proposals").mkdir(parents=True)
        (tmp_path / "models").mkdir(parents=True)
        (tmp_path / "core").mkdir(parents=True)
        (tmp_path / "logs").mkdir(parents=True)
        
        # 2. Create Required Files
        brain_path = tmp_path / "RESEARCH_BRAIN.md"
        brain_path.write_text("<!-- STRATEGY_START -->Test strategy<!-- STRATEGY_END -->\n## 9. Roadmap & SOTA Gaps\n| Gap | Priority |\n|---|---|\n| burgers_1d | High |")
        
        experiments_path = tmp_path / "experiments.yaml"
        experiments_path.write_text("[]")
        
        template_path = tmp_path / "docs" / "proposals" / "TEMPLATE.md"
        # Copy original template or use a simple one
        orig_template = (REPO_ROOT / "docs" / "proposals" / "TEMPLATE.md").read_text()
        template_path.write_text(orig_template)
        
        plugins_path = tmp_path / "core" / "research_plugins.py"
        orig_plugins = (REPO_ROOT / "core" / "research_plugins.py").read_text()
        plugins_path.write_text(orig_plugins)

        # 3. Monkeypatch Paths in relevant modules
        for module in [asil_ideate, asil_scaffold, sys.modules["core.utils"], sys.modules["core.scaffold"]]:
            if hasattr(module, "REPO_ROOT"):
                monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
            if hasattr(module, "PROPOSALS_DIR"):
                monkeypatch.setattr(module, "PROPOSALS_DIR", tmp_path / "docs" / "proposals")
            if hasattr(module, "TEMPLATE_PATH"):
                monkeypatch.setattr(module, "TEMPLATE_PATH", tmp_path / "docs" / "proposals" / "TEMPLATE.md")
            if hasattr(module, "BRAIN_PATH"):
                monkeypatch.setattr(module, "BRAIN_PATH", tmp_path / "RESEARCH_BRAIN.md")

        # Mock results.json path in core.utils
        monkeypatch.setattr("core.utils.RESULTS_FILE", tmp_path / "results.json")

        yield tmp_path

def test_asil_end_to_end_loop(mock_asil_env, monkeypatch):
    tmp_path = mock_asil_env
    
    # --- PHASE 1: ASIL IDEATE ---
    
    # Mock LLM and ArXiv
    mock_model = MagicMock()
    mock_response = MagicMock()
    proposal_content = """---
title: "Hybrid Mamba FNO"
date: "YYYY-MM-DD"
author: "AgenticScientist"
target_pde: "burgers_1d"
novelty_score: 8
estimated_complexity: "MEDIUM"
tags: ["Architecture", "Hybridization"]
---

# Research Proposal: Hybrid Mamba FNO

## Implementation Specs
**Registry Key**: `HybridMambaFNO`
**Hard Limits**: hidden_dim < 64, n_layers < 8
"""
    mock_response.text = proposal_content
    mock_model.generate_content.return_value = mock_response
    
    monkeypatch.setattr("scripts.asil_ideate._init_gemini", lambda: mock_model)
    monkeypatch.setattr("core.arxiv_agent.ArXivAgent.search", lambda self, q, max_results: [])
    monkeypatch.setattr("core.arxiv_agent.ArXivAgent.update_registry", lambda self, query: None)
    
    # Run Ideate
    keywords = ["Mamba", "FNO", "Burgers"]
    proposal = asil_ideate.synthesize_proposal(keywords, "medium", 5)
    saved_path = asil_ideate.save_proposal(proposal)
    
    assert saved_path.exists()
    assert "Hybrid Mamba FNO" in saved_path.read_text()
    assert "HybridMambaFNO" in saved_path.read_text()
    
    # --- PHASE 2: ASIL SCAFFOLD ---
    
    # Mock sys.argv for asil_scaffold
    monkeypatch.setattr(sys, "argv", ["asil_scaffold.py", "--proposal", str(saved_path)])
    
    # Run Scaffold main
    asil_scaffold.main()
    
    # --- ASSERTIONS ---
    
    # 1. Model file creation (Dual backend files created)
    torch_file = tmp_path / "models" / "hybridmambafno_torch.py"
    mlx_file = tmp_path / "models" / "hybridmambafno_mlx.py"
    assert torch_file.exists(), "Torch model stub should have been created"
    assert mlx_file.exists(), "MLX model stub should have been created"
    assert "class HybridMambaFNO" in torch_file.read_text()
    
    # 2. Registration in research_plugins.py
    plugins_content = (tmp_path / "core" / "research_plugins.py").read_text()
    # Refactored uses lazy registration with format string for FRAMEWORK
    # Note: we check for the presence of the registry key and the lazy registration pattern
    assert 'MODEL_REGISTRY.register_lazy("HybridMambaFNO"' in plugins_content
    assert '{FRAMEWORK.lower()}' in plugins_content
    
    # 3. Queue in experiments.yaml
    with open(tmp_path / "experiments.yaml", "r") as f:
        experiments = yaml.safe_load(f)
    
    assert len(experiments) > 0
    assert experiments[-1]["model"] == "HybridMambaFNO"
    assert experiments[-1]["benchmark"] == "burgers_1d"
    
    # 4. Update in RESEARCH_BRAIN.md
    brain_content = (tmp_path / "RESEARCH_BRAIN.md").read_text()
    assert "## 11. Hypothesis Tracking" in brain_content
    assert "Hybrid Mamba FNO" in brain_content
    assert "HybridMambaFNO" in brain_content
    assert "Scaffolded" in brain_content

if __name__ == "__main__":
    # If run directly, use pytest
    pytest.main([__file__])
