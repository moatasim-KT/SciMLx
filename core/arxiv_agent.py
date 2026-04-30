"""
ArXiv Agent — Refactored ArXiv integration for the ASIL pipeline.

Handles searching, distilling papers into structured metadata, and 
generating framework-native model code (Torch or MLX) using LLM synthesis.
"""

import os
import re
import requests
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import google.generativeai as genai
from core.utils import REPO_ROOT, PAPERS_DIR
from core.device import FRAMEWORK

class ArXivAgent:
    """
    ArXivAgent — Refactored ArXiv integration for the ASIL pipeline.
    Handles searching, distilling papers, and generating framework-native model code.
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the ArXivAgent with an optional Google API Key.
        Defaults to GOOGLE_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._model = None
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel('gemini-1.5-pro')

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch papers from ArXiv API matching the query.
        
        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            
        Returns:
            List of dictionaries containing paper metadata (id, title, summary, published, link).
        """
        base_url = "http://export.arxiv.org/api/query?"
        params = f"search_query=all:{query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
        try:
            response = requests.get(base_url + params, timeout=15)
            if response.status_code != 200:
                print(f"ArXiv API error: {response.status_code}")
                return []
            
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
        except Exception as e:
            print(f"Error searching ArXiv: {e}")
            return []

    def distill(self, paper_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Use Gemini to distill a paper summary into a structured YAML format.
        
        Args:
            paper_info: Dictionary containing paper metadata.
            
        Returns:
            Dictionary (YAML-compatible) or None if paper is irrelevant or distillation fails.
        """
        if not self._model:
            print("Warning: ArXivAgent initialized without GOOGLE_API_KEY. Distillation disabled.")
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
          - name: (unique_experiment_name)
            benchmark: (benchmark_key, e.g., burgers_1d, ns_2d)
            model: (model_key, e.g., FNO, TFNO)
            hidden_dim: (int)
            n_layers: (int)
            n_modes: (int)
            rationale: (short explanation of why to run this)
        
        If the paper is not relevant to Neural Operators, PINNs, or SciML, return 'IRRELEVANT'.
        """
        
        try:
            response = self._model.generate_content(prompt)
            text = response.text.strip()
            
            if "IRRELEVANT" in text:
                return None
            
            # Clean up markdown code blocks if the LLM wrapped it
            text = re.sub(r"```yaml\n?|```", "", text).strip()
            
            # Load YAML
            import yaml
            return yaml.safe_load(text)
        except Exception as e:
            print(f"Error distilling paper {paper_info['id']}: {e}")
            return None

    def get_proposals(self) -> List[Dict[str, Any]]:
        """
        Read distilled papers from registry and return a list of proposed experiments.
        
        Returns:
            List of dictionaries compatible with ExperimentConfig.
        """
        proposals = []
        if not PAPERS_DIR.exists():
            return []
            
        for yaml_path in PAPERS_DIR.glob("*.yaml"):
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    paper_id = data.get('id', yaml_path.stem)
                    
                    for exp in data.get('suggested_experiments', []):
                        # Ensure basic fields exist
                        if all(k in exp for k in ('benchmark', 'model', 'hidden_dim', 'n_layers')):
                            proposals.append({
                                'name': exp.get('name'),
                                'benchmark': exp.get('benchmark'),
                                'model': exp.get('model'),
                                'hidden_dim': int(exp.get('hidden_dim', 64)),
                                'n_layers': int(exp.get('n_layers', 4)),
                                'n_modes': int(exp.get('n_modes', 16)),
                                'rationale': exp.get('rationale'),
                                'paper_ref': paper_id,
                                'source': 'arxiv'
                            })
            except Exception as e:
                print(f"Error reading {yaml_path}: {e}")
        return proposals

    def generate_model_code(self, paper_info: Dict[str, Any], framework: str = FRAMEWORK) -> Optional[str]:
        """
        Use Gemini to generate framework-native code for the model described in the paper.
        
        Args:
            paper_info: Dictionary containing paper metadata.
            framework: Target framework ('torch' or 'mlx').
            
        Returns:
            String containing the Python code or None if generation fails.
        """
        if not self._model:
            return None

        prompt = f"""
        You are a Senior SciML Engineer. Generate a production-grade {framework.upper()} implementation of the model described in this paper.
        
        Paper Title: {paper_info['title']}
        Summary: {paper_info['summary']}
        
        The code MUST:
        1. Be a single Python file.
        2. Use '{framework}' and relevant libraries ('torch.nn' or 'mlx.nn' as mx_nn).
        3. Implement a class that inherits from '{'nn.Module' if framework == 'torch' else 'mx_nn.Module'}'.
        4. Follow the SciMLx model interface: __init__ with hyperparameters, and forward(self, x) method.
        5. Be self-contained and ready to run.
        6. Include a docstring referencing the paper.
        
        Output ONLY the Python code. No explanations.
        """
        
        try:
            response = self._model.generate_content(prompt)
            code = response.text.strip()
            # Clean up markdown code blocks
            code = re.sub(r"```python\n?|```", "", code).strip()
            return code
        except Exception as e:
            print(f"Error generating model code for {paper_info['id']}: {e}")
            return None

    def update_registry(self, query: str = "Neural Operator", generate_models: bool = False) -> int:
        """
        High-level method to sync ArXiv findings with the local paper registry.
        
        Args:
            query: ArXiv search query.
            generate_models: If True, also attempts to synthesize code for new models.
            
        Returns:
            Number of new papers added to the registry.
        """
        print(f"[ArXivAgent] Syncing registry for: {query}...")
        papers = self.search(query)
        new_count = 0
        
        for p in papers:
            paper_id = p['id']
            yaml_path = PAPERS_DIR / f"{paper_id}.yaml"
            if yaml_path.exists():
                continue
                
            print(f"[ArXivAgent] Distilling: {p['title'][:60]}...")
            distilled = self.distill(p)
            if distilled:
                PAPERS_DIR.mkdir(parents=True, exist_ok=True)
                with open(yaml_path, 'w') as f:
                    yaml.dump(distilled, f, sort_keys=False)
                new_count += 1
                print(f"  ✓ Added to registry: {yaml_path.name}")
                
                if generate_models:
                    model_name = distilled.get('id', paper_id).replace('-', '_')
                    model_path = REPO_ROOT / "models" / f"{model_name}.py"
                    if not model_path.exists():
                        print(f"  → Generating {FRAMEWORK} code for {model_name}...")
                        code = self.generate_model_code(p)
                        if code:
                            model_path.write_text(code)
                            print(f"  ✓ Model created: models/{model_name}.py")
                            
        return new_count
