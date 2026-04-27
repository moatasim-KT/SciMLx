import pytest
import textwrap
from unittest.mock import MagicMock, patch
from core.arxiv_agent import ArXivAgent

def test_arxiv_agent_init():
    agent = ArXivAgent(api_key="test_key")
    assert agent.api_key == "test_key"
    assert agent._model is not None

@patch('requests.get')
def test_arxiv_agent_search(mock_get):
    # Mock ArXiv XML response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = textwrap.dedent("""
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <id>http://arxiv.org/abs/2401.00001v1</id>
            <title>Test Paper</title>
            <summary>This is a test summary.</summary>
            <published>2024-01-01T00:00:00Z</published>
        </entry>
    </feed>
    """)
    mock_get.return_value = mock_response
    
    agent = ArXivAgent()
    results = agent.search("test query", max_results=1)
    
    assert len(results) == 1
    assert results[0]['title'] == "Test Paper"
    assert results[0]['id'] == "2401.00001v1"

@patch('google.generativeai.GenerativeModel')
def test_arxiv_agent_distill(mock_model_class):
    mock_model = MagicMock()
    mock_model_class.return_value = mock_model
    
    # Mock Gemini response
    mock_response = MagicMock()
    mock_response.text = textwrap.dedent("""
        id: test-paper
        title: Test Paper
        year: 2024
        model_class: FNO
        status: pending
    """)
    mock_model.generate_content.return_value = mock_response
    
    agent = ArXivAgent(api_key="test_key")
    agent._model = mock_model # Ensure it uses our mock
    
    paper_info = {
        'id': '123',
        'title': 'Test Paper',
        'summary': 'Summary'
    }
    
    result = agent.distill(paper_info)
    assert result is not None
    assert result['id'] == 'test-paper'
    assert result['model_class'] == 'FNO'

@patch('google.generativeai.GenerativeModel')
def test_arxiv_agent_generate_model_code(mock_model_class):
    mock_model = MagicMock()
    mock_model_class.return_value = mock_model
    
    mock_response = MagicMock()
    mock_response.text = "class TestModel(nn.Module): pass"
    mock_model.generate_content.return_value = mock_response
    
    agent = ArXivAgent(api_key="test_key")
    agent._model = mock_model
    
    paper_info = {'id': '123', 'title': 'Test Paper', 'summary': 'Summary'}
    code = agent.generate_model_code(paper_info, framework='torch')
    
    assert "class TestModel" in code

def test_arxiv_agent_get_proposals(tmp_path):
    from core import arxiv_agent
    import yaml
    
    # Mock PAPERS_DIR
    original_papers_dir = arxiv_agent.PAPERS_DIR
    arxiv_agent.PAPERS_DIR = tmp_path
    
    try:
        # Create a mock paper yaml
        paper_data = {
            'id': 'test-paper',
            'suggested_experiments': [
                {
                    'name': 'exp1',
                    'benchmark': 'burgers_1d',
                    'model': 'FNO',
                    'hidden_dim': 64,
                    'n_layers': 4,
                    'n_modes': 16,
                    'rationale': 'test'
                }
            ]
        }
        with open(tmp_path / "test-paper.yaml", 'w') as f:
            yaml.dump(paper_data, f)
            
        agent = ArXivAgent()
        proposals = agent.get_proposals()
        
        assert len(proposals) == 1
        assert proposals[0]['name'] == 'exp1'
        assert proposals[0]['benchmark'] == 'burgers_1d'
        assert proposals[0]['paper_ref'] == 'test-paper'
    finally:
        arxiv_agent.PAPERS_DIR = original_papers_dir
