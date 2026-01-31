"""
Prompt loading utilities for CRISP-T MCP Server.
"""

from pathlib import Path


def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts directory.
    
    Args:
        name: Name of the prompt (without .txt extension)
        
    Returns:
        The prompt text content
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    prompt_file = Path(__file__).parent / f"{name}.txt"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt '{name}' not found at {prompt_file}")
    
    return prompt_file.read_text()


# Pre-load commonly used prompts
ANALYSIS_WORKFLOW = load_prompt("analysis_workflow")
TRIANGULATION_GUIDE = load_prompt("triangulation_guide")
