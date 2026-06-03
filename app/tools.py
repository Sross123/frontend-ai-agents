# tools.py
"""Tools for the AI Multi-Agent Studio."""

import os
from langchain_core.tools import tool

@tool
def write_to_filesystem(file_path: str, content: str) -> str:
    """Writes code content to a specific file path within the project directory.
    
    Args:
        file_path: The relative path of the file (e.g., 'index.html', 'styles.css').
        content: The complete, raw code content to write to the file.
    """
    # Safety: Ensure the path is relative to the designated generated_projects folder
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "generated_projects"))
    
    # Resolve the final absolute path
    full_path = os.path.abspath(os.path.join(base_path, file_path))
    
    # Security check: Prevent directory traversal out of generated_projects
    if not full_path.startswith(base_path):
        return "Error: Access denied. Cannot write outside the generated_projects workspace."
    
    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing to {file_path}: {str(e)}"
