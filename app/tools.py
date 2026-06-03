# tools.py
"""Filesystem tools for the AI Multi-Agent Studio.

Provides secure, logging-aware file operations with path traversal protection.
"""

import os
import logging
from typing import Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


class FileSystemSecurityError(Exception):
    """Raised when a file operation violates security constraints."""
    pass


def validate_workspace_path(base_dir: str, target_path: str) -> str:
    """Validate that target_path is within base_dir (prevents directory traversal).
    
    Args:
        base_dir: The allowed base directory (absolute path).
        target_path: The target path to validate (may be relative or absolute).
        
    Returns:
        Absolute path if valid.
        
    Raises:
        FileSystemSecurityError: If path traversal is attempted.
    """
    base_dir = os.path.abspath(base_dir)
    full_path = os.path.normpath(os.path.join(base_dir, target_path))
    
    if not full_path.startswith(base_dir):
        logger.warning(f"Path traversal attempt blocked: {target_path} from {base_dir}")
        raise FileSystemSecurityError(
            f"Access denied: Path '{target_path}' attempts to escape workspace."
        )
    
    return full_path


@tool
async def write_to_file(path: str, content: str, workspace_dir: str = "./workspace") -> str:
    """Writes code to a file in the workspace.
    
    Args:
        path: The relative path of the file (e.g., 'index.html').
        content: The complete raw code content to write.
        workspace_dir: The workspace base directory (default: './workspace').
        
    Returns:
        Success or error message.
    """
    try:
        base_dir = os.path.abspath(workspace_dir)
        full_path = validate_workspace_path(base_dir, path)
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Written {len(content)} bytes to {path}")
        return f"File {path} created/updated successfully."
    except FileSystemSecurityError as e:
        logger.error(f"Security violation: {e}")
        return f"Error: {str(e)}"
    except IOError as e:
        logger.error(f"IO error writing {path}: {e}")
        return f"Error writing to {path}: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error writing {path}: {e}")
        return f"Error writing to {path}: {str(e)}"


@tool
async def read_file(path: str, workspace_dir: str = "./workspace") -> str:
    """Reads the content of a file in the workspace.
    
    Args:
        path: The relative path of the file (e.g., 'index.html').
        workspace_dir: The workspace base directory (default: './workspace').
        
    Returns:
        File contents or error message.
    """
    try:
        base_dir = os.path.abspath(workspace_dir)
        full_path = validate_workspace_path(base_dir, path)
        
        if not os.path.exists(full_path):
            logger.warning(f"File not found: {path}")
            return f"Error: File {path} does not exist."
        
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        logger.info(f"Read {len(content)} bytes from {path}")
        return content
    except FileSystemSecurityError as e:
        logger.error(f"Security violation: {e}")
        return f"Error: {str(e)}"
    except IOError as e:
        logger.error(f"IO error reading {path}: {e}")
        return f"Error reading {path}: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error reading {path}: {e}")
        return f"Error reading {path}: {str(e)}"


@tool
async def list_files(directory: str = ".", workspace_dir: str = "./workspace") -> list[str]:
    """Lists files and directories in the workspace.
    
    Args:
        directory: The sub-directory to list (default: '.').
        workspace_dir: The workspace base directory (default: './workspace').
        
    Returns:
        List of filenames or error message wrapped in a list.
    """
    try:
        base_dir = os.path.abspath(workspace_dir)
        target_dir = validate_workspace_path(base_dir, directory)
        
        if not os.path.exists(target_dir):
            logger.warning(f"Directory not found: {directory}")
            return []
        
        items = os.listdir(target_dir)
        logger.info(f"Listed {len(items)} items in {directory}")
        return items
    except FileSystemSecurityError as e:
        logger.error(f"Security violation: {e}")
        return [f"Error: {str(e)}"]
    except Exception as e:
        logger.error(f"Error listing {directory}: {e}")
        return [f"Error: {str(e)}"]
