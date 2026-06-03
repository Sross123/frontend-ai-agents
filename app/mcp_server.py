from mcp.server.fastmcp import FastMCP
import os

# Initialize the MCP server
mcp = FastMCP("FrontendAgentServer")

# Define the file writing tool
@mcp.tool()
async def write_to_file(path: str, content: str) -> str:
    """Writes code to a file in the workspace."""
    base_dir = os.path.abspath("./workspace")
    full_path = os.path.normpath(os.path.join(base_dir, path))
    
    # Security: Ensure we stay in the workspace folder
    if not full_path.startswith(base_dir):
        return "Error: Access denied."
        
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    return f"File {path} created/updated successfully."

# Define a tool to list files (for the agent to know what it has)
@mcp.tool()
async def list_files(directory: str = ".") -> list[str]:
    """Lists files in the workspace."""
    return os.listdir(os.path.join("./workspace", directory))

@mcp.tool()
async def read_file(path: str) -> str:
    """Reads the content of a file in the workspace."""
    base_dir = os.path.abspath("./workspace")
    full_path = os.path.normpath(os.path.join(base_dir, path))
    
    # Security: Ensure we stay in the workspace folder
    if not full_path.startswith(base_dir):
        return "Error: Access denied."
        
    if not os.path.exists(full_path):
        return f"Error: File {path} does not exist."
        
    with open(full_path, "r") as f:
        return f.read()

if __name__ == "__main__":
    mcp.run()