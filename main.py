"""
================================================================================
🤖 AI MULTI-AGENT STUDIO - ORCHESTRATION ENGINE
================================================================================
This script coordinates a team of specialized AI agents (a Product Manager 
and a Frontend Engineer) using LangGraph to build pristine modern frontends.

It is structured to be read like a story, from setting the stage (imports and
configuration) to defining the characters (prompts), memory (state), actions 
(nodes), and final publication (FastAPI endpoints).
"""

# ================================================================================
# PROLOGUE: THE ASSEMBLY OF TOOLS
# ================================================================================
import json
import operator
import os
import pathlib
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from app.tools import write_to_file, read_file, list_files

# Initialize environment variables from the .env file
load_dotenv()


# ================================================================================
# CHAPTER 1: THE CONFIGURATION & MODEL FACTORY
# ================================================================================
def initialize_agent_models() -> tuple[ChatOpenAI, ChatOpenAI]:
    """
    Decides the AI model provider (Local Ollama or Cloud OpenRouter)
    based on environment configurations, returning (pm_llm, coder_llm).
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if provider == "gemini" and gemini_key:
        # Gemini Cloud Execution: Uses Google's native OpenAI-compatible API
        pm_model = ChatOpenAI(
            model="gemini-2.5-flash",
            openai_api_key=gemini_key,
            openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=0.2,
        )
        coder_model = ChatOpenAI(
            model="gemini-2.5-flash",
            openai_api_key=gemini_key,
            openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=0.1,
        )
    elif provider == "openrouter" and openrouter_key:
        # Cloud Execution: Uses OpenRouter to load Claude 3 Haiku and Qwen-Coder-Next
        pm_model = ChatOpenAI(
            model="anthropic/claude-3-haiku",
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.2,
        )
        coder_model = ChatOpenAI(
            model="qwen/qwen3-coder-next",
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1,
        )
    else:
        # Local Execution: Connects to your local Ollama server running Llama 3.2
        # Uses the OpenAI-compatible local API endpoint (requires zero extra packages)
        pm_model = ChatOpenAI(
            model="llama3.2",
            openai_api_key="ollama",
            openai_api_base="http://localhost:11434/v1",
            temperature=0.2, 
        )
        coder_model = ChatOpenAI(
            model="llama3.2",
            openai_api_key="ollama",
            openai_api_base="http://localhost:11434/v1",
            temperature=0.1,
        )

    return pm_model, coder_model


# Spawn our two specialized AI models
pm_llm, coder_llm = initialize_agent_models()


# ================================================================================
# CHAPTER 2: THE AGENT PERSONAS (PROMPTS DEFINITION)
# ================================================================================
class Personas:
    """
    A clean repository of system instructions defining the behavior, 
    expertise, and specific delivery goals of our AI agents.
    """

    PRODUCT_MANAGER_SYSTEM = (
        "You are an expert Frontend Product Manager and UI/UX Designer. "
        "Before creating anything, understand the business purpose, the problem being solved, "
        "and the user's core goals. If the user provides full project business logic, preserve it and "
        "only suggest code or structure changes around that existing logic. "
        "Treat every request as a frontend development or frontend integration task, including UI design, component wiring, "
        "API integration, build setup, or framework adaptation. "
        "Then expand the interface request into a clear, step-by-step technical specification outlining:\n"
        "1. Component structure and layout hierarchy.\n"
        "2. Styling requirements, responsiveness, and color themes.\n"
        "3. Interactive behavior, events, and state management logic needed."
    )

    CODER_SYSTEM = (
        "You are an elite Frontend Software Engineer.\n\n"
        "CRITICAL WORKFLOW (You MUST follow this sequence before writing any source code files):\n"
        "1. Inspect existing files using `list_files` and `read_file` tools if updating or integrating.\n"
        "2. **Create Plan**: Call `write_to_file` to create `plan.md` inside the workspace outlining the implementation design, layout decisions, and dependencies.\n"
        "3. **Create Tasks Checklist**: Call `write_to_file` to create `tasks.md` inside the workspace containing a structured checklist of all source files to be created/updated.\n"
        "4. **Write Code**: Implement files and use `write_to_file` to write source code files.\n"
        "5. **Update Tasks Checklist**: Modify `tasks.md` to check off (`[x]`) all completed files and tasks.\n"
        "6. Do NOT just output code as text in chat—always save the plan, tasks, and code files to the workspace using the tools."
    )

    @staticmethod
    def get_coder_user_prompt(specification: str) -> str:
        return (
            f"Build pristine, modern, functional code based strictly on the following UI specification:\n\n"
            f"{specification}\n\n"
            "INSTRUCTIONS:\n"
            "1. Read any existing files using `read_file` to understand the current state if relevant.\n"
            "2. **CRITICAL FIRST STEP**: Call `write_to_file` to create `plan.md` (detailing implementation design) and `tasks.md` (checklist of files/tasks) inside the workspace folder BEFORE writing any code files.\n"
            "3. Create or modify the necessary code files and save them using `write_to_file`.\n"
            "4. Update `tasks.md` using `write_to_file` to check off (`[x]`) completed items.\n"
            "5. Keep code clean, well-commented, and properly formatted.\n\n"
            "CRITICAL: Always use `write_to_file` to save your work (planning, checklist, and code files) to the workspace!"
        )


# ================================================================================
# CHAPTER 3: THE SHARED MEMORY (STATE SCHEMA)
# ================================================================================
class TeamState(TypedDict):
    """
    Represents the shared memory and context passed between agents
    as they collaborate along the workflow.
    """
    # The message history, automatically appending new messages via operator.add
    messages: Annotated[list[BaseMessage], operator.add]
    
    # The technical specification document written by the PM
    specification: str
    
    # The identifier of the next agent scheduled to run
    next_agent: str

    # Target directory for read/write operations
    workspace: str


# ================================================================================
# CHAPTER 4: THE AGENT WORKERS (NODES)
# ================================================================================
async def pm_agent(state: TeamState) -> dict:
    """
    The Product Manager Agent Node.
    Consumes the user request and drafts a comprehensive UI component specification.
    """
    user_messages = state["messages"]
    
    # Generate the specification using standard system message constraints
    response = await pm_llm.ainvoke([
        SystemMessage(content=Personas.PRODUCT_MANAGER_SYSTEM)
    ] + user_messages)
    
    return {
        "messages": [AIMessage(content=response.content, name="ProductManager")],
        "specification": response.content,
        "next_agent": "coder",
    }


async def coder_agent(state: TeamState) -> dict:
    """
    The Frontend Software Engineer Agent Node.
    Consumes the PM's specification and writes copy-pasteable UI code blocks.
    Also executes tool calls (writing files to workspace).
    """
    workspace = state.get("workspace", "./workspace")
    spec = state["specification"]
    
    # Retrieve system role and prompt instructions
    system_instruction = SystemMessage(content=Personas.CODER_SYSTEM)
    user_prompt = HumanMessage(content=Personas.get_coder_user_prompt(spec))
    
    # Bind the filesystem tools to the coder LLM
    coder_with_tools = coder_llm.bind_tools([write_to_file, list_files, read_file])
    
    messages = [system_instruction, user_prompt]
    response = await coder_with_tools.ainvoke(messages)
    
    # Agentic loop: execute tool calls in response
    tool_iteration = 0
    max_iterations = 10  # Prevent infinite loops
    
    while response.tool_calls and tool_iteration < max_iterations:
        tool_iteration += 1
        messages.append(response)
        tool_messages = []
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Inject workspace directory parameter if not present
            if "workspace_dir" not in tool_args:
                tool_args["workspace_dir"] = workspace
                
            try:
                if tool_name == "write_to_file":
                    tool_result = await write_to_file.ainvoke(tool_args)
                elif tool_name == "list_files":
                    tool_result = await list_files.ainvoke(tool_args)
                elif tool_name == "read_file":
                    tool_result = await read_file.ainvoke(tool_args)
                else:
                    tool_result = f"Error: Tool '{tool_name}' not found."
                    
            except Exception as e:
                tool_result = f"Error executing tool: {str(e)}"
                
            tool_messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
                name=tool_name
            ))
            
        messages.extend(tool_messages)
        response = await coder_with_tools.ainvoke(messages)
        
    return {
        "messages": [AIMessage(content=response.content, name="Coder")],
        "next_agent": "end",
    }


# ================================================================================
# CHAPTER 5: THE COLLABORATION WORKFLOW (LANGGRAPH)
# ================================================================================
def routing_router(state: TeamState) -> str:
    """
    Reads the state's `next_agent` flag to decide where to transition next.
    """
    if state.get("next_agent") == "coder":
        return "coder"
    return END


# Create the state orchestration builder
builder = StateGraph(TeamState)

# Register the specialized worker nodes
builder.add_node("pm", pm_agent)
builder.add_node("coder", coder_agent)

# Set the flow logic and transitions
builder.add_edge(START, "pm")
builder.add_conditional_edges(
    "pm",
    routing_router,
    {
        "coder": "coder",
        END: END,
    },
)
builder.add_edge("coder", END)

# Compile into an executable agent application
app = builder.compile()


# ================================================================================
# CHAPTER 6: THE STREAMING GATEWAY (FASTAPI)
# ================================================================================
api = FastAPI(
    title="AI Multi-Agent Studio Backend",
    description="Secure real-time streaming orchestrator serving beautiful UI generation workflow.",
)

# Standard Cross-Origin Resource Sharing (CORS) rules
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserRequest(BaseModel):
    task: str
    workspace: str = "./workspace"


@api.post("/generate")
async def generate_team_flow(request: UserRequest):
    """
    Executes the LangGraph multi-agent flow asynchronously,
    streaming node updates to the client using Server-Sent Events (SSE) v2.
    """

    async def event_stream():
        initial_state = {
            "messages": [HumanMessage(content=request.task)],
            "workspace": request.workspace
        }
        
        try:
            started_nodes = set()
            async for event in app.astream_events(initial_state, version="v2"):
                event_type = event["event"]
                node_name = event["metadata"].get("langgraph_node")
                
                if node_name:
                    clean_name = node_name.replace("_agent", "")
                    if clean_name in ["pm", "coder"]:
                        if event_type == "on_chain_start" and node_name not in started_nodes:
                            started_nodes.add(node_name)
                            payload = {"agent": clean_name, "content": "", "start": True}
                            yield f"data: {json.dumps(payload)}\n\n"
                            
                        elif event_type == "on_chat_model_stream":
                            chunk_content = event["data"]["chunk"].content
                            if chunk_content:
                                payload = {"agent": clean_name, "content": chunk_content, "chunk": True}
                                yield f"data: {json.dumps(payload)}\n\n"
                                
                        elif event_type == "on_chain_end" and node_name in started_nodes:
                            payload = {"agent": clean_name, "content": "", "end": True}
                            yield f"data: {json.dumps(payload)}\n\n"
                            
        except Exception as e:
            # Gracefully streams any execution error back to the frontend
            error_payload = {
                "agent": "error",
                "content": f"Graph Execution Failed: {str(e)}"
            }
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ================================================================================
# EPILOGUE: THE SECURE STATIC GATE
# ================================================================================
static_dir = pathlib.Path(__file__).parent


@api.get("/api/browse")
async def browse_directory(path: str = None):
    """Browse directories on the server's local machine."""
    import os
    import platform
    try:
        drives = []
        is_windows = platform.system() == "Windows"
        if is_windows:
            import ctypes
            bitmask = ctypes.windll.kernel32.GetLogicalDrives()
            for letter in map(chr, range(65, 91)):
                if bitmask & 1:
                    drives.append(f"{letter}:\\")
                bitmask >>= 1

        if not path:
            current_dir = os.getcwd()
        else:
            current_dir = os.path.abspath(path)
            
        if not os.path.exists(current_dir):
            return {"error": f"Path '{current_dir}' does not exist"}
            
        if not os.path.isdir(current_dir):
            return {"error": f"Path '{current_dir}' is not a directory"}
            
        items = []
        try:
            for entry in os.scandir(current_dir):
                try:
                    if entry.is_dir() and not entry.name.startswith('.'):
                        items.append(entry.name)
                except PermissionError:
                    continue
        except PermissionError:
            return {
                "current_path": current_dir,
                "parent_path": os.path.dirname(current_dir) if current_dir != os.path.dirname(current_dir) else None,
                "directories": [],
                "drives": drives,
                "error": "Permission Denied"
            }
            
        items.sort(key=str.lower)
        
        return {
            "current_path": current_dir,
            "parent_path": os.path.dirname(current_dir) if current_dir != os.path.dirname(current_dir) else None,
            "directories": items,
            "drives": drives
        }
    except Exception as e:
        return {"error": str(e)}


@api.get("/")
async def serve_index():
    """
    Serves the frontend safely. Explicitly sends index.html,
    preventing direct folder traversal or exposure of source code and credentials.
    """
    return FileResponse(static_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    # Launches uvicorn local development server on port 8000
    uvicorn.run(api, host="0.0.0.0", port=8000)