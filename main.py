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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

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
        "Analyze the user's interface request and expand it into a clear, "
        "step-by-step technical specification outlining:\n"
        "1. Component structure and layout hierarchy.\n"
        "2. Styling requirements, responsiveness, and color themes.\n"
        "3. Interactive behavior, events, and state management logic needed."
    )

    CODER_SYSTEM = (
        "You are an elite Frontend Software Engineer."
    )

    @staticmethod
    def get_coder_user_prompt(specification: str) -> str:
        return (
            f"Build pristine, modern, functional code based strictly on the following UI specification:\n\n"
            f"{specification}\n\n"
            f"Provide complete, copy-pasteable code blocks (HTML, CSS, or JavaScript/React as requested). "
            f"Enclose code snippets cleanly in markdown codeblocks."
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
    """
    spec = state["specification"]
    
    # Retrieve system role and prompt instructions
    system_instruction = SystemMessage(content=Personas.CODER_SYSTEM)
    user_prompt = HumanMessage(content=Personas.get_coder_user_prompt(spec))
    
    # Generate complete frontend code
    response = await coder_llm.ainvoke([system_instruction, user_prompt])
    
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


@api.post("/generate")
async def generate_team_flow(request: UserRequest):
    """
    Executes the LangGraph multi-agent flow asynchronously,
    streaming node updates to the client using Server-Sent Events (SSE).
    """

    async def event_stream():
        initial_state = {"messages": [HumanMessage(content=request.task)]}
        
        try:
            # Streams structural updates node-by-node
            async for output in app.astream(initial_state, stream_mode="updates"):
                for node_name, data in output.items():
                    if "messages" in data and data["messages"]:
                        # Extract the final completed content from the node
                        latest_msg = data["messages"][-1].content
                        payload = {"agent": node_name, "content": latest_msg}
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