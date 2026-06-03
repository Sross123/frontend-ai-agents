# routes.py
"""FastAPI routes for the AI Multi‑Agent Studio backend.

The heavy lifting (LangGraph workflow) lives in ``app.graph``. This module
exposes a ``FastAPI`` instance named ``api`` that the entry point imports.
"""

import json
import pathlib
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from .graph import graph as app_graph
from .config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
settings = get_settings()

api = FastAPI(
    title="AI Multi-Agent Studio Backend",
    description="Secure real‑time streaming orchestrator serving beautiful UI generation workflow.",
)

# CORS – use environment settings for flexibility
cors_origins = settings.get_cors_origins()
logger.info(f"Configuring CORS with origins: {cors_origins}")

api.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class UserRequest(BaseModel):
    task: str
    workspace: str = "./workspace"


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------
@api.post("/generate")
async def generate_team_flow(request: UserRequest):
    """Execute the LangGraph flow and stream node results via SSE.
    
    Args:
        request: User request containing the task and optional workspace path.
        
    Returns:
        StreamingResponse with Server-Sent Events containing agent outputs.
    """

    async def event_stream():
        initial_state = {
            "messages": [HumanMessage(content=request.task)],
            "workspace": request.workspace
        }
        try:
            logger.info(f"Starting agent flow for task: {request.task[:50]}...")
            started_nodes = set()
            async for event in app_graph.astream_events(initial_state, version="v2"):
                event_type = event["event"]
                node_name = event["metadata"].get("langgraph_node")
                
                if node_name:
                    clean_name = node_name.replace("_agent", "")
                    if clean_name in ["pm", "coder"]:
                        if event_type == "on_chain_start" and node_name not in started_nodes:
                            started_nodes.add(node_name)
                            payload = {"agent": clean_name, "content": "", "start": True}
                            yield f"data: {json.dumps(payload)}\n\n"
                            logger.debug(f"Agent {clean_name} started")
                            
                        elif event_type == "on_chat_model_stream":
                            chunk_content = event["data"]["chunk"].content
                            if chunk_content:
                                payload = {"agent": clean_name, "content": chunk_content, "chunk": True}
                                yield f"data: {json.dumps(payload)}\n\n"
                                
                        elif event_type == "on_chain_end" and node_name in started_nodes:
                            payload = {"agent": clean_name, "content": "", "end": True}
                            yield f"data: {json.dumps(payload)}\n\n"
                            logger.debug(f"Agent {clean_name} completed")
            
            logger.info("Agent flow completed successfully")
        except Exception as e:
            logger.error(f"Graph execution failed: {str(e)}", exc_info=True)
            error_payload = {"agent": "error", "content": f"Graph Execution Failed: {str(e)}"}
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------
@api.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Static file serving (index.html)
# ---------------------------------------------------------------------------
static_dir = pathlib.Path(__file__).parent.parent  # project root

@api.get("/")
async def serve_index():
    """Serve the frontend ``index.html`` securely."""
    logger.debug("Serving index.html")
    return FileResponse(static_dir / "index.html")
