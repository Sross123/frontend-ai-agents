# routes.py
"""FastAPI routes for the AI Multi‑Agent Studio backend.

The heavy lifting (LangGraph workflow) lives in ``app.graph``. This module
exposes a ``FastAPI`` instance named ``api`` that the entry point imports.
"""

import json
import pathlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# Import the compiled LangGraph application.
from .graph import graph as app_graph

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
api = FastAPI(
    title="AI Multi-Agent Studio Backend",
    description="Secure real‑time streaming orchestrator serving beautiful UI generation workflow.",
)

# CORS – allow all origins for development (tighten in production).
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    """Execute the LangGraph flow and stream node results via SSE."""

    async def event_stream():
        initial_state = {
            "messages": [HumanMessage(content=request.task)],
            "workspace": request.workspace
        }
        try:
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
                            
                        elif event_type == "on_chat_model_stream":
                            chunk_content = event["data"]["chunk"].content
                            if chunk_content:
                                payload = {"agent": clean_name, "content": chunk_content, "chunk": True}
                                yield f"data: {json.dumps(payload)}\n\n"
                                
                        elif event_type == "on_chain_end" and node_name in started_nodes:
                            payload = {"agent": clean_name, "content": "", "end": True}
                            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            error_payload = {"agent": "error", "content": f"Graph Execution Failed: {str(e)}"}
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# Static file serving (index.html)
# ---------------------------------------------------------------------------
static_dir = pathlib.Path(__file__).parent.parent  # project root

@api.get("/")
async def serve_index():
    """Serve the frontend ``index.html`` securely."""
    return FileResponse(static_dir / "index.html")
