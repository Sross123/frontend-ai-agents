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

# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------
@api.post("/generate")
async def generate_team_flow(request: UserRequest):
    """Execute the LangGraph flow and stream node results via SSE."""

    async def event_stream():
        initial_state = {"messages": [HumanMessage(content=request.task)]}
        try:
            async for output in app_graph.astream(initial_state, stream_mode="updates"):
                for node_name, data in output.items():
                    if "messages" in data and data["messages"]:
                        latest_msg = data["messages"][-1].content
                        payload = {"agent": node_name, "content": latest_msg}
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
