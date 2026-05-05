"""
ContextRevive — FastAPI Server
5 endpoints: create session, chat, simulate gap, session status, health check.
"""

import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.agent import ContextReviveAgent
from core.ollama_client import test_ollama_connection


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ContextRevive API",
    description="AI context recovery engine for broken conversations.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global agent instance
agent = ContextReviveAgent()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    scenario: str | None = None  # "customer_support", "student_tutor", or null


class ChatRequest(BaseModel):
    message: str


class SimulateGapRequest(BaseModel):
    turn_indices: list[int]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/sessions")
async def create_session(request: CreateSessionRequest):
    """Create a new conversation session, optionally from a scenario script."""
    try:
        session_id = agent.new_session(request.scenario)
        return {
            "session_id": session_id,
            "scenario": request.scenario,
            "status": "ready",
        }
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario: '{request.scenario}'",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/chat")
async def chat(session_id: str, request: ChatRequest):
    """Send a message and receive an AI response with reconstruction metadata."""
    if session_id not in agent.tracker.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        result = agent.chat(session_id, request.message)
        return {
            "response_text": result.response_text,
            "used_reconstruction": result.used_reconstruction,
            "confidence": result.confidence,
            "strategy": result.strategy,
            "integrity_score": result.integrity_score,
            "inferred_summary": result.inferred_summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/simulate-gap")
async def simulate_gap(session_id: str, request: SimulateGapRequest):
    """Mark specific turns as missing to simulate a context gap."""
    if session_id not in agent.tracker.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        gap_info = agent.simulate_gap(session_id, request.turn_indices)
        return gap_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/status")
async def session_status(session_id: str):
    """Full session snapshot — polled by the UI transparency panel."""
    if session_id not in agent.tracker.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        status = agent.get_session_status(session_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Verify Ollama connectivity and model availability."""
    try:
        ollama_ok = test_ollama_connection()
        return {
            "ollama_running": ollama_ok,
            "model": "llama3.1:8b",
            "embed_model": "nomic-embed-text",
            "status": "ok" if ollama_ok else "error",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))