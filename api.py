"""
api.py
------
FastAPI REST layer for the Multi-Agent Loan Underwriting system.

WHY THIS EXISTS
  The Streamlit app (app.py) is an internal ops tool — useful for manual
  demos and credit officer HITL reviews. A production NBFC integrates the
  underwriting engine from their Loan Origination System (LOS), a backend
  service that cannot call a Streamlit UI.

  This module exposes the same LangGraph graph via a REST API so any
  service (Node.js LOS, Python microservice, test runner) can call it
  programmatically without touching the UI layer.

ENDPOINTS
  POST /underwrite          — run a full underwriting analysis
  POST /underwrite/resume   — resume a paused HITL case with officer decision
  GET  /underwrite/{thread_id} — fetch the current state of any thread
  GET  /health              — liveness check

DESIGN DECISIONS
  - One shared graph instance (module-level singleton), same as the
    Streamlit app. The checkpointer handles thread isolation.
  - thread_id is client-supplied on /underwrite so the caller owns the
    correlation ID and can query state later. A UUID is generated if omitted.
  - HITL resume uses the same Command(resume=...) pattern as app.py —
    no changes to the graph or agent code.
  - Pydantic v2 models are used for request/response validation.
    FastAPI uses Pydantic natively for OpenAPI schema generation.
  - The API does not stream events — it returns the final state as a
    single JSON response. Streaming via SSE is a natural extension.

RUN LOCALLY
  pip install fastapi uvicorn[standard]
  uvicorn api:app --reload --port 8080

  # Submit an application
  curl -X POST http://localhost:8080/underwrite \\
    -H "Content-Type: application/json" \\
    -d '{
      "applicant_name": "Arjun Mehta",
      "applicant_age": 32,
      "annual_income": 1200000,
      "loan_amount_requested": 3000000,
      "loan_purpose": "Home Purchase",
      "employment_type": "Salaried",
      "existing_obligations": 10000,
      "document_text": "Salary certificate: Arjun Mehta, Annual CTC 12,00,000"
    }'

AZURE DEPLOYMENT
  Add to startup.sh alongside Streamlit, or deploy as a separate App Service.
  The graph and checkpointer are shared-nothing per process — each App Service
  instance has its own PostgreSQL ConnectionPool; checkpoints are shared via DB.

References:
  FastAPI: https://fastapi.tiangolo.com/
  Pydantic v2: https://docs.pydantic.dev/latest/
  LangGraph Command: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import os

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from graph import build_graph, get_initial_state
from langgraph.types import Command

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")

# ── API key authentication ─────────────────────────────────────────────────────
# Protects /underwrite and /underwrite/resume from unauthenticated callers.
# In production (Azure), set API_KEY as an Application Setting.
# For local dev the fallback is "demo-key-change-in-production".
#
# Usage: pass the key as a request header:
#   curl -H "X-API-Key: your-key" -X POST .../underwrite ...
#
# /health and GET /underwrite/{thread_id} are intentionally left unprotected:
#   /health  — Azure App Service liveness probe must reach it without auth.
#   GET state — read-only polling; no sensitive write operations.
#
# auto_error=False means a missing/wrong key returns our custom 403,
# not FastAPI's default 403 with a different message shape.
# Reference: https://fastapi.tiangolo.com/tutorial/security/api-key/
API_KEY = os.environ.get("API_KEY", "demo-key-change-in-production")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _verify_api_key(api_key: str = Depends(_api_key_header)) -> None:
    """FastAPI dependency — raises 403 if the X-API-Key header is missing or wrong."""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Pass X-API-Key header.",
        )


# ── Shared graph instance ──────────────────────────────────────────────────────
# Built once at import time, same lifecycle as the Streamlit app.
# The checkpointer (MemorySaver in dev, PostgresSaver in production) handles
# thread isolation — each thread_id is an independent conversation.
_graph = build_graph()


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Loan Underwriting AI",
    description=(
        "Multi-agent loan underwriting pipeline built with LangGraph, "
        "Groq LLaMA 3.3-70B, and sklearn GradientBoostingRegressor. "
        "Supports Human-in-the-Loop (HITL) for borderline Refer cases."
    ),
    version="1.0.0",
)


# ── Request / Response models ──────────────────────────────────────────────────

class UnderwriteRequest(BaseModel):
    """
    Loan application submitted to the underwriting pipeline.
    All monetary values in INR.
    """
    applicant_name: str = Field(..., description="Full name of the applicant")
    applicant_age: int = Field(..., ge=21, le=65, description="Age in years (21–65)")
    annual_income: float = Field(..., gt=0, description="Annual income in INR")
    loan_amount_requested: float = Field(..., gt=0, description="Requested loan amount in INR")
    loan_purpose: str = Field(..., description="Purpose: Home Purchase / Business Expansion / Personal Loan / Vehicle / Education")
    employment_type: str = Field(..., description="Salaried / Self-Employed / Business")
    existing_obligations: float = Field(0.0, ge=0, description="Existing monthly EMI obligations in INR")
    document_text: str = Field("", description="Paste document text (salary cert / ITR / bank statement)")
    thread_id: Optional[str] = Field(
        None,
        description=(
            "Client-supplied correlation ID. A UUID is generated if omitted. "
            "Use this ID for /underwrite/resume and /underwrite/{thread_id}."
        ),
    )
    cibil_score_override: Optional[int] = Field(
        None,
        description="Inject a deterministic CIBIL score (demo/test only). None = use mock hash.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "applicant_name": "Arjun Mehta",
            "applicant_age": 32,
            "annual_income": 1200000,
            "loan_amount_requested": 3000000,
            "loan_purpose": "Home Purchase",
            "employment_type": "Salaried",
            "existing_obligations": 10000,
            "document_text": "Salary certificate — Arjun Mehta. Annual CTC: ₹12,00,000.",
        }
    }}


class HITLResumeRequest(BaseModel):
    """
    Credit officer decision for a paused Refer case.
    Submit to POST /underwrite/resume to continue the graph.
    """
    thread_id: str = Field(..., description="thread_id returned by the original /underwrite call")
    hitl_decision: str = Field(..., description="Officer decision: 'Approve' or 'Reject'")
    hitl_notes: str = Field("No notes provided.", description="Officer notes / override conditions")

    model_config = {"json_schema_extra": {
        "example": {
            "thread_id": "abc-123",
            "hitl_decision": "Approve",
            "hitl_notes": "Approved with condition: co-applicant required.",
        }
    }}


class UnderwriteResponse(BaseModel):
    """
    Response from the underwriting pipeline.

    When status == "refer_pending", the graph has paused for HITL review.
    Submit the officer decision to POST /underwrite/resume with the same thread_id.
    """
    thread_id: str
    status: str = Field(..., description="'complete' | 'refer_pending' | 'error'")
    decision: Optional[str] = None
    final_decision: Optional[str] = None
    cibil_score: Optional[int] = None
    risk_score: Optional[float] = None
    risk_label: Optional[str] = None
    foir: Optional[float] = None
    loan_to_income_ratio: Optional[float] = None
    rule_flags: list[str] = []
    rule_passed: Optional[bool] = None
    confidence: Optional[float] = None
    decision_reasons: list[str] = []
    hitl_payload: Optional[dict] = None
    explanation_en: Optional[str] = None
    explanation_hi: Optional[str] = None
    hitl_decision: Optional[str] = None
    hitl_notes: Optional[str] = None


# ── Helper ─────────────────────────────────────────────────────────────────────

def _make_config(thread_id: str) -> dict:
    """Build graph invocation config with thread_id."""
    config: dict = {"configurable": {"thread_id": thread_id}}

    # Optionally attach Langfuse tracing if available
    try:
        from utils.tracing import get_langfuse_handler
        handler, lf_metadata = get_langfuse_handler(session_id=thread_id)
        config["callbacks"] = [handler]
        config["metadata"] = lf_metadata
    except Exception:
        pass  # Tracing is optional; don't break the API if keys are missing

    return config


def _state_to_response(thread_id: str, state_values: dict, status: str) -> UnderwriteResponse:
    """Convert a LangGraph state dict to an UnderwriteResponse."""
    return UnderwriteResponse(
        thread_id=thread_id,
        status=status,
        decision=state_values.get("decision"),
        final_decision=state_values.get("final_decision"),
        cibil_score=state_values.get("cibil_score"),
        risk_score=state_values.get("risk_score"),
        risk_label=state_values.get("risk_label"),
        foir=state_values.get("foir"),
        loan_to_income_ratio=state_values.get("loan_to_income_ratio"),
        rule_flags=state_values.get("rule_flags", []),
        rule_passed=state_values.get("rule_passed"),
        confidence=state_values.get("confidence"),
        decision_reasons=state_values.get("decision_reasons", []),
        explanation_en=state_values.get("explanation_en"),
        explanation_hi=state_values.get("explanation_hi"),
        hitl_decision=state_values.get("hitl_decision"),
        hitl_notes=state_values.get("hitl_notes"),
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    """Liveness check. Returns 200 if the API is up and the graph is compiled."""
    return {"status": "ok", "graph_nodes": list(_graph.nodes)}


@app.post("/underwrite", response_model=UnderwriteResponse, dependencies=[Depends(_verify_api_key)])
def underwrite(req: UnderwriteRequest) -> UnderwriteResponse:
    """
    Submit a loan application and run the full underwriting pipeline.

    The graph runs synchronously and returns when it either:
      a) Completes all six agents → status='complete', decision in response.
      b) Pauses at a HITL interrupt (Refer case) → status='refer_pending'.
         POST /underwrite/resume with the same thread_id to continue.

    The thread_id in the response is the correlation ID for all subsequent
    calls to /underwrite/resume and /underwrite/{thread_id}.
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    config = _make_config(thread_id)

    initial_state = get_initial_state(
        applicant_name=req.applicant_name,
        applicant_age=req.applicant_age,
        annual_income=req.annual_income,
        loan_amount_requested=req.loan_amount_requested,
        loan_purpose=req.loan_purpose,
        employment_type=req.employment_type,
        existing_obligations=req.existing_obligations,
        document_text=req.document_text,
        document_bytes=None,  # PDF upload not supported via REST in this version
        cibil_score_override=req.cibil_score_override,
    )

    status = "complete"
    hitl_payload: Optional[dict] = None

    try:
        for event in _graph.stream(initial_state, config=config):
            for node_name, _ in event.items():
                if node_name == "__interrupt__":
                    status = "refer_pending"
                    # Extract the HITL review payload from the interrupt value
                    interrupt_value = event.get("__interrupt__", [])
                    if isinstance(interrupt_value, (list, tuple)) and interrupt_value:
                        item = interrupt_value[0]
                        hitl_payload = item.value if hasattr(item, "value") else item
                    break
            if status == "refer_pending":
                break
    except Exception as exc:
        logger.exception("[API /underwrite] Graph execution error for thread %s", thread_id)
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {exc}") from exc

    # Fetch the full merged state from the checkpointer
    full_state = _graph.get_state(config)
    response = _state_to_response(thread_id, dict(full_state.values), status)

    if hitl_payload:
        response.hitl_payload = hitl_payload

    logger.info(
        "[API /underwrite] thread=%s status=%s decision=%s",
        thread_id, status, response.decision,
    )
    return response


@app.post("/underwrite/resume", response_model=UnderwriteResponse, dependencies=[Depends(_verify_api_key)])
def underwrite_resume(req: HITLResumeRequest) -> UnderwriteResponse:
    """
    Resume a paused Refer case with the credit officer's decision.

    The thread_id must match a thread previously paused at status='refer_pending'.
    On success returns status='complete' with the final decision and letters.

    Uses the same Command(resume=...) pattern as the Streamlit HITL panel.
    Reference: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
    """
    if req.hitl_decision not in ("Approve", "Reject"):
        raise HTTPException(
            status_code=422,
            detail=f"hitl_decision must be 'Approve' or 'Reject', got '{req.hitl_decision}'",
        )

    config = _make_config(req.thread_id)
    resume_command = Command(resume={
        "hitl_decision": req.hitl_decision,
        "hitl_notes": req.hitl_notes,
    })

    try:
        for event in _graph.stream(resume_command, config=config):
            pass  # drain the stream; all state is captured via get_state below
    except Exception as exc:
        logger.exception("[API /resume] Resume error for thread %s", req.thread_id)
        raise HTTPException(status_code=500, detail=f"Resume failed: {exc}") from exc

    full_state = _graph.get_state(config)
    response = _state_to_response(req.thread_id, dict(full_state.values), "complete")

    logger.info(
        "[API /resume] thread=%s hitl_decision=%s final_decision=%s",
        req.thread_id, req.hitl_decision, response.final_decision,
    )
    return response


@app.get("/underwrite/{thread_id}", response_model=UnderwriteResponse)
def get_underwrite_state(thread_id: str) -> UnderwriteResponse:
    """
    Fetch the current state of any underwriting thread by ID.

    Useful for polling the status of an async job, or for retrieving
    the result after a HITL resume completes.

    Returns 404 if the thread_id is unknown to the checkpointer.
    """
    config = _make_config(thread_id)

    try:
        full_state = _graph.get_state(config)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Thread '{thread_id}' not found: {exc}",
        ) from exc

    if not full_state or not full_state.values:
        raise HTTPException(
            status_code=404,
            detail=f"Thread '{thread_id}' not found or has no state.",
        )

    state_values = dict(full_state.values)

    # Determine status from state
    if state_values.get("final_decision"):
        status = "complete"
    elif state_values.get("decision") == "Refer" and not state_values.get("hitl_decision"):
        status = "refer_pending"
    else:
        status = "complete"

    return _state_to_response(thread_id, state_values, status)
