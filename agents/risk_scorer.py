"""
agents/risk_scorer.py
---------------------
Agent 4: Risk Scorer

Calls the ML risk model as a tool. This is the "tool use" pattern —
the agent delegates to a non-LLM component for the actual prediction.

This is where your existing credit fraud detection model plugs in.
See utils/risk_model.py → score_applicant() for the interface.

Interview note: "This demonstrates tool use — the agent orchestrates a
call to a separate ML model, which is how production systems work.
The LLM handles language tasks; specialised models handle prediction tasks."
"""
import logging

from state import UnderwritingState
from utils.risk_model import score_applicant

logger = logging.getLogger(__name__)


def risk_scorer_node(state: UnderwritingState) -> dict:
    """
    Score applicant using the ML risk model.
    Reads from state fields produced by the previous three agents.
    """
    logger.info("[Agent 4: Risk Scorer] Scoring risk for %s", state["applicant_name"])

    risk_score, risk_label, features_used = score_applicant(
        cibil_score=state["cibil_score"],
        foir=state["foir"],
        loan_to_income_ratio=state["loan_to_income_ratio"],
        doc_confidence=state.get("doc_confidence", 1.0),
        active_loans_count=state["active_loans_count"],
        payment_history_months=state["payment_history_months"],
    )

    logger.info(
        "[Risk Scorer] score=%.3f label=%s top_features=%s",
        risk_score, risk_label, features_used
    )

    return {
        "risk_score": risk_score,
        "risk_label": risk_label,
        "risk_features_used": features_used,
    }
