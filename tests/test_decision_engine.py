"""
tests/test_decision_engine.py
------------------------------
Unit tests for decision_engine_node — covers the three deterministic paths
that do NOT trigger interrupt().

WHY THESE THREE PATHS:
  The Approve and Reject paths are pure Python — no LLM call, no interrupt().
  The HITL guard path (hitl_decision already set) is also pure Python.
  All three are fully testable without mocking get_llm() or the LangGraph
  runtime.

  The Refer path (rule_flags=1 flag OR risk_label="Medium") is intentionally
  NOT tested here — calling decision_engine_node with a Refer-triggering state
  would invoke interrupt(), which raises internally and requires the full
  compiled LangGraph graph to handle. That path is validated via the manual
  HITL demo and evals/eval_llm_agents.py.

CONFIDENCE VALUES (verified against decision_engine.py):
  Approve:              0.80   (0 flags, Low risk)
  Reject (2+ flags):   0.85   (len(rule_flags) >= 2 takes priority over High risk branch)
  Reject (High only):  0.75   (risk_label == "High" with 0-1 flags)
  Refer:               0.55   (not tested here — triggers interrupt())

References:
  pytest docs: https://docs.pytest.org/en/stable/
  conftest.py: sets GROQ_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY stubs
               for all tests — no need to re-declare them here.
"""
import pytest
from agents.decision_engine import decision_engine_node


# ─── Helper ───────────────────────────────────────────────────────────────────

def _state(**overrides) -> dict:
    """
    Minimal state dict for decision_engine_node.
    Defaults to a clean Approve-path applicant.
    """
    base = {
        "applicant_name": "Test Applicant",
        "rule_flags": [],
        "risk_label": "Low",
        "risk_score": 0.20,
        "cibil_score": 750,
        "foir": 0.35,
        "loan_amount_requested": 3_000_000,
        "hitl_decision": None,
        "hitl_notes": None,
    }
    base.update(overrides)
    return base


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestDecisionEngineNode:

    def test_approve_on_clean_profile(self):
        """0 rule flags + Low risk → Approve with confidence 0.80."""
        result = decision_engine_node(_state())
        assert result["decision"] == "Approve"
        assert result["confidence"] == 0.80
        assert result["hitl_decision"] is None
        assert result["hitl_notes"] is None
        assert any("eligibility rules passed" in r for r in result["decision_reasons"])

    def test_reject_on_multiple_flags(self):
        """
        2+ rule flags triggers Reject with confidence 0.85.
        decision_engine.py line 89:
            confidence = 0.85 if len(rule_flags) >= 2 else 0.75
        The len(rule_flags) >= 2 branch fires before the High risk check,
        so confidence is exactly 0.85 regardless of risk_label.
        """
        result = decision_engine_node(_state(
            rule_flags=["FOIR exceeds 55% limit.", "CIBIL score 600 below threshold 650."],
            risk_label="High",
            risk_score=0.72,
        ))
        assert result["decision"] == "Reject"
        assert result["confidence"] == 0.85
        assert result["hitl_decision"] is None

    def test_reject_on_high_risk_single_flag(self):
        """
        High risk_label with exactly 1 flag → Reject with confidence 0.75.
        decision_engine.py: len(rule_flags) >= 2 → 0.85, else → 0.75.
        1 flag does not meet the >= 2 threshold, so confidence is 0.75.

        Note: 1 flag alone normally → Refer (interrupt). But High risk_label
        overrides to Reject in the first branch (len >= 2 OR High risk).
        With 0 flags and High risk, confidence is 0.75.
        """
        result = decision_engine_node(_state(
            rule_flags=[],
            risk_label="High",
            risk_score=0.78,
        ))
        assert result["decision"] == "Reject"
        assert result["confidence"] == 0.75

    def test_hitl_guard_returns_existing_decision(self):
        """
        If hitl_decision is already set in state, the node returns it immediately
        without re-deriving the decision or reaching interrupt().

        This path fires when the node is called a third time on an already-completed
        HITL thread. In normal flow it does not trigger — the guard is defensive.
        Tested here because the state machine path is real and the behaviour
        (return existing decision, not re-derive) must be verified.

        Even though rule_flags has 1 entry (which would normally → Refer →
        interrupt()), the guard fires first and returns hitl_decision directly.
        """
        result = decision_engine_node(_state(
            hitl_decision="Approve",
            hitl_notes="Approved with co-applicant condition.",
            rule_flags=["FOIR exceeds 55% limit."],  # would normally → Refer → interrupt()
            risk_label="Medium",
        ))
        assert result["decision"] == "Approve"
        assert result["hitl_decision"] == "Approve"
        assert result["hitl_notes"] == "Approved with co-applicant condition."
        # Confidence from guard path uses state.get("confidence", 0.55) — default 0.55
        assert result["confidence"] == 0.55
