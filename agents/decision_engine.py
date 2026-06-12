"""
agents/decision_engine.py
--------------------------
Agent 5: Decision Engine
 
Synthesises all upstream signals into a final decision:
  - Approve: all rules pass + risk is Low
  - Refer:   borderline — needs human credit officer review (HITL)
  - Reject:  hard rule failure(s) or High risk score
 
Decision logic:
┌─────────────────────────────────────────────────────────────────┐
│ 2+ rule flags OR High risk     → Reject (high confidence)       │
│ 1 rule flag  OR Medium risk    → Refer  (needs officer review)   │
│ 0 flags AND  Low risk          → Approve                         │
└─────────────────────────────────────────────────────────────────┘
 
CIBIL below 650 is caught by the rule engine as a flag already.
No duplication here — single source of truth.
"""
import logging
 
from langgraph.types import interrupt
 
from state import UnderwritingState
 
logger = logging.getLogger(__name__)
 
 
def decision_engine_node(state: UnderwritingState) -> dict:
    """
    Make the primary underwriting decision, then HITL-gate Refer cases.
 
    How LangGraph interrupt() / resume works (reference):
      https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
 
    First pass (normal run):
      - hitl_decision is None in state (initial value from get_initial_state).
      - Guard below does NOT fire.
      - Decision logic runs; if Refer, interrupt(payload) raises internally,
        pausing the graph. The checkpoint is saved with the state AS IT WAS
        before this node ran — hitl_decision is still None in that checkpoint.
 
    Second pass (after Command(resume=val)):
      - LangGraph re-executes this node from the top with the saved checkpoint.
      - hitl_decision is STILL None in checkpoint state (the return dict that
        sets it has not run yet — it was never reached before interrupt() paused).
      - Guard below does NOT fire on normal resume.
      - Decision logic re-runs deterministically (same upstream state, same result).
      - interrupt(payload) is reached again; this time it returns val immediately
        without pausing — the resume value is injected here.
      - officer_input.get(...) extracts hitl_decision / hitl_notes from val.
      - Return dict is written; new checkpoint saved with hitl_decision set.
 
    When the guard DOES fire:
      - Only if this node is called a third time (e.g. thread_id reused after
        full completion) when hitl_decision is already non-None in state.
      - In normal Streamlit flow this doesn't happen: thread_id is reset on
        each new submission (app.py line 368).
      - Guard is a defensive safety net, not part of the normal resume path.
    """
    # ── Defensive guard: skip re-derivation if decision already finalised ──
    # Fires only when this node is called after a completed HITL cycle with
    # the same thread_id — not on normal first-pass or resume execution.
    # See docstring above for the full interrupt/resume lifecycle.
    if state.get("hitl_decision") is not None:
        logger.info(
            "[Decision Engine] Resume path — hitl_decision=%s already set, skipping re-derivation.",
            state["hitl_decision"],
        )
        return {
            "decision": state["hitl_decision"],
            "decision_reasons": state.get("decision_reasons", []),
            "confidence": state.get("confidence", 0.55),
            "hitl_decision": state["hitl_decision"],
            "hitl_notes": state.get("hitl_notes", ""),
        }
 
    logger.info("[Agent 5: Decision Engine] Making decision for %s", state["applicant_name"])
 
    rule_flags = state.get("rule_flags", [])
    risk_label = state.get("risk_label", "Medium")
    cibil = state.get("cibil_score", 0)
    risk_score = state.get("risk_score", 0.5)
 
    # ── Determine initial decision ────────────────────────────────────────
    if len(rule_flags) >= 2 or risk_label == "High":
        decision = "Reject"
        confidence = 0.85 if len(rule_flags) >= 2 else 0.75
        reasons = rule_flags[:] if rule_flags else []
        if risk_label == "High":
            reasons.append(f"ML risk score {risk_score:.2f} — classified High risk.")
 
    elif len(rule_flags) == 1 or risk_label == "Medium":
        decision = "Refer"
        confidence = 0.55
        reasons = rule_flags[:] + [
            f"Risk level: {risk_label} (score: {risk_score:.2f}) — borderline, requires officer review.",
        ]
 
    else:
        # 0 flags AND Low risk → clean approval
        decision = "Approve"
        confidence = 0.80
        reasons = [
            "All eligibility rules passed.",
            f"CIBIL score {cibil} — satisfactory.",
            f"Risk score {risk_score:.2f} — {risk_label} risk.",
            f"FOIR {state.get('foir', 0):.1%} within acceptable limits.",
        ]
 
    logger.info(
        "[Decision Engine] decision=%s confidence=%.2f flags=%d risk=%s cibil=%d",
        decision, confidence, len(rule_flags), risk_label, cibil,
    )
 
    # ── HITL: Pause graph for officer review on Refer cases ───────────────
    if decision == "Refer":
        logger.info("[Decision Engine] Refer case — triggering HITL interrupt")
 
        review_payload = {
            "applicant_name": state["applicant_name"],
            "loan_amount": state["loan_amount_requested"],
            "cibil_score": cibil,
            "risk_score": risk_score,
            "risk_label": risk_label,
            "foir": state.get("foir", 0),
            "rule_flags": rule_flags,
            "bureau_summary": state.get("bureau_report_summary", ""),
            "reasons_for_referral": reasons,
        }
 
        # interrupt() pauses the graph here. Execution resumes from the
        # START of this node when Command(resume=...) is called by Streamlit.
        # The re-execution guard at the top of this function handles the
        # second pass — it checks hitl_decision and returns immediately.
        # Reference: https://reference.langchain.com/python/langgraph/types/interrupt
        officer_input = interrupt(review_payload)
 
        hitl_decision = officer_input.get("hitl_decision", "Reject")
        hitl_notes = officer_input.get("hitl_notes", "No notes provided.")
 
        logger.info("[Decision Engine] HITL resumed with decision=%s", hitl_decision)
 
        reasons.append(f"Credit officer override: {hitl_notes}")
 
        return {
            "decision": hitl_decision,
            "decision_reasons": reasons,
            "confidence": confidence,
            "hitl_decision": hitl_decision,
            "hitl_notes": hitl_notes,
        }
 
    # ── Approve or Reject ──────────────────────────────────────────────────
    return {
        "decision": decision,
        "decision_reasons": reasons,
        "confidence": confidence,
        "hitl_decision": None,
        "hitl_notes": None,
    }