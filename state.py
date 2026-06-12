"""
state.py
--------
Shared TypedDict state schema for the LangGraph underwriting graph.

Every agent node receives this full state and returns a PARTIAL dict —
only the keys it changes. LangGraph merges the partial return into the
running state.

document_bytes — checkpoint size fix (Bug 5):
  Raw PDF bytes stored in TypedDict state are serialised into EVERY
  checkpoint the checkpointer writes (after each node). With MemorySaver
  this wastes RAM proportional to PDF size × number of nodes (6 writes
  per application). With PostgresSaver (production) it inflates every
  checkpoint row and can exceed jsonplus serialisation limits for large PDFs.

  Fix: document_bytes is still accepted as an input field so graph.py /
  app.py do not need to change their call sites. But document_parser_node
  returns {"document_bytes": None} at the end of its execution, which
  LangGraph merges back into state — clearing the bytes from all subsequent
  checkpoints. The extracted text is preserved in document_text.

  This means the bytes are in state for exactly ONE checkpoint (before
  Agent 1 runs) and gone from all subsequent ones. The field stays in the
  TypedDict so the type system remains consistent.

cibil_score_override — demo/test determinism fix:
  The mock bureau pull in bureau_fetcher.py derives the CIBIL score from an
  MD5 hash of the applicant name, producing scores that do not match the
  documented sample-applicant outcomes (e.g. Rahul Sharma hashes to 429, not
  the ~760 needed for an Approve path). Injecting a deterministic score via
  this field is standard practice for systems with external API mocks and
  does not alter the graph architecture.

  When set to a non-None integer, bureau_fetcher_node substitutes this value
  for the hash-derived score. When None (default), the mock hash runs
  normally — correct behaviour for manually-entered applicants.

doc_parse_attempts — conditional re-routing counter:
  Tracks how many times document_parser has been invoked for this application.
  Used by the conditional edge after document_parser to decide whether to
  re-run parsing (when doc_confidence < 0.4 and attempts < MAX_DOC_PARSE_ATTEMPTS)
  or proceed to bureau_fetcher regardless.

  This is the field that makes the graph non-linear. Without it, LangGraph
  has no memory of how many times the node has already run, so the router
  function would loop forever on a genuinely unreadable document.

  Set to 0 in get_initial_state(); incremented by document_parser_node on
  each invocation.

  Reference: LangGraph conditional edges —
  https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges
"""
from __future__ import annotations
from typing import TypedDict, Optional


class UnderwritingState(TypedDict):
    # ── INPUT (set once at graph start) ─────────────────────────────────────
    applicant_name: str
    applicant_age: int
    annual_income: float
    loan_amount_requested: float
    loan_purpose: str
    employment_type: str
    existing_obligations: float
    document_text: str          # pre-pasted text (demo / sample applicants)
    document_bytes: Optional[bytes]  # raw PDF bytes — cleared after Agent 1 runs
    cibil_score_override: Optional[int]  # inject deterministic CIBIL for demo/tests; None = use mock hash

    # ── AGENT 1: Document Parser ─────────────────────────────────────────────
    doc_parse_attempts: int       # how many times document_parser has run; guards re-route loop
    parsed_income: float
    parsed_assets: float
    parsed_liabilities: float
    doc_confidence: float

    # ── AGENT 2: Bureau Score Fetcher ────────────────────────────────────────
    cibil_score: int
    bureau_report_summary: str
    active_loans_count: int
    payment_history_months: int

    # ── AGENT 3: Rule Engine ─────────────────────────────────────────────────
    foir: float
    loan_to_income_ratio: float
    rule_flags: list[str]
    rule_passed: bool

    # ── AGENT 4: Risk Scorer ─────────────────────────────────────────────────
    risk_score: float
    risk_label: str
    risk_features_used: list[str]

    # ── AGENT 5: Decision Engine ─────────────────────────────────────────────
    decision: str
    decision_reasons: list[str]
    confidence: float

    # ── HITL: Human Credit Officer ───────────────────────────────────────────
    hitl_decision: Optional[str]
    hitl_notes: Optional[str]

    # ── AGENT 6: Explainer ───────────────────────────────────────────────────
    explanation_en: str
    explanation_hi: str
    final_decision: str
