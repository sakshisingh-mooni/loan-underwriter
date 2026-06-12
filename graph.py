"""
graph.py
--------
LangGraph graph definition for the multi-agent loan underwriting system.

Checkpointer selection:
  development (default) → MemorySaver (no DB needed)
  production (APP_ENV=production) → PostgresSaver (Azure PostgreSQL)

ADDED: document_bytes parameter in get_initial_state()
  Passes raw PDF bytes from Streamlit file_uploader to the document_parser
  agent, which extracts text via PyMuPDF — the same loader used in the
  RAG AI Assistant project.

Graph topology (non-linear):
  START → document_parser ──── confidence OK ────────────────► bureau_fetcher
                           └─── low confidence, attempts < 2 ──► document_parser (retry)
                           └─── low confidence, attempts >= 2 ──► bureau_fetcher (proceed anyway)
        bureau_fetcher → rule_engine → risk_scorer → decision_engine
          ↑ interrupt() on Refer cases (HITL)
        decision_engine → explainer → END

The conditional edge after document_parser demonstrates LangGraph's
non-linear routing capability. The doc_parse_attempts counter in state
prevents infinite loops — the graph always proceeds after MAX_DOC_PARSE_ATTEMPTS
regardless of confidence.

Reference: LangGraph conditional edges —
https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges
"""
import logging
import os
from typing import Optional, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import UnderwritingState
from agents.document_parser import document_parser_node
from agents.bureau_fetcher import bureau_fetcher_node
from agents.rule_engine import rule_engine_node
from agents.risk_scorer import risk_scorer_node
from agents.decision_engine import decision_engine_node
from agents.explainer import explainer_node

logger = logging.getLogger(__name__)

# Maximum number of document parse attempts before proceeding regardless of confidence.
# On the first attempt the LLM may return low confidence because the document
# text is sparse or structured unusually. A single retry with the same text
# (or with a degraded-gracefully fallback) often yields better extraction.
# Two attempts is sufficient — beyond that we accept the low-confidence result
# and let the rule engine flag it (Rule 5: doc_confidence < 0.4).
MAX_DOC_PARSE_ATTEMPTS = 2

# Minimum doc_confidence to skip the retry branch.
# Matches the Rule 5 threshold in rule_engine.py so the routing decision and
# the rule flag are consistent with the same boundary.
MIN_DOC_CONFIDENCE_TO_PROCEED = 0.4


def _doc_parser_router(
    state: UnderwritingState,
) -> Literal["bureau_fetcher", "document_parser"]:
    """
    Conditional edge function called after document_parser_node completes.

    Returns:
      "document_parser"  — re-run parsing (low confidence, have retries left)
      "bureau_fetcher"   — proceed with pipeline (confidence OK, or retries exhausted)

    Design notes:
    - doc_parse_attempts is incremented by document_parser_node itself, so by
      the time this router runs the counter already reflects the just-completed run.
    - Routing back to "document_parser" lets the agent retry with the same text;
      in a real system you might inject a different extraction prompt or a higher
      OCR threshold on retry. For this demo the retry uses the same logic, which
      means confidence typically improves when the first attempt was a transient
      LLM stochasticity event (temperature 0.1 but not 0).
    - We never route to "document_parser" when attempts >= MAX_DOC_PARSE_ATTEMPTS
      to guarantee the graph terminates.

    Reference: LangGraph conditional edges —
    https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges
    """
    confidence = state.get("doc_confidence", 0.0)
    attempts = state.get("doc_parse_attempts", 1)

    if confidence >= MIN_DOC_CONFIDENCE_TO_PROCEED:
        logger.info(
            "[Graph Router] doc_confidence=%.2f >= %.2f — proceeding to bureau_fetcher.",
            confidence, MIN_DOC_CONFIDENCE_TO_PROCEED,
        )
        return "bureau_fetcher"

    if attempts < MAX_DOC_PARSE_ATTEMPTS:
        logger.info(
            "[Graph Router] doc_confidence=%.2f < %.2f, attempt %d/%d — retrying document_parser.",
            confidence, MIN_DOC_CONFIDENCE_TO_PROCEED, attempts, MAX_DOC_PARSE_ATTEMPTS,
        )
        return "document_parser"

    logger.info(
        "[Graph Router] doc_confidence=%.2f < %.2f but attempts=%d/%d exhausted — "
        "proceeding to bureau_fetcher. Rule engine will flag low confidence.",
        confidence, MIN_DOC_CONFIDENCE_TO_PROCEED, attempts, MAX_DOC_PARSE_ATTEMPTS,
    )
    return "bureau_fetcher"


def _build_checkpointer():
    """
    Return the appropriate checkpointer based on APP_ENV.
    development  → MemorySaver
    production   → PostgresSaver (Azure PostgreSQL)
    Reference: https://pypi.org/project/langgraph-checkpoint-postgres/
    """
    app_env = os.environ.get("APP_ENV", "development")

    if app_env == "production":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg_pool import ConnectionPool
            from psycopg.rows import dict_row

            db_url = os.environ.get("DATABASE_URL", "")
            if not db_url:
                logger.error(
                    "[Graph] APP_ENV=production but DATABASE_URL not set. "
                    "Falling back to MemorySaver."
                )
                return MemorySaver()

            # PostgresSaver.from_conn_string() is a @contextmanager — it closes
            # the connection when you exit the `with` block, making it unusable
            # for a long-lived Streamlit app.
            #
            # Correct pattern for a persistent app: ConnectionPool.
            # Reference: https://pypi.org/project/langgraph-checkpoint-postgres/
            #
            # DATABASE_URL must be a libpq URI (postgresql://...) —
            # NOT the SQLAlchemy dialect prefix (postgresql+psycopg://...).
            pool = ConnectionPool(
                conninfo=db_url,
                max_size=10,
                kwargs={"autocommit": True, "row_factory": dict_row},
            )
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            logger.info("[Graph] Using PostgresSaver with ConnectionPool (Azure PostgreSQL)")
            return checkpointer

        except ImportError:
            logger.error(
                "[Graph] langgraph-checkpoint-postgres or psycopg-pool not installed. "
                "Falling back to MemorySaver."
            )
            return MemorySaver()

        except Exception as exc:
            logger.error(
                "[Graph] Failed to connect to PostgreSQL: %s. "
                "Falling back to MemorySaver.",
                exc,
            )
            return MemorySaver()

    logger.info("[Graph] Using MemorySaver (development mode)")
    return MemorySaver()


def build_graph():
    """
    Build and compile the underwriting graph.

    The graph is non-linear: document_parser can be retried once if it
    returns low confidence, controlled by _doc_parser_router().
    All other edges are fixed (linear).
    """
    builder = StateGraph(UnderwritingState)

    builder.add_node("document_parser", document_parser_node)
    builder.add_node("bureau_fetcher", bureau_fetcher_node)
    builder.add_node("rule_engine", rule_engine_node)
    builder.add_node("risk_scorer", risk_scorer_node)
    builder.add_node("decision_engine", decision_engine_node)
    builder.add_node("explainer", explainer_node)

    builder.add_edge(START, "document_parser")

    # Conditional edge: re-run document_parser on low confidence (max once),
    # then always continue to bureau_fetcher.
    builder.add_conditional_edges(
        "document_parser",
        _doc_parser_router,
        {
            "document_parser": "document_parser",
            "bureau_fetcher": "bureau_fetcher",
        },
    )

    builder.add_edge("bureau_fetcher", "rule_engine")
    builder.add_edge("rule_engine", "risk_scorer")
    builder.add_edge("risk_scorer", "decision_engine")
    builder.add_edge("decision_engine", "explainer")
    builder.add_edge("explainer", END)

    checkpointer = _build_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)
    logger.info("[Graph] Compiled. Nodes: %s", list(builder.nodes.keys()))
    return graph


def get_initial_state(
    applicant_name: str,
    applicant_age: int,
    annual_income: float,
    loan_amount_requested: float,
    loan_purpose: str,
    employment_type: str,
    existing_obligations: float,
    document_text: str,
    document_bytes: Optional[bytes] = None,
    cibil_score_override: Optional[int] = None,
) -> UnderwritingState:
    """
    Construct a clean initial state.

    document_bytes: raw PDF bytes from st.file_uploader().
      If provided, document_parser_node extracts text via PyMuPDF
      (same loader as RAG AI Assistant project).
      If None, document_text (pre-pasted string) is used instead.

    cibil_score_override: inject a deterministic CIBIL score for the demo
      sample applicants, bypassing the MD5-hash mock in bureau_fetcher_node.
      None (default) means the mock hash runs normally.
    """
    return UnderwritingState(
        # Input
        applicant_name=applicant_name,
        applicant_age=applicant_age,
        annual_income=annual_income,
        loan_amount_requested=loan_amount_requested,
        loan_purpose=loan_purpose,
        employment_type=employment_type,
        existing_obligations=existing_obligations,
        document_text=document_text,
        document_bytes=document_bytes,
        cibil_score_override=cibil_score_override,
        # Agent 1 defaults
        doc_parse_attempts=0,     # incremented by document_parser_node on each run
        parsed_income=0.0,
        parsed_assets=0.0,
        parsed_liabilities=0.0,
        doc_confidence=0.0,
        # Agent 2 defaults
        cibil_score=0,
        bureau_report_summary="",
        active_loans_count=0,
        payment_history_months=0,
        # Agent 3 defaults
        foir=0.0,
        loan_to_income_ratio=0.0,
        rule_flags=[],
        rule_passed=False,
        # Agent 4 defaults
        risk_score=0.0,
        risk_label="Unknown",
        risk_features_used=[],
        # Agent 5 defaults
        decision="",
        decision_reasons=[],
        confidence=0.0,
        # HITL defaults
        hitl_decision=None,
        hitl_notes=None,
        # Agent 6 defaults
        explanation_en="",
        explanation_hi="",
        final_decision="",
    )
