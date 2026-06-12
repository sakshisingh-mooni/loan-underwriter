"""
app.py
------
Streamlit frontend for the Multi-Agent Loan Underwriting Assistant.

Key UI flows:
  1. Normal (Approve/Reject): form → run graph → show results
  2. HITL (Refer): form → run graph (pauses) → show review panel →
     officer submits decision → resume graph → show results

Langfuse tracing:
  Every graph run is traced via CallbackHandler injected at invoke() time.
  Open cloud.langfuse.com to see agent-by-agent latency traces.

Run:
    streamlit run app.py
"""

import streamlit as st
import logging
import uuid

from dotenv import load_dotenv
load_dotenv()

from graph import build_graph, get_initial_state
from data.sample_applicants import SAMPLE_APPLICANTS
from config import cfg

# Langfuse import — v2 SDK (pinned langfuse>=2.0.0,<3.0.0 in requirements.txt)
try:
    from utils.tracing import get_langfuse_handler
    LANGFUSE_AVAILABLE = True
except Exception as e:
    LANGFUSE_AVAILABLE = False
    logging.warning("Langfuse not available: %s. Traces will not be sent.", e)
    
# LangGraph Command for HITL resume
from langgraph.types import Command

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Loan Underwriting AI",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Multi-Agent Loan Underwriting Assistant")
st.caption(
    "LangGraph · Groq LLaMA 3.3-70B · Langfuse Observability · Human-in-the-Loop"
)


if LANGFUSE_AVAILABLE:
    try:
        from utils.tracing import _verify_auth
        _verify_auth()
        st.success("✅ Langfuse tracing active — view traces at cloud.langfuse.com")
    except Exception as _lf_err:
        LANGFUSE_AVAILABLE = False
        st.warning(f"⚠️ Langfuse auth failed — traces will not be sent. ({_lf_err})")
else:
    st.warning("⚠️ Langfuse keys not configured — traces will not be sent.")

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "run_state" not in st.session_state:
    # "idle" | "hitl_pending" | "complete"
    st.session_state.run_state = "idle"

if "final_result" not in st.session_state:
    st.session_state.final_result = None

if "hitl_payload" not in st.session_state:
    st.session_state.hitl_payload = None

# ─────────────────────────────────────────────
# SIDEBAR — SAMPLE APPLICANTS
# ─────────────────────────────────────────────

with st.sidebar:
    st.subheader("🧪 Sample Applicants")
    st.caption("Click to pre-fill the form with test cases.")

    for label in SAMPLE_APPLICANTS:
        if st.button(label, use_container_width=True):
            st.session_state.sample = SAMPLE_APPLICANTS[label]
            st.session_state.run_state = "idle"
            st.session_state.final_result = None
            st.session_state.hitl_payload = None
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

    st.divider()
    st.markdown("**Expected outcomes:**")
    st.markdown("• Rahul Sharma → ✅ **Approve**")
    st.markdown("• Priya Patel → ⏸ **Refer** (HITL demo)")
    st.markdown("• Vikram Singh → ❌ **Reject**")

    st.divider()
    st.caption(
        "Each run creates a Langfuse trace. "
        "View all agent decisions at cloud.langfuse.com"
    )

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _get_config(session_id: str, user_id: str = "unknown") -> dict:
    """
    Build graph config with thread_id and optional Langfuse callbacks.

    get_langfuse_handler() now returns
    (CallbackHandler, metadata_dict). The metadata_dict contains
    "langfuse_session_id" and "langfuse_user_id" which LangGraph passes
    to the handler so they appear on the trace in the Langfuse UI.
    """
    config: dict = {"configurable": {"thread_id": st.session_state.thread_id}}
    if LANGFUSE_AVAILABLE:
        try:
            handler, lf_metadata = get_langfuse_handler(
                session_id=session_id,
                user_id=user_id,
            )
            config["callbacks"] = [handler]
            # Spread Langfuse session/user metadata into config["metadata"]
            # so traces are tagged correctly in the Langfuse dashboard.
            config["metadata"] = lf_metadata
        except Exception as e:
            logger.warning("Could not init Langfuse handler: %s", e)
    return config


def _decision_badge(decision: str) -> str:
    return {
        "Approve": "✅ **APPROVED**",
        "Reject": "❌ **REJECTED**",
        "Refer": "⏸ **REFERRED FOR REVIEW**",
    }.get(decision, f"**{decision}**")


def _show_results(result: dict):
    """Render the final underwriting result panel from FULL merged state."""
    decision = result.get("final_decision") or result.get("decision", "Unknown")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Decision", decision)
    with col2:
        st.metric("CIBIL Score", result.get("cibil_score", "—"))
    with col3:
        risk_label = result.get("risk_label", "—")
        risk_score = result.get("risk_score", 0)
        st.metric("Risk Score", f"{risk_label} ({risk_score:.2f})")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        foir = result.get("foir", 0)
        st.metric("FOIR", f"{foir:.1%}" if isinstance(foir, float) else "—")
    with col_b:
        lti = result.get("loan_to_income_ratio", 0)
        st.metric("Loan-to-Income", f"{lti:.1f}×" if isinstance(lti, float) else "—")

    st.divider()

    flags = result.get("rule_flags", [])
    if flags:
        st.subheader("⚠️ Rule Violations")
        for flag in flags:
            st.warning(flag)
    else:
        st.success("✅ All eligibility rules passed.")

    reasons = result.get("decision_reasons", [])
    if reasons:
        st.subheader("📋 Decision Reasons")
        for r in reasons:
            st.markdown(f"- {r}")

    if result.get("hitl_notes"):
        st.info(f"👤 **Credit Officer Notes:** {result['hitl_notes']}")

    st.divider()
    tab_en, tab_hi = st.tabs(["📄 English Letter", "📄 Hindi Letter (हिंदी)"])
    with tab_en:
        st.text_area("", result.get("explanation_en", "Generating..."), height=250, disabled=True)
    with tab_hi:
        st.text_area("", result.get("explanation_hi", "Generating..."), height=250, disabled=True)

    with st.expander("📊 Credit Bureau Report"):
        st.text(result.get("bureau_report_summary", "—"))
        col_x, col_y = st.columns(2)
        with col_x:
            st.metric("Active Loans", result.get("active_loans_count", "—"))
        with col_y:
            st.metric(
                "Clean Payment History",
                f"{result.get('payment_history_months', 0)} months",
            )

    if LANGFUSE_AVAILABLE:
        st.caption("🔍 Full agent trace available in your Langfuse dashboard.")


# ─────────────────────────────────────────────
# MAIN FORM
# ─────────────────────────────────────────────

sample = st.session_state.get("sample", {})

st.subheader("📝 Loan Application")

with st.form("application_form"):
    col1, col2 = st.columns(2)

    with col1:
        applicant_name = st.text_input(
            "Applicant Name", value=sample.get("applicant_name", "")
        )
        applicant_age = st.number_input(
            "Age", min_value=cfg.min_age, max_value=cfg.max_age,
            value=int(sample.get("applicant_age", 30)),
        )
        annual_income = st.number_input(
            "Annual Income (₹)", min_value=0.0,
            value=float(sample.get("annual_income", 600_000.0)),
            step=10_000.0, format="%.0f",
        )
        employment_type = st.selectbox(
            "Employment Type",
            ["Salaried", "Self-Employed", "Business"],
            index=["Salaried", "Self-Employed", "Business"].index(
                sample.get("employment_type", "Salaried")
            ),
        )

    with col2:
        loan_amount = st.number_input(
            "Loan Amount Requested (₹)", min_value=0.0,
            value=float(sample.get("loan_amount_requested", 1_000_000.0)),
            step=50_000.0, format="%.0f",
        )
        _purpose_options = [
            "Home Purchase", "Business Expansion", "Personal Loan", "Vehicle", "Education"
        ]
        _default_purpose = sample.get("loan_purpose", "Home Purchase")
        loan_purpose = st.selectbox(
            "Loan Purpose",
            _purpose_options,
            index=_purpose_options.index(_default_purpose)
            if _default_purpose in _purpose_options
            else 0,
        )
        existing_obligations = st.number_input(
            "Existing Monthly EMI Obligations (₹)", min_value=0.0,
            value=float(sample.get("existing_obligations", 0.0)),
            step=1_000.0, format="%.0f",
        )

    st.markdown("**Supporting Documents**")
    pdf_col, text_col = st.columns([1, 1])

    with pdf_col:
        uploaded_pdf = st.file_uploader(
            "Upload PDF (salary cert / ITR / bank statement)",
            type=["pdf"],
            help=(
                "Extracted via PyMuPDF — the same layout-aware loader "
                "used in the RAG AI Assistant project. "
                "Handles multi-column salary certificates and ITRs."
            ),
        )

    with text_col:
        document_text = st.text_area(
            "Or paste document text (for demo / sample applicants)",
            value=sample.get("document_text", "") if not uploaded_pdf else "",
            height=150,
            help="If a PDF is uploaded above, this field is ignored.",
            disabled=bool(uploaded_pdf),
        )

    if uploaded_pdf:
        st.caption(
            f"📄 **{uploaded_pdf.name}** ({uploaded_pdf.size / 1024:.1f} KB) "
            "— will be parsed via PyMuPDF on submit."
        )

    submitted = st.form_submit_button(
        "🚀 Run Underwriting Analysis", use_container_width=True
    )

# ─────────────────────────────────────────────
# GRAPH EXECUTION
# ─────────────────────────────────────────────

if submitted and st.session_state.run_state == "idle":
    if not applicant_name.strip():
        st.error("Please enter the applicant name.")
        st.stop()

    if annual_income <= 0:
        st.error("Annual income must be greater than zero.")
        st.stop()

    if loan_amount <= 0:
        st.error("Loan amount must be greater than zero.")
        st.stop()

    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.final_result = None
    st.session_state.hitl_payload = None

    # Read PDF bytes if uploaded — passed to document_parser_node
    # for PyMuPDF extraction (same loader as RAG AI Assistant project).
    # If no PDF, document_text (pasted string) is used as fallback.
    pdf_bytes: bytes | None = uploaded_pdf.read() if uploaded_pdf else None

    initial_state = get_initial_state(
        applicant_name=applicant_name.strip(),
        applicant_age=int(applicant_age),
        annual_income=float(annual_income),
        loan_amount_requested=float(loan_amount),
        loan_purpose=loan_purpose,
        employment_type=employment_type,
        existing_obligations=float(existing_obligations),
        document_text=document_text.strip(),
        document_bytes=pdf_bytes,           # PDF path (PyMuPDF extraction)
        # cibil_score_override: present in sample applicant dicts to produce
        # documented demo outcomes; None for manually-entered applicants.
        cibil_score_override=sample.get("cibil_score_override"),
    )

    config = _get_config(
        session_id=st.session_state.thread_id,
        user_id=applicant_name.strip() or "unknown",
    )

    with st.spinner("🤖 Agents are processing the application..."):
        progress_placeholder = st.empty()
        progress_steps = []

        try:
            for event in st.session_state.graph.stream(initial_state, config=config):
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        # ── HITL interrupt triggered ──────────────────────
                        interrupt_value = node_output
                        if (
                            isinstance(interrupt_value, (list, tuple))
                            and len(interrupt_value) > 0
                        ):
                            item = interrupt_value[0]
                            payload = item.value if hasattr(item, "value") else item
                        else:
                            payload = interrupt_value

                        st.session_state.run_state = "hitl_pending"
                        st.session_state.hitl_payload = payload
                        progress_steps.append("⏸ Refer — awaiting credit officer review")
                        break
                    else:
                        step_map = {
                            "document_parser": "✅ Agent 1: Documents parsed",
                            "bureau_fetcher": "✅ Agent 2: Bureau score fetched",
                            "rule_engine": "✅ Agent 3: Eligibility rules checked",
                            "risk_scorer": "✅ Agent 4: Risk model scored",
                            "decision_engine": "✅ Agent 5: Decision made",
                            "explainer": "✅ Agent 6: Letters generated",
                        }
                        if node_name in step_map:
                            progress_steps.append(step_map[node_name])

                        if node_name == "explainer":
                            st.session_state.run_state = "complete"

                with progress_placeholder.container():
                    for step in progress_steps:
                        st.markdown(step)

                if st.session_state.run_state in ("hitl_pending", "complete"):
                    break

            if st.session_state.run_state == "complete":
                full_state = st.session_state.graph.get_state(config)
                st.session_state.final_result = dict(full_state.values)

        except Exception as e:
            st.error(f"Graph execution failed: {type(e).__name__}: {e}")
            logger.exception("Graph execution error")
            st.stop()

# ─────────────────────────────────────────────
# HITL REVIEW PANEL
# ─────────────────────────────────────────────

if st.session_state.run_state == "hitl_pending":
    payload = st.session_state.hitl_payload or {}

    st.divider()
    st.subheader("⏸ HUMAN REVIEW REQUIRED — Referred Case")
    st.info(
        "This application has been referred for manual review. "
        "The underwriting agent has paused and is awaiting your decision."
    )

    with st.expander("📋 Agent Analysis Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CIBIL Score", payload.get("cibil_score", "—"))
        with col2:
            st.metric("Risk Label", payload.get("risk_label", "—"))
        with col3:
            st.metric("Risk Score", f"{payload.get('risk_score', 0):.2f}")

        st.markdown("**Reasons for referral:**")
        for r in payload.get("reasons_for_referral", []):
            st.markdown(f"- {r}")

        if payload.get("rule_flags"):
            st.markdown("**Rule flags:**")
            for f in payload.get("rule_flags", []):
                st.warning(f)

        st.text(payload.get("bureau_summary", ""))

    st.subheader("👤 Credit Officer Decision")
    with st.form("hitl_form"):
        officer_decision = st.radio(
            "Your Decision",
            ["Approve", "Reject"],
            horizontal=True,
        )
        officer_notes = st.text_area(
            "Notes (reason for override / additional conditions)",
            placeholder="e.g. 'Approved with condition: co-applicant required'",
            height=100,
        )
        hitl_submitted = st.form_submit_button(
            "✅ Submit Decision & Resume", use_container_width=True
        )

    if hitl_submitted:
        config = _get_config(
            session_id=st.session_state.thread_id,
            user_id=st.session_state.get("hitl_payload", {}).get("applicant_name", "unknown"),
        )

        with st.spinner("Resuming agents with officer decision..."):
            try:
                for event in st.session_state.graph.stream(
                    Command(resume={
                        "hitl_decision": officer_decision,
                        "hitl_notes": officer_notes or "No notes.",
                    }),
                    config=config,
                ):
                    for node_name, node_output in event.items():
                        if node_name == "explainer":
                            st.session_state.run_state = "complete"
                    if st.session_state.run_state == "complete":   # ← add this
                        break

                # ── FIX: capture full state after HITL resume ─────────────
                if st.session_state.run_state == "complete":
                    full_state = st.session_state.graph.get_state(config)
                    st.session_state.final_result = dict(full_state.values)

            except Exception as e:
                st.error(f"Resume failed: {type(e).__name__}: {e}")
                logger.exception("HITL resume error")
                st.stop()

        st.rerun()

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

if st.session_state.run_state == "complete" and st.session_state.final_result:
    st.divider()
    result = st.session_state.final_result
    decision = result.get("final_decision") or result.get("decision", "Unknown")
    st.subheader(f"📊 Underwriting Result — {_decision_badge(decision)}")
    _show_results(result)

    if st.button("🔄 New Application", use_container_width=True):
        st.session_state.run_state = "idle"
        st.session_state.final_result = None
        st.session_state.hitl_payload = None
        st.session_state.sample = {}
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
