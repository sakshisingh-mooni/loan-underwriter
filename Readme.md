# Multi-Agent Loan Underwriting AI

A production-ready loan underwriting pipeline built with **LangGraph**, **Groq LLaMA 3.3-70B**, and **Langfuse observability**. The system routes each application through six specialised agents, pauses for a human credit officer on borderline cases (Human-in-the-Loop), and generates a formal decision letter in both English and Hindi.

---

## Architecture

```
Applicant Input
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  LangGraph Graph (StateGraph + PostgresSaver checkpointer)  │
│                                                             │
│  [1] Document Parser  →  PyMuPDF extraction + LLM parse    │
│          │                                                  │
│          ├── confidence ≥ 0.4 ──────────────────────────► [2]
│          └── confidence < 0.4, attempts < 2 ──► retry [1]  │
│                                                             │
│  [2] Bureau Fetcher   →  CIBIL score + payment history      │
│          │                                                  │
│  [3] Rule Engine      →  FOIR / LTI / age / doc quality     │
│          │              (pure Python, no LLM — auditable)   │
│  [4] Risk Scorer      →  GradientBoostingRegressor pipeline  │
│          │                                                  │
│  [5] Decision Engine  →  Approve / Refer / Reject           │
│          │                    │                             │
│          │              interrupt() ◄── Human Credit Officer│
│          │              Command(resume=...)                  │
│          │                                                  │
│  [6] Explainer        →  Bilingual letter (EN + HI)         │
│                           asyncio.gather() — parallel LLM   │
│                           Python quality checks (HI letter) │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
  Streamlit UI  +  FastAPI REST  +  Langfuse trace
```

**Non-linear routing:** The graph is not a strict pipeline. After Agent 1 (Document Parser), a conditional edge checks `doc_confidence`. If confidence is below 0.4 and the parser has run fewer than 2 times, the graph re-routes back to Agent 1 for a retry. After 2 attempts (or once confidence is sufficient), it always proceeds to Agent 2. This demonstrates LangGraph's `add_conditional_edges` API and ensures the graph always terminates.

**Checkpointer:** `MemorySaver` in development, `PostgresSaver` (Azure PostgreSQL) in production. PostgresSaver ensures HITL checkpoints survive container restarts on Azure App Service.

---
## Screenshots

| Form | Approve | HITL Panel |
|------|---------|------------|
| ![](screenshots/01_form.png) | ![](screenshots/02_approve_result.png) | ![](screenshots/03_hitl_panel.png) |

| Reject | Langfuse Traces | FastAPI Swagger |
|--------|----------------|-----------------|
| ![](screenshots/04_reject_result.png) | ![](screenshots/05_langfuse_traces.png) | ![](screenshots/06_swagger_ui.png) |

---
## Demo

### Human-in-the-Loop (HITL) — Graph pauses for credit officer review
![HITL Demo](screenshots/hitl_demo.gif)

---
### Full Demo (1:59)
[![Demo Video](screenshots/01_form.png)](https://youtu.be/lTuAOtFdx80)
## Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph 1.x (StateGraph, conditional edges, interrupt, Command) |
| LLM | Groq LLaMA 3.3-70B via langchain-groq |
| ML model | sklearn GradientBoostingRegressor Pipeline |
| Observability | Langfuse v2 (CallbackHandler, per-run traces) |
| LLM Evaluation | Deterministic Python checks (Hindi quality), keyword checks (English letter) |
| PDF extraction | PyMuPDF via langchain-community PyMuPDFLoader |
| Frontend | Streamlit |
| REST API | FastAPI + uvicorn |
| Production DB | Azure PostgreSQL Flexible Server (psycopg3 + psycopg-pool) |
| Deployment | Azure App Service B2 (Linux, Python 3.11) |

---

## Local setup

### 1. Prerequisites

- Python 3.10 or later
- A free [Groq API key](https://console.groq.com)
- A free [Langfuse account](https://cloud.langfuse.com) (project → Settings → API Keys)

### 2. Install

```bash
git clone <your-repo-url>
cd loan_underwriter

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Open .env and fill in GROQ_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
```

### 4. Train the risk model (optional)

The model trains automatically on first use via an inline fallback. To train the full model (10,000 samples, ~30s) and cache it:

```bash
python train_model.py
```

### 5. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Sample applicants

Three pre-built test cases are available in the sidebar, covering all three decision paths:

| Applicant | Expected outcome | What it demonstrates |
|-----------|-----------------|----------------------|
| Rahul Sharma | ✅ **Approve** | Clean salaried profile, good CIBIL, sensible LTI |
| Priya Patel | ⏸ **Refer** (HITL demo) | Self-employed, FOIR borderline — triggers human review panel |
| Vikram Singh | ❌ **Reject** | Multiple hard failures: FOIR, LTI, low CIBIL, irregular income |

---

## Human-in-the-Loop (HITL)

When the Decision Engine classifies a case as **Refer**, the graph pauses via LangGraph's `interrupt()`. Streamlit renders a credit officer review panel showing the agent's analysis. The officer submits Approve or Reject with notes, and the graph resumes from the exact pause point via `Command(resume={...})`. The Explainer then generates the final bilingual letter with the officer's decision.

This pattern uses the official LangGraph interrupt/resume API as documented at [langchain-ai.github.io/langgraph/concepts/human_in_the_loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/).

In production (Azure), `PostgresSaver` persists checkpoints so the HITL pause survives container restarts and cold starts. In development, `MemorySaver` is used — no database required.

---

## Observability

Every graph run produces a trace in [cloud.langfuse.com](https://cloud.langfuse.com) showing per-agent latency, LLM inputs/outputs, and token counts. The trace is tagged with the applicant name and session ID for filtering.

Langfuse is pinned to `langfuse==2.60.10` with the `langfuse-langchain==2.60.10.1` companion package. The companion package is required for Langfuse v2 + LangChain 1.x compatibility — without it, `langfuse.callback.CallbackHandler` raises `ModuleNotFoundError` because Langfuse v2 internally imports legacy LangChain modules removed in LangChain 1.x. v3 changed the CallbackHandler signature and v4 replaced the SDK entirely with an OpenTelemetry-native API. Neither is backwards compatible with the `CallbackHandler` usage in this project.

---

## Tests

```bash
pytest tests/ -v
```

`tests/test_rule_engine.py` covers 16 cases: EMI formula correctness (standard reducing-balance, zero-interest, zero-tenure, comparison against old approximation), FOIR boundary (pass / fail), loan-to-income ratio, CIBIL threshold (pass / fail), age boundaries (min, max, in-range), document confidence, negative net worth, clean-applicant pass, parsed income override, and multiple simultaneous violations.

---

## LLM Evaluation

Unit tests cover the deterministic rule engine. The LLM agents (document parser, explainer) are evaluated separately via `evals/eval_llm_agents.py`:

```bash
python evals/eval_llm_agents.py
```

Three evals run:

| Eval | Method | Metric | Threshold |
|------|--------|--------|-----------|
| Document Parser extraction accuracy | Ground-truth numeric comparison | Field accuracy | 70% |
| Explainer English letter | Keyword presence checks | Hit rate | 80% |
| Explainer Hindi letter | Devanagari ratio + keyword checks | Pass rate | 80% |

The Hindi eval uses deterministic Python checks to verify that the generated letter has ≥30% Devanagari characters (Unicode U+0900–U+097F), contains the correct decision word (अनुमोदित / अस्वीकृत / समीक्षाधीन), and has no unreplaced template placeholders. The `explainer_node` itself runs the same checks at inference time and substitutes a guaranteed-correct fallback template on failure. A name post-processing step replaces `[आपका नाम]`-style placeholders with the actual applicant name before any check runs.

Exits with code 1 if any eval falls below threshold — suitable for CI/CD pipelines.

---

## REST API

The FastAPI layer in `api.py` exposes the same graph as a REST endpoint for integration with Loan Origination Systems (LOS) or any backend service:

```bash
uvicorn api:app --reload --port 8080
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/underwrite` | POST | Submit application; returns decision or `refer_pending` |
| `/underwrite/resume` | POST | Submit officer decision for a Refer case |
| `/underwrite/{thread_id}` | GET | Poll the state of any thread |
| `/health` | GET | Liveness check |
| `/docs` | GET | Auto-generated OpenAPI / Swagger UI |

Example:

```bash
curl -X POST http://localhost:8080/underwrite \
  -H "Content-Type: application/json" \
  -d '{"applicant_name":"Arjun Mehta","applicant_age":32,"annual_income":1200000,
       "loan_amount_requested":3000000,"loan_purpose":"Home Purchase",
       "employment_type":"Salaried","existing_obligations":10000,
       "document_text":"Salary cert: Arjun Mehta, CTC 12L","cibil_score_override":760}'
```

---

## Project structure

```
loan_underwriter/
├── agents/
│   ├── document_parser.py    # Agent 1 — PDF extraction + LLM parse; increments doc_parse_attempts
│   ├── bureau_fetcher.py     # Agent 2 — CIBIL score (mock → replace with bureau API)
│   ├── rule_engine.py        # Agent 3 — deterministic RBI/NBFC eligibility rules
│   ├── risk_scorer.py        # Agent 4 — ML risk model tool call
│   ├── decision_engine.py    # Agent 5 — Approve/Refer/Reject + HITL interrupt
│   └── explainer.py          # Agent 6 — bilingual letter (EN + HI, parallel LLM + Hindi quality check)
├── evals/
│   └── eval_llm_agents.py    # LLM component evals: doc parser accuracy, explainer quality
├── utils/
│   ├── llm.py                # Cached ChatGroq factory (lru_cache)
│   ├── risk_model.py         # sklearn Pipeline loader + score_applicant()
│   └── tracing.py            # Langfuse v2 CallbackHandler setup
├── data/
│   └── sample_applicants.py  # Three demo applicants (Approve / Refer / Reject)
├── models/
│   └── .gitkeep              # Directory placeholder; .joblib generated at runtime
├── tests/
│   └── test_rule_engine.py   # 16 unit tests for the rule engine
├── app.py                    # Streamlit frontend
├── api.py                    # FastAPI REST layer (POST /underwrite, HITL resume, state polling)
├── graph.py                  # LangGraph StateGraph definition + conditional routing + checkpointer
├── state.py                  # UnderwritingState TypedDict schema (incl. doc_parse_attempts)
├── config.py                 # Lazy singleton config (no import-time side effects)
├── train_model.py            # One-time model training script
├── conftest.py               # pytest root configuration
├── startup.sh                # Azure App Service startup command
├── requirements.txt
├── .env.example
```