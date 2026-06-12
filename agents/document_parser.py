"""
agents/document_parser.py
--------------------------
Agent 1: Document Parser

Extracts structured financial data from uploaded documents.

PDF EXTRACTION — reuses the RAG project's PyMuPDF pipeline:
  The RAG AI Assistant (separate project) uses PyMuPDF (PyMuPDFLoader)
  for layout-aware text extraction with noise-chunk filtering.
  This agent reuses the same extraction logic directly — same library,
  same _is_noisy_chunk() filter, same _clean_text() pass.

  Two input modes (selected automatically):
    1. PDF bytes in state["document_bytes"]  → PyMuPDF extraction
    2. Plain text in state["document_text"]  → used as-is (demo/fallback)

  The Streamlit app.py sends whichever is available. For the demo
  sample applicants, document_text is pre-populated. For a real upload,
  document_bytes is set from st.file_uploader().

Returns partial state update: parsed_income, parsed_assets,
parsed_liabilities, doc_confidence.

Interview note: "The document parser reuses the same PyMuPDF extraction
and noise-filtering logic from my RAG assistant project — same _is_noisy_chunk()
checks for TOC dot-leaders, short-token ratio, and sentence repetition.
The RAG project proved this pipeline on research PDFs; here it handles
salary certificates and ITRs, which have similar layout challenges."
"""
import json
import logging
import re
import tempfile
import os

from langchain_core.messages import HumanMessage

from state import UnderwritingState
from utils.llm import get_llm

logger = logging.getLogger(__name__)

# ── PyMuPDF text cleaning — same logic as RAG project's pipeline.py ──────────
# Copied here to keep the underwriting project self-contained.
# In a monorepo, this would be a shared utility imported from a common package.

_ARTIFACT_PATTERNS = re.compile(
    r"<EOS>|<pad>|<unk>|<s>|</s>|<mask>|<sep>|<cls>",
    flags=re.IGNORECASE,
)
_WHITESPACE_RUNS = re.compile(r"[ \t]{3,}")
_NEWLINE_RUNS = re.compile(r"\n{3,}")
_TOC_PATTERN = re.compile(r"\.{4,}")
_MIN_CHUNK_CHARS = 80
_MAX_NOISE_RATIO = 0.60
_MAX_REPEAT_RATIO = 0.50


def _is_noisy_chunk(text: str) -> bool:
    """
    Return True if text block should be dropped before LLM processing.
    Identical to RAG project's pipeline.py _is_noisy_chunk().
    Three checks: TOC dot-leaders, short-token noise ratio, sentence repetition.
    """
    tokens = text.split()
    if not tokens:
        return True
    lines = text.splitlines()
    if lines:
        toc_lines = sum(1 for line in lines if _TOC_PATTERN.search(line))
        if toc_lines / len(lines) > 0.4:
            return True
    short_tokens = sum(1 for t in tokens if len(t) <= 2)
    if short_tokens / len(tokens) > _MAX_NOISE_RATIO:
        return True
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 10]
    if len(sentences) > 2:
        unique = set(sentences)
        if 1 - (len(unique) / len(sentences)) > _MAX_REPEAT_RATIO:
            return True
    return False


def _clean_text(text: str) -> str:
    """Clean tokenizer artifacts and excess whitespace. Same as RAG pipeline.py."""
    text = _ARTIFACT_PATTERNS.sub(" ", text)
    text = _WHITESPACE_RUNS.sub(" ", text)
    text = _NEWLINE_RUNS.sub("\n\n", text)
    return text.strip()


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using PyMuPDF (PyMuPDFLoader).

    Reuses the RAG project's loader choice:
      PyMuPDF is layout-aware — handles multi-column salary certificates
      and ITRs better than pypdf (which loses column ordering).
      Same library, same reason: consistent extraction quality.

    Writes bytes to a tempfile because PyMuPDFLoader requires a path.
    Tempfile is deleted in the finally block — same pattern as RAG app.py.

    Reference: https://python.langchain.com/docs/integrations/document_loaders/pymupdfloader
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()

        # Filter noise chunks and join — same as RAG pipeline.py _clean_documents()
        clean_pages = []
        dropped = 0
        for doc in docs:
            cleaned = _clean_text(doc.page_content)
            if len(cleaned) >= _MIN_CHUNK_CHARS and not _is_noisy_chunk(cleaned):
                clean_pages.append(cleaned)
            else:
                dropped += 1

        if dropped:
            logger.info(
                "[Document Parser] Dropped %d/%d noisy pages from PDF.",
                dropped, len(docs),
            )

        extracted = "\n\n".join(clean_pages)

        if not extracted.strip():
            logger.warning(
                "[Document Parser] PDF extraction produced empty text. "
                "Document may be scanned/image-only — OCR not supported in demo."
            )
            return ""

        logger.info(
            "[Document Parser] PyMuPDF extracted %d chars from %d pages (%d clean).",
            len(extracted), len(docs), len(clean_pages),
        )
        return extracted

    except ImportError:
        logger.warning(
            "[Document Parser] PyMuPDF not installed. "
            "Run: pip install pymupdf. Falling back to document_text field."
        )
        return ""
    except Exception:
        logger.exception("[Document Parser] PDF extraction failed.")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _extract_json(text: str) -> dict:
    """Extract first JSON object from LLM response text."""
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {}


def document_parser_node(state: UnderwritingState) -> dict:
    """
    Parse documents and extract verified financial figures.

    Input priority:
      1. state["document_bytes"] — PDF uploaded via Streamlit file_uploader
         → extracted via PyMuPDF (same loader as RAG project)
      2. state["document_text"] — pre-pasted text (demo / sample applicants)

    The LLM receives the extracted text and returns structured JSON with
    income, assets, liabilities, and a confidence score.
    """
    logger.info(
        "[Agent 1: Document Parser] Processing documents for %s",
        state["applicant_name"],
    )

    # ── Determine document text source ────────────────────────────────────
    document_bytes: bytes | None = state.get("document_bytes")  # type: ignore[assignment]
    document_text: str = state.get("document_text", "")

    if document_bytes:
        logger.info("[Document Parser] PDF bytes present — using PyMuPDF extraction.")
        extracted = _extract_text_from_pdf(document_bytes)
        if extracted:
            document_text = extracted
        else:
            logger.warning(
                "[Document Parser] PyMuPDF returned empty — falling back to document_text."
            )

    if not document_text.strip():
        logger.warning(
            "[Document Parser] No document text available. "
            "Returning declared values with low confidence."
        )
        # document_bytes cleared: bytes have served their purpose (or were absent).
        # Returning None here merges into state and removes the raw bytes from
        # all subsequent checkpoints — fixing the checkpoint bloat bug (Bug 5).
        # doc_parse_attempts is incremented so the conditional router in graph.py
        # knows this run has been counted against the retry budget.
        return {
            "parsed_income": state["annual_income"],
            "parsed_assets": 0.0,
            "parsed_liabilities": 0.0,
            "doc_confidence": 0.3,
            "document_bytes": None,
            "doc_parse_attempts": state.get("doc_parse_attempts", 0) + 1,
        }

    # ── LLM extraction ────────────────────────────────────────────────────
    prompt = f"""You are a financial document parsing specialist for an Indian bank.
Extract structured financial information from the document text below.

APPLICANT: {state["applicant_name"]}
DECLARED ANNUAL INCOME: ₹{state["annual_income"]:,.0f}
DOCUMENT TEXT:
{document_text[:4000]}

Rules:
- Extract figures in INR only. If currency is unclear, assume INR.
- doc_confidence: 0.0 if document is vague/incomplete, 1.0 if all figures are clearly stated.
- If a figure is not mentioned, use the declared value as fallback.
- Truncated text is acceptable — extract from whatever is available.

Return ONLY valid JSON, no other text:
{{
  "parsed_income": <annual income in INR as number>,
  "parsed_assets": <total assets in INR as number>,
  "parsed_liabilities": <total liabilities in INR as number>,
  "doc_confidence": <0.0 to 1.0 as number>,
  "parser_notes": "<one-line summary of document quality>"
}}"""

    response = get_llm().invoke([HumanMessage(content=prompt)])
    data = _extract_json(response.content)

    if not data or "parsed_income" not in data:
        logger.warning(
            "[Document Parser] LLM extraction failed — falling back to declared values."
        )
        data = {
            "parsed_income": state["annual_income"],
            "parsed_assets": 0.0,
            "parsed_liabilities": 0.0,
            "doc_confidence": 0.5,
        }

    logger.info(
        "[Document Parser] income=₹%.0f assets=₹%.0f liabilities=₹%.0f confidence=%.2f",
        data.get("parsed_income", 0),
        data.get("parsed_assets", 0),
        data.get("parsed_liabilities", 0),
        data.get("doc_confidence", 0),
    )

    # Clear document_bytes from state: extraction is done, no need to persist
    # raw PDF bytes through all remaining checkpoints (Bug 5 fix).
    # Increment doc_parse_attempts so the conditional router in graph.py can
    # track retries and guarantee termination.
    return {
        "parsed_income": float(data.get("parsed_income", state["annual_income"])),
        "parsed_assets": float(data.get("parsed_assets", 0.0)),
        "parsed_liabilities": float(data.get("parsed_liabilities", 0.0)),
        "doc_confidence": float(data.get("doc_confidence", 0.5)),
        "document_bytes": None,
        "doc_parse_attempts": state.get("doc_parse_attempts", 0) + 1,
    }
