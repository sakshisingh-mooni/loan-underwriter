"""
evals/eval_llm_agents.py
------------------------
Automated evaluation of the three LLM-backed agents:
  - Agent 1: Document Parser    — extraction accuracy
  - Agent 6: Explainer (EN)     — letter coherence and completeness
  - Agent 6: Explainer (HI)     — Hindi language quality (deterministic checks)

WHY EVALS EXIST ALONGSIDE LANGFUSE
  Langfuse gives you observability: you can see what the LLM produced
  for a given run. Evals give you a *score*: a number that tells you
  whether the output was correct. Without evals you can answer
  "what did the parser return?" but not "is the parser accurate across
  a range of inputs?"

  This is the standard distinction in MLOps:
    observability  → traces, logs, latency (Langfuse covers this)
    evaluation     → accuracy, coherence, correctness (this file covers this)

  Reference: https://docs.smith.langchain.com/evaluation (LangSmith eval concepts
  also apply conceptually to any LLM eval framework)

APPROACH
  For document parsing and Hindi letter quality we use deterministic checks:
  ground-truth numeric comparison for parsing, and Devanagari character ratio +
  decision word + placeholder detection for Hindi quality. These are consistent
  across runs and avoid the LLM-as-judge reliability problem where the model
  invents failure reasons not present in the criteria.

  For English letter quality we use keyword presence checks (deterministic).

  For document parsing we use ground-truth comparison (exact/near-exact
  numeric match), which is more reliable than LLM-based evaluation for
  structured extraction tasks.

RUN
  python evals/eval_llm_agents.py

  Requires GROQ_API_KEY and LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
  to be set in .env (same as the main app).

OUTPUT
  Prints a table of scores to stdout. Exits with code 1 if any score
  falls below its threshold, making it suitable for CI/CD pipelines.

EXTENDING
  Add test cases to DOCUMENT_PARSER_CASES or EXPLAINER_CASES below.
  The eval harness picks them up automatically.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import textwrap

# Load .env so this script works when run directly (not via the app).
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ground-truth test cases for Agent 1 (Document Parser)
# ---------------------------------------------------------------------------
# Each case has:
#   document_text   — the input text the LLM will parse
#   expected        — the ground-truth values we compare against
#   tolerance_pct   — acceptable relative error for numeric fields (0.05 = 5%)
#
# Values chosen to be unambiguous in the document text so near-exact match
# is a reasonable expectation.

DOCUMENT_PARSER_CASES = [
    {
        "name": "Salaried — Form 16 style",
        "applicant_name": "Arjun Mehta",
        "annual_income_declared": 900_000.0,
        "document_text": textwrap.dedent("""
            Form 16 — Arjun Mehta (PAN: ABCPM9876Q)
            Employer: Wipro Technologies, Bengaluru
            Gross Salary: ₹9,00,000
            Deductions (80C, HRA): ₹1,50,000
            Net Taxable Income: ₹7,50,000

            Assets: Flat in Bengaluru valued at ₹45,00,000
            Savings & FD: ₹3,20,000

            Outstanding home loan: ₹12,00,000
            Credit card balance: ₹80,000
        """).strip(),
        "expected": {
            "parsed_income": 900_000.0,
            "parsed_assets": 4_820_000.0,    # 45L flat + 3.2L savings
            "parsed_liabilities": 1_280_000.0,  # 12L loan + 80k card
        },
        "tolerance_pct": 0.10,  # 10% — LLM may aggregate differently
    },
    {
        "name": "Self-employed — ITR style",
        "applicant_name": "Sneha Rao",
        "annual_income_declared": 720_000.0,
        "document_text": textwrap.dedent("""
            ITR-3 Summary — Sneha Rao (AY 2023-24)
            Business Income (Photography Studio): ₹72,000/month
            Annual Business Income: ₹8,64,000
            Expenses: ₹1,44,000
            Net Income: ₹7,20,000

            Business equipment (cameras, lighting): ₹4,50,000
            Current account balance: ₹95,000

            Business loan outstanding: ₹2,50,000 (EMI ₹12,000/month)
        """).strip(),
        "expected": {
            "parsed_income": 720_000.0,
            "parsed_assets": 545_000.0,     # 4.5L equipment + 95k balance
            "parsed_liabilities": 250_000.0,
        },
        "tolerance_pct": 0.10,
    },
]

# ---------------------------------------------------------------------------
# Test cases for Agent 6 (Explainer)
# ---------------------------------------------------------------------------
# Each case has:
#   state_fragment  — the minimal state dict needed to generate the letter
#   checks_en       — list of strings that MUST appear in the English letter
#   judge_criteria_hi — kept for documentation; no longer used in eval logic
#
# The English letter checks are keyword-based (fast, deterministic).
# The Hindi letter check uses deterministic Python checks (Devanagari ratio,
# decision word presence, placeholder detection) — see eval_explainer_hindi().

EXPLAINER_CASES = [
    {
        "name": "Approve — salaried",
        "state": {
            "applicant_name": "Arjun Mehta",
            "decision": "Approve",
            "loan_amount_requested": 2_500_000.0,
            "loan_purpose": "Home Purchase",
            "decision_reasons": [
                "All eligibility rules passed.",
                "CIBIL score 760 — satisfactory.",
                "Risk score 0.21 — Low risk.",
            ],
            "hitl_decision": None,
        },
        # The English letter for an Approve must mention next steps.
        "must_contain_en": ["disburse", "verif"],
        # The Hindi letter must be in Hindi (Devanagari) and mention the decision.
        "judge_criteria_hi": (
            "The text must be in Hindi (Devanagari script). "
            "It must clearly state that the loan was approved (अनुमोदित or approved). "
            "It must not contain any placeholder text like {name} or {decision}. "
            "It must be at least 50 words long."
        ),
    },
    {
        "name": "Reject — multiple rule failures",
        "state": {
            "applicant_name": "Vikram Singh",
            "decision": "Reject",
            "loan_amount_requested": 5_000_000.0,
            "loan_purpose": "Personal Loan",
            "decision_reasons": [
                "FOIR 228% exceeds maximum 55%.",
                "Loan-to-income ratio 13.9× exceeds maximum 5×.",
                "CIBIL score 580 is below the minimum threshold of 650.",
            ],
            "hitl_decision": None,
        },
        "must_contain_en": ["CIBIL", "FOIR", "improv"],
        "judge_criteria_hi": (
            "The text must be in Hindi (Devanagari script). "
            "It must clearly state that the loan was rejected (अस्वीकृत or rejected). "
            "It must mention at least one specific reason for rejection. "
            "It must not contain any placeholder text like {name} or {decision}. "
            "It must be at least 50 words long."
        ),
    },
]

# ---------------------------------------------------------------------------
# Minimum passing thresholds
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "doc_parser_field_accuracy": 0.70,   # fraction of numeric fields within tolerance
    "explainer_en_keyword_hit_rate": 0.80,  # fraction of required keywords found
    "explainer_hi_judge_pass_rate": 0.80,   # fraction of LLM-judge verdicts = PASS
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """
    Call the Groq LLM directly (same model as the app).
    Returns the response content string.
    """
    # Import here (not at module level) so the file can be imported by tests
    # without triggering config validation.
    from utils.llm import get_llm
    from langchain_core.messages import HumanMessage

    response = get_llm().invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def _extract_json_safe(text: str) -> dict:
    """Extract first JSON object from text, returning {} on failure."""
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


def _relative_error(predicted: float, expected: float) -> float:
    """Relative error as a fraction. Returns 0 if both are zero."""
    if expected == 0:
        return 0.0 if predicted == 0 else 1.0
    return abs(predicted - expected) / abs(expected)


# ---------------------------------------------------------------------------
# Eval 1: Document Parser extraction accuracy
# ---------------------------------------------------------------------------

def eval_document_parser() -> dict:
    """
    Run each DOCUMENT_PARSER_CASES test case through document_parser_node
    and compare extracted numeric fields to ground-truth values.

    Returns a result dict with per-case details and an aggregate accuracy score.

    The accuracy metric is: fraction of (field, case) pairs where the
    relative error is within the case's tolerance_pct.
    """
    from agents.document_parser import document_parser_node

    total_fields = 0
    passed_fields = 0
    case_results = []

    for case in DOCUMENT_PARSER_CASES:
        logger.info("[Eval DocParser] Running case: %s", case["name"])

        # Minimal state for document_parser_node
        state = {
            "applicant_name": case["applicant_name"],
            "annual_income": case["annual_income_declared"],
            "document_text": case["document_text"],
            "document_bytes": None,
            "doc_parse_attempts": 0,
        }

        result = document_parser_node(state)

        case_detail = {"name": case["name"], "fields": {}}
        tol = case["tolerance_pct"]

        for field, expected_val in case["expected"].items():
            predicted_val = float(result.get(field, 0.0))
            err = _relative_error(predicted_val, expected_val)
            passed = err <= tol
            total_fields += 1
            if passed:
                passed_fields += 1

            case_detail["fields"][field] = {
                "expected": expected_val,
                "predicted": predicted_val,
                "relative_error": round(err, 4),
                "passed": passed,
            }
            logger.info(
                "  %s | %s: expected=%.0f predicted=%.0f err=%.1f%% %s",
                case["name"], field, expected_val, predicted_val,
                err * 100, "✓" if passed else "✗",
            )

        # doc_confidence should be > 0 for a real document
        conf = float(result.get("doc_confidence", 0.0))
        case_detail["doc_confidence"] = conf
        if conf <= 0:
            logger.warning("  %s | doc_confidence=0 — LLM may have failed to parse.", case["name"])

        case_results.append(case_detail)

    accuracy = passed_fields / total_fields if total_fields > 0 else 0.0
    passed_threshold = accuracy >= THRESHOLDS["doc_parser_field_accuracy"]

    logger.info(
        "[Eval DocParser] field_accuracy=%.2f (%d/%d) threshold=%.2f → %s",
        accuracy, passed_fields, total_fields,
        THRESHOLDS["doc_parser_field_accuracy"],
        "PASS" if passed_threshold else "FAIL",
    )

    return {
        "eval": "document_parser",
        "field_accuracy": round(accuracy, 4),
        "passed_threshold": passed_threshold,
        "threshold": THRESHOLDS["doc_parser_field_accuracy"],
        "cases": case_results,
    }


# ---------------------------------------------------------------------------
# Eval 2: Explainer English letter — keyword checks
# ---------------------------------------------------------------------------

def eval_explainer_english() -> dict:
    """
    Generate English letters for each EXPLAINER_CASES test case and check
    that required keywords appear in the output.

    Keyword checks are fast (no LLM call) and catch the most obvious failures:
    - Approve letters that forget to mention next steps (disbursement, verification)
    - Reject letters that forget to mention improvement advice (CIBIL, FOIR)

    The check is case-insensitive substring matching.
    """
    from agents.explainer import explainer_node

    total_keywords = 0
    found_keywords = 0
    case_results = []

    for case in EXPLAINER_CASES:
        logger.info("[Eval Explainer EN] Running case: %s", case["name"])

        # Build minimal state for explainer_node
        state = dict(case["state"])
        state.setdefault("hitl_notes", None)
        state.setdefault("explanation_en", "")
        state.setdefault("explanation_hi", "")
        state.setdefault("final_decision", "")

        result = explainer_node(state)
        letter_en = result.get("explanation_en", "")

        keyword_results = {}
        for kw in case["must_contain_en"]:
            found = kw.lower() in letter_en.lower()
            keyword_results[kw] = found
            total_keywords += 1
            if found:
                found_keywords += 1
            logger.info(
                "  %s | keyword '%s': %s",
                case["name"], kw, "✓" if found else "✗",
            )

        case_results.append({
            "name": case["name"],
            "letter_en_snippet": letter_en[:200] + "..." if len(letter_en) > 200 else letter_en,
            "keyword_results": keyword_results,
        })

    hit_rate = found_keywords / total_keywords if total_keywords > 0 else 0.0
    passed_threshold = hit_rate >= THRESHOLDS["explainer_en_keyword_hit_rate"]

    logger.info(
        "[Eval Explainer EN] keyword_hit_rate=%.2f (%d/%d) threshold=%.2f → %s",
        hit_rate, found_keywords, total_keywords,
        THRESHOLDS["explainer_en_keyword_hit_rate"],
        "PASS" if passed_threshold else "FAIL",
    )

    return {
        "eval": "explainer_english",
        "keyword_hit_rate": round(hit_rate, 4),
        "passed_threshold": passed_threshold,
        "threshold": THRESHOLDS["explainer_en_keyword_hit_rate"],
        "cases": case_results,
    }


# ---------------------------------------------------------------------------
# Eval 3: Explainer Hindi letter — deterministic Python checks
# ---------------------------------------------------------------------------

def eval_explainer_hindi() -> dict:
    """
    Generate Hindi letters for each EXPLAINER_CASES test case and verify
    quality with deterministic Python checks — same logic as
    agents/explainer.py _verify_hindi_quality().

    Three checks per letter:
      1. >= 30% of non-whitespace chars are Devanagari (U+0900-U+097F).
         Tolerates English financial terms mixed in naturally.
      2. Contains the correct Hindi decision word.
      3. No unreplaced {placeholder} tokens remain.

    Replaces LLM-as-judge which was unreliably failing on legitimate
    letters — the model was treating applicant names like 'Arjun Mehta'
    as placeholder text.
    """
    from agents.explainer import explainer_node, _DECISION_HI

    total_cases = 0
    passed_cases = 0
    case_results = []

    for case in EXPLAINER_CASES:
        logger.info("[Eval Explainer HI] Running case: %s", case["name"])

        state = dict(case["state"])
        state.setdefault("hitl_notes", None)
        state.setdefault("explanation_en", "")
        state.setdefault("explanation_hi", "")
        state.setdefault("final_decision", "")

        result = explainer_node(state)
        letter_hi = result.get("explanation_hi", "")

        # ── Check 1: Devanagari character ratio ──────────────────────────
        non_ws = [c for c in letter_hi if not c.isspace()]
        devanagari_count = sum(1 for c in non_ws if '\u0900' <= c <= '\u097F')
        ratio = devanagari_count / len(non_ws) if non_ws else 0.0
        check1_passed = ratio >= 0.30
        check1_reason = f"Devanagari ratio {ratio:.0%} ({'OK' if check1_passed else 'too low, < 30%'})"

        # ── Check 2: Correct decision word ───────────────────────────────
        decision = state.get("hitl_decision") or state.get("decision", "")
        expected_word = _DECISION_HI.get(decision, "")
        check2_passed = bool(expected_word and expected_word in letter_hi)
        check2_reason = f"Decision word '{expected_word}' {'found' if check2_passed else 'NOT found'}"

        # ── Check 3: No unreplaced placeholders ──────────────────────────
        # Catches both {brace} style and [bracket] style template markers.
        # The name post-processing step in explainer_node already replaces
        # [आपका नाम], but this catches any remaining bracket placeholders.
        placeholders = re.findall(r'\{[a-zA-Z_]+\}', letter_hi)
        bracket_placeholders = re.findall(r'\[[^\]]{1,40}\]', letter_hi)
        all_placeholders = placeholders + bracket_placeholders
        check3_passed = len(all_placeholders) == 0
        check3_reason = f"Placeholders: {all_placeholders if all_placeholders else 'none'}"

        passed = check1_passed and check2_passed and check3_passed
        verdict = "PASS" if passed else "FAIL"
        reasons = " | ".join([check1_reason, check2_reason, check3_reason])

        total_cases += 1
        if passed:
            passed_cases += 1

        logger.info(
            "  %s | Hindi check verdict: %s — %s",
            case["name"], verdict, reasons,
        )

        case_results.append({
            "name": case["name"],
            "letter_hi_snippet": letter_hi[:200] + "..." if len(letter_hi) > 200 else letter_hi,
            "verdict": verdict,
            "reasons": reasons,
        })

    pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0
    passed_threshold = pass_rate >= THRESHOLDS["explainer_hi_judge_pass_rate"]

    logger.info(
        "[Eval Explainer HI] hi_check_pass_rate=%.2f (%d/%d) threshold=%.2f → %s",
        pass_rate, passed_cases, total_cases,
        THRESHOLDS["explainer_hi_judge_pass_rate"],
        "PASS" if passed_threshold else "FAIL",
    )

    return {
        "eval": "explainer_hindi",
        "judge_pass_rate": round(pass_rate, 4),
        "passed_threshold": passed_threshold,
        "threshold": THRESHOLDS["explainer_hi_judge_pass_rate"],
        "cases": case_results,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_evals() -> list[dict]:
    """Run all three evals and return a list of result dicts."""
    results = []

    logger.info("=" * 60)
    logger.info("RUNNING EVAL: Document Parser (extraction accuracy)")
    logger.info("=" * 60)
    results.append(eval_document_parser())

    logger.info("=" * 60)
    logger.info("RUNNING EVAL: Explainer — English letter (keyword checks)")
    logger.info("=" * 60)
    results.append(eval_explainer_english())

    logger.info("=" * 60)
    logger.info("RUNNING EVAL: Explainer — Hindi letter (deterministic checks)")
    logger.info("=" * 60)
    results.append(eval_explainer_hindi())

    return results


def _print_summary(results: list[dict]) -> bool:
    """Print a summary table. Returns True if all evals passed."""
    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    all_passed = True
    for r in results:
        eval_name = r["eval"]
        threshold = r["threshold"]

        # Pick the primary metric key (varies per eval)
        if "field_accuracy" in r:
            metric_name, metric_val = "field_accuracy", r["field_accuracy"]
        elif "keyword_hit_rate" in r:
            metric_name, metric_val = "keyword_hit_rate", r["keyword_hit_rate"]
        else:
            metric_name, metric_val = "judge_pass_rate", r["judge_pass_rate"]

        status = "PASS ✓" if r["passed_threshold"] else "FAIL ✗"
        if not r["passed_threshold"]:
            all_passed = False
        print(
            f"  {eval_name:<28} {metric_name}={metric_val:.2f}  "
            f"threshold={threshold:.2f}  {status}"
        )
    print("=" * 60)
    if all_passed:
        print("All evals passed.")
    else:
        print("One or more evals failed — review the logs above.")
    return all_passed


if __name__ == "__main__":
    # Ensure the project root is on sys.path when run directly.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    results = run_all_evals()
    all_passed = _print_summary(results)
    sys.exit(0 if all_passed else 1)
