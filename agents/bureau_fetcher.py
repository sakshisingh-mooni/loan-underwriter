"""
agents/bureau_fetcher.py
------------------------
Agent 2: Bureau Score Fetcher
 
Simulates a CIBIL / Experian credit bureau API pull.
In production, replace _mock_bureau_pull() with your actual bureau API call.
Many fintechs (CRED, Navi, Slice) use the Experian or CRIF bureau APIs.
 
Returns: cibil_score, bureau_report_summary, active_loans_count,
         payment_history_months
"""
import logging
import hashlib
 
from state import UnderwritingState
 
logger = logging.getLogger(__name__)
 
 
def _mock_bureau_pull(applicant_name: str, annual_income: float) -> dict:
    """
    Deterministic mock bureau pull.
    Uses applicant name hash to produce consistent scores across runs —
    same applicant always gets the same score (useful for demo testing).
 
    Production replacement:
        import requests
        response = requests.post(
            "https://experian-bureau-api.example.com/score",
            json={"pan": pan_number, "name": applicant_name},
            headers={"Authorization": f"Bearer {BUREAU_API_KEY}"},
        )
        return response.json()
    """
    # Deterministic score from name hash — full CIBIL range 300–900.
    # Previously ranged 580–820, which skipped the entire poor-credit segment
    # (<580) and made the rule_engine CIBIL<650 branch unreachable from the mock.
    # CIBIL scores in India range 300–900 per TransUnion CIBIL documentation:
    # https://www.cibil.com/faq/what-is-credit-score
    name_hash = int(hashlib.md5(applicant_name.encode(), usedforsecurity=False).hexdigest(), 16)
    base_score = 300 + (name_hash % 551)  # 300 to 850
 
    # Income-adjusted: higher income applicants tend to have better scores in our mock
    income_boost = min(30, int(annual_income / 200_000))
    cibil_score = min(850, base_score + income_boost)
 
    # Derive other bureau fields from score
    active_loans = max(0, (base_score % 5))           # 0–4 active loans
    payment_months = min(120, (name_hash % 48) + 12)  # 12–120 months
 
    return {
        "cibil_score": cibil_score,
        "active_loans_count": active_loans,
        "payment_history_months": payment_months,
    }
 
 
def bureau_fetcher_node(state: UnderwritingState) -> dict:
    """
    Fetch credit bureau data for the applicant.
    Generates a human-readable report summary alongside the numeric score.

    cibil_score_override: if set in state, the hash-derived mock score is
    replaced with the override value. This makes demo sample applicants
    produce their documented outcomes deterministically without changing
    the graph architecture. For manually-entered applicants the field is
    None and the mock hash runs normally.
    """
    logger.info("[Agent 2: Bureau Fetcher] Pulling bureau data for %s", state["applicant_name"])
 
    bureau_data = _mock_bureau_pull(state["applicant_name"], state["annual_income"])

    # Apply deterministic override for demo/test applicants.
    # The override value is set in data/sample_applicants.py and threaded
    # through state so the mock hash does not break documented outcomes.
    override = state.get("cibil_score_override")
    if override is not None:
        logger.info(
            "[Bureau Fetcher] cibil_score_override=%d applied (mock hash was %d)",
            override, bureau_data["cibil_score"],
        )
        bureau_data["cibil_score"] = override
 
    score = bureau_data["cibil_score"]
    active_loans = bureau_data["active_loans_count"]
    payment_months = bureau_data["payment_history_months"]
 
    # Build human-readable summary (shown in UI and traced in Langfuse)
    if score >= 750:
        score_band = "Excellent"
    elif score >= 700:
        score_band = "Good"
    elif score >= 650:
        score_band = "Fair"
    else:
        score_band = "Poor"
 
    summary = (
        f"CIBIL Score: {score} ({score_band}). "
        f"Active loan accounts: {active_loans}. "
        f"Clean payment history: {payment_months} months. "
        f"{'No adverse entries found.' if score >= 650 else 'Adverse entries may be present — manual review recommended.'}"
    )
 
    logger.info("[Bureau Fetcher] score=%d band=%s active_loans=%d", score, score_band, active_loans)
 
    return {
        "cibil_score": score,
        "bureau_report_summary": summary,
        "active_loans_count": active_loans,
        "payment_history_months": payment_months,
    }