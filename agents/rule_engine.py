"""
agents/rule_engine.py
---------------------
Agent 3: Rule Engine

Applies hard RBI / NBFC eligibility rules deterministically.
No LLM call needed here — rules are pure Python. This is intentional:
rule checks must be deterministic and auditable for regulators.

Rules implemented (simplified RBI/NBFC guidelines):
  1. FOIR ≤ 55%  (Fixed Obligation to Income Ratio)
  2. Loan-to-income ≤ 5×
  3. CIBIL score ≥ 650
  4. Applicant age between 21 and 65
  5. Document confidence ≥ 0.4
  6. Negative net worth check (liabilities > 1.5× assets)

FIX — FOIR EMI calculation:
  Original code used `loan_amount * 0.10 / 12` — a rough approximation
  that assumes 10% rate over 12 months regardless of loan size or tenure.
  This is wrong: a ₹30L home loan has a 15–20 year tenure, not 1 year.

  Correct formula: standard reducing-balance EMI (annuity formula)
      EMI = P × r × (1+r)^n / ((1+r)^n - 1)
  where:
      P = principal (loan amount)
      r = monthly interest rate (annual_rate / 12)
      n = loan tenure in months

  We default to 8.5% p.a. / 240 months (20 years) for home loans,
  which is representative of Indian NBFC/bank rates for FY2024-25.

  Interview note: "I separated rule checks from ML scoring because rules
  must be auditable and explainable to regulators — you can't point a
  regulator at a gradient boosting model for a policy compliance question."
"""
import logging

from state import UnderwritingState
from config import cfg

logger = logging.getLogger(__name__)

# Default loan parameters for EMI estimation
# Used only when we don't know the actual loan product terms.
# Source: RBI average home loan rates FY2024-25
_DEFAULT_ANNUAL_RATE = 0.085   # 8.5% p.a.
_DEFAULT_TENURE_MONTHS = 240   # 20 years


def _calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    """
    Standard reducing-balance EMI (annuity formula).

        EMI = P × r × (1+r)^n / ((1+r)^n - 1)

    where r = monthly interest rate, n = tenure in months.

    Reference: RBI EMI calculation methodology
    https://rbi.org.in/Scripts/FAQView.aspx?Id=96

    Edge cases:
      - If annual_rate == 0 (interest-free loan): EMI = principal / tenure_months
      - If tenure_months == 0: returns 0.0 (guard against division by zero)
    """
    if tenure_months <= 0:
        return 0.0
    if annual_rate <= 0:
        return principal / tenure_months

    r = annual_rate / 12                     # monthly rate
    factor = (1 + r) ** tenure_months        # (1+r)^n
    return principal * r * factor / (factor - 1)


def rule_engine_node(state: UnderwritingState) -> dict:
    """
    Apply all eligibility rules. Collects ALL violations before returning
    so the applicant sees every issue at once, not one at a time.
    """
    logger.info(
        "[Agent 3: Rule Engine] Checking eligibility rules for %s",
        state["applicant_name"],
    )

    flags: list[str] = []

    # ── Income to use ──────────────────────────────────────────────────────
    # Prefer document-verified income over declared; fall back if zero
    verified_income = state.get("parsed_income") or state["annual_income"]
    monthly_income = verified_income / 12 if verified_income > 0 else 0.0

    # ── EMI estimate (reducing-balance formula) ────────────────────────────
    # We don't know the actual contracted rate or tenure at application stage,
    # so we use conservative NBFC defaults: 8.5% p.a., 20-year tenure.
    # This produces a realistic EMI estimate for FOIR calculation.
    estimated_new_emi = _calculate_emi(
        principal=state["loan_amount_requested"],
        annual_rate=_DEFAULT_ANNUAL_RATE,
        tenure_months=_DEFAULT_TENURE_MONTHS,
    )

    total_obligations = state["existing_obligations"] + estimated_new_emi
    foir = total_obligations / monthly_income if monthly_income > 0 else 1.0

    loan_to_income = (
        state["loan_amount_requested"] / verified_income
        if verified_income > 0
        else 99.0
    )

    # ── Rule 1: FOIR ──────────────────────────────────────────────────────
    if foir > cfg.max_foir:
        flags.append(
            f"FOIR {foir:.1%} exceeds maximum {cfg.max_foir:.0%}. "
            f"Total monthly obligations ₹{total_obligations:,.0f} "
            f"(existing ₹{state['existing_obligations']:,.0f} + "
            f"estimated new EMI ₹{estimated_new_emi:,.0f}) "
            f"against monthly income ₹{monthly_income:,.0f}."
        )

    # ── Rule 2: Loan-to-income ratio ──────────────────────────────────────
    if loan_to_income > cfg.max_loan_to_income_ratio:
        flags.append(
            f"Loan-to-income ratio {loan_to_income:.1f}× exceeds maximum "
            f"{cfg.max_loan_to_income_ratio:.0f}×. "
            f"Requested ₹{state['loan_amount_requested']:,.0f} against "
            f"annual income ₹{verified_income:,.0f}."
        )

    # ── Rule 3: CIBIL score ───────────────────────────────────────────────
    if state["cibil_score"] < cfg.min_cibil_score:
        flags.append(
            f"CIBIL score {state['cibil_score']} is below the minimum "
            f"threshold of {cfg.min_cibil_score}. "
            f"Please clear outstanding defaults before reapplying."
        )

    # ── Rule 4: Age ───────────────────────────────────────────────────────
    if not (cfg.min_age <= state["applicant_age"] <= cfg.max_age):
        flags.append(
            f"Applicant age {state['applicant_age']} is outside the eligible "
            f"range of {cfg.min_age}–{cfg.max_age} years."
        )

    # ── Rule 5: Document quality ──────────────────────────────────────────
    doc_conf = state.get("doc_confidence", 1.0)
    if doc_conf < 0.4:
        flags.append(
            f"Document confidence {doc_conf:.0%} is too low. "
            f"Please resubmit legible income proof "
            f"(Form 16 / ITR / salary slips)."
        )

    # ── Rule 6: Negative net worth ────────────────────────────────────────
    parsed_assets = state.get("parsed_assets", 0.0)
    parsed_liabilities = state.get("parsed_liabilities", 0.0)
    if parsed_liabilities > 0 and parsed_liabilities > parsed_assets * 1.5:
        flags.append(
            f"Declared liabilities (₹{parsed_liabilities:,.0f}) significantly "
            f"exceed assets (₹{parsed_assets:,.0f}). "
            f"Negative net worth is a high-risk indicator."
        )

    rule_passed = len(flags) == 0

    logger.info(
        "[Rule Engine] FOIR=%.3f (EMI=₹%.0f) LTI=%.2f flags=%d passed=%s",
        foir, estimated_new_emi, loan_to_income, len(flags), rule_passed,
    )

    return {
        "foir": round(foir, 4),
        "loan_to_income_ratio": round(loan_to_income, 4),
        "rule_flags": flags,
        "rule_passed": rule_passed,
    }
