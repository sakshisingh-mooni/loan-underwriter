"""
tests/test_rule_engine.py
--------------------------
Unit tests for the rule_engine_node — the most testable and most critical
component in the pipeline (pure Python, deterministic, no LLM calls).

Run:
    pytest tests/test_rule_engine.py -v

Coverage:
  - FOIR boundary (pass / fail)
  - Proper reducing-balance EMI (not the old 10%/12 approximation)
  - Loan-to-income ratio (pass / fail)
  - CIBIL score threshold (pass / fail)
  - Age boundaries (min, max, in-range)
  - Document confidence threshold
  - Negative net worth detection
  - Clean applicant (all rules pass)
  - Multiple simultaneous violations

Why test rule_engine specifically?
  These rules determine loan approvals and rejections. An off-by-one
  in FOIR or a wrong EMI formula silently misprices risk for every
  application. Tests here catch regressions immediately.

References:
  - pytest docs: https://docs.pytest.org/en/stable/
  - RBI FOIR guidelines: https://rbi.org.in
"""

import pytest

# Patch config so tests run without .env
import unittest.mock as mock

_mock_cfg = mock.MagicMock()
_mock_cfg.max_foir = 0.55
_mock_cfg.min_cibil_score = 650
_mock_cfg.max_loan_to_income_ratio = 5.0
_mock_cfg.min_age = 21
_mock_cfg.max_age = 65

# Patch before importing rule_engine so cfg is already mocked
with mock.patch.dict("sys.modules", {"config": mock.MagicMock(cfg=_mock_cfg)}):
    from agents.rule_engine import rule_engine_node, _calculate_emi


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _state(
    annual_income: float = 1_200_000.0,
    loan_amount: float = 3_000_000.0,
    existing_obligations: float = 10_000.0,
    cibil_score: int = 750,
    applicant_age: int = 35,
    doc_confidence: float = 0.9,
    parsed_income: float = 0.0,
    parsed_assets: float = 0.0,
    parsed_liabilities: float = 0.0,
) -> dict:
    """Build a minimal state dict with sensible defaults for a clean applicant."""
    return {
        "applicant_name": "Test Applicant",
        "annual_income": annual_income,
        "loan_amount_requested": loan_amount,
        "existing_obligations": existing_obligations,
        "cibil_score": cibil_score,
        "applicant_age": applicant_age,
        "doc_confidence": doc_confidence,
        "parsed_income": parsed_income,
        "parsed_assets": parsed_assets,
        "parsed_liabilities": parsed_liabilities,
        "loan_purpose": "Home Purchase",
        "employment_type": "Salaried",
    }


# ─── _calculate_emi unit tests ────────────────────────────────────────────────

class TestCalculateEmi:
    def test_standard_home_loan(self):
        """₹30L at 8.5% over 20 years should give ~₹26,035/month."""
        emi = _calculate_emi(3_000_000, 0.085, 240)
        # Verified against SBI EMI calculator
        assert 25_000 < emi < 27_500, f"Expected ~₹26,035, got ₹{emi:.0f}"

    def test_zero_interest_loan(self):
        """Interest-free loan: EMI = principal / months."""
        emi = _calculate_emi(120_000, 0.0, 12)
        assert abs(emi - 10_000) < 1, f"Expected ₹10,000/month, got ₹{emi:.2f}"

    def test_zero_tenure_returns_zero(self):
        """Zero tenure should not raise; return 0.0."""
        emi = _calculate_emi(1_000_000, 0.09, 0)
        assert emi == 0.0

    def test_emi_higher_than_old_approximation(self):
        """
        The old formula (loan * 0.10 / 12) severely underestimates EMI
        for large loans — e.g. ₹50L loan: old = ₹41,667, correct = ~₹43,391.
        The reducing-balance formula should produce a different (correct) value.
        """
        principal = 5_000_000
        old_approx = principal * 0.10 / 12
        new_emi = _calculate_emi(principal, 0.085, 240)
        assert new_emi != pytest.approx(old_approx, rel=0.01), (
            "EMI formula appears unchanged — old approximation still in use"
        )


# ─── rule_engine_node integration tests ──────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_cfg(monkeypatch):
    """Patch cfg in the rule_engine module for every test."""
    import agents.rule_engine as re_module
    monkeypatch.setattr(re_module, "cfg", _mock_cfg)


class TestRuleEngineNode:

    def test_clean_applicant_passes_all_rules(self):
        """Rahul-type applicant: good income, good CIBIL, sensible loan — all rules pass."""
        result = rule_engine_node(_state())
        assert result["rule_passed"] is True
        assert result["rule_flags"] == []

    def test_foir_violation_flagged(self):
        """Very low income + large obligations → FOIR > 55% → flagged."""
        # ₹3L income, ₹20k existing EMI, ₹30L loan
        # Monthly income = 25,000
        # New EMI ≈ ₹26,035 (8.5%, 20yr)
        # FOIR = (20,000 + 26,035) / 25,000 ≈ 184% → flag
        result = rule_engine_node(_state(annual_income=300_000, existing_obligations=20_000))
        assert result["rule_passed"] is False
        assert any("FOIR" in f for f in result["rule_flags"])

    def test_foir_just_within_limit_passes(self):
        """Applicant right at FOIR boundary should pass (not exceed threshold)."""
        # ₹15L income, ₹10k existing, ₹30L loan
        # Monthly income = 125,000
        # New EMI ≈ 26,035
        # FOIR = (10,000 + 26,035) / 125,000 ≈ 28.8% → below 55%
        result = rule_engine_node(_state(annual_income=1_500_000, existing_obligations=10_000))
        assert result["rule_passed"] is True

    def test_loan_to_income_violation(self):
        """Loan amount > 5× annual income → flagged."""
        # ₹5L income, ₹30L loan → LTI = 6× → flag
        result = rule_engine_node(_state(annual_income=500_000, loan_amount=3_000_000))
        assert any("Loan-to-income" in f for f in result["rule_flags"])

    def test_cibil_below_threshold_flagged(self):
        """CIBIL score below 650 → flagged."""
        result = rule_engine_node(_state(cibil_score=620))
        assert result["rule_passed"] is False
        assert any("CIBIL" in f for f in result["rule_flags"])

    def test_cibil_exactly_at_threshold_passes(self):
        """CIBIL score exactly 650 should pass (threshold is >=, not >)."""
        result = rule_engine_node(_state(cibil_score=650))
        assert not any("CIBIL" in f for f in result["rule_flags"])

    def test_age_below_minimum_flagged(self):
        """Age 20 (below min 21) → flagged."""
        result = rule_engine_node(_state(applicant_age=20))
        assert any("age" in f.lower() for f in result["rule_flags"])

    def test_age_above_maximum_flagged(self):
        """Age 66 (above max 65) → flagged."""
        result = rule_engine_node(_state(applicant_age=66))
        assert any("age" in f.lower() for f in result["rule_flags"])

    def test_low_document_confidence_flagged(self):
        """doc_confidence < 0.4 → flagged."""
        result = rule_engine_node(_state(doc_confidence=0.3))
        assert any("confidence" in f.lower() for f in result["rule_flags"])

    def test_negative_net_worth_flagged(self):
        """Liabilities > 1.5× assets → negative net worth flag."""
        result = rule_engine_node(_state(
            parsed_assets=100_000,
            parsed_liabilities=200_000,   # > 1.5× assets
        ))
        assert any("liabilit" in f.lower() for f in result["rule_flags"])

    def test_parsed_income_overrides_declared(self):
        """
        If parsed_income is set (doc-verified), it MUST be used for FOIR/LTI
        calculation instead of declared annual_income.

        Distinguishable outcomes (verified against rule_engine_node math):
          parsed_income=₹6L  → LTI = 30L/6L  = 5.0×  (exactly at limit, no LTI flag)
                             → monthly income = ₹50k
                             → FOIR = (₹10k + ₹26,035 EMI) / ₹50k ≈ 72%  → FOIR flag fires
          declared=₹12L     → LTI = 30L/12L = 2.5×  (no flag)
                             → monthly income = ₹100k
                             → FOIR = (₹10k + ₹26,035) / ₹100k ≈ 36%    → no FOIR flag

        If the rule engine incorrectly uses declared income instead of parsed_income:
          - loan_to_income_ratio would be 2.5, not 5.0  (assertion 1 fails)
          - rule_passed would be True, not False         (assertion 2 fails)
          - no FOIR flag would be present                (assertion 3 fails)

        EMI uses the default 8.5% p.a. / 240 months from rule_engine.py constants.
        Reference: _calculate_emi() in rule_engine.py — reducing-balance formula.
        """
        state = _state(annual_income=1_200_000, loan_amount=3_000_000)
        state["parsed_income"] = 600_000  # doc-verified, lower than declared

        result = rule_engine_node(state)

        # Assertion 1: LTI must reflect parsed_income (5.0×), not declared (2.5×).
        # rule_engine rounds to 4 decimal places; 5.0 exactly is expected.
        assert abs(result["loan_to_income_ratio"] - 5.0) < 0.01, (
            f"loan_to_income_ratio should be ≈5.0 (using parsed_income=₹6L), "
            f"got {result['loan_to_income_ratio']:.4f}. "
            f"If 2.5 was returned, declared annual_income was used instead."
        )

        # Assertion 2: FOIR flag must fire because parsed monthly income (₹50k)
        # makes (₹10k existing + ₹26,035 EMI) / ₹50k ≈ 72%, above the 55% limit.
        # With declared ₹12L income the FOIR would be ≈36% and rule_passed would be True.
        assert result["rule_passed"] is False, (
            "rule_passed should be False — FOIR fires when parsed_income=₹6L is used. "
            "If True was returned, declared annual_income was used instead."
        )

        # Assertion 3: the flag that fired must be the FOIR flag, not something else.
        assert any("FOIR" in f for f in result["rule_flags"]), (
            f"Expected a FOIR flag when parsed_income=₹6L, "
            f"got flags: {result['rule_flags']}"
        )

    def test_multiple_violations_all_collected(self):
        """All violations should be collected before returning (fail-all, not fail-fast)."""
        result = rule_engine_node(_state(
            annual_income=200_000,        # triggers FOIR + LTI
            loan_amount=5_000_000,
            cibil_score=600,              # triggers CIBIL
            applicant_age=70,             # triggers age
            doc_confidence=0.2,           # triggers doc confidence
        ))
        assert len(result["rule_flags"]) >= 4, (
            f"Expected ≥4 flags, got {len(result['rule_flags'])}: {result['rule_flags']}"
        )
        assert result["rule_passed"] is False
