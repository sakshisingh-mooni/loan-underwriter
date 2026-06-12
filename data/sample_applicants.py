"""
data/sample_applicants.py
--------------------------
Pre-built test cases that demonstrate all three decision paths:
  1. Rahul Sharma   → Approve (clean profile)
  2. Priya Patel    → Refer   (borderline — triggers HITL)
  3. Vikram Singh   → Reject  (multiple hard failures)
 
Use these in the Streamlit UI via the "Load Sample" buttons.
"""
 
SAMPLE_APPLICANTS = {
    "Rahul Sharma (Approve)": {
        "applicant_name": "Rahul Sharma",
        "applicant_age": 32,
        "annual_income": 1_200_000.0,       # ₹12 LPA
        "loan_amount_requested": 3_000_000.0, # ₹30L home loan
        "loan_purpose": "Home Purchase",
        "employment_type": "Salaried",
        "existing_obligations": 10_000.0,    # ₹10k/month existing EMI
        # cibil_score_override: bureau mock uses name-hash which gives Rahul CIBIL=429.
        # Override to 760 so the demo produces the documented Approve outcome.
        # FOIR=36%, LTI=2.5x, CIBIL=760 → 0 rule flags → Low risk → Approve.
        "cibil_score_override": 760,
        "document_text": """
Salary Certificate — Rahul Sharma
Employer: Infosys Technologies Ltd, Pune
Designation: Senior Software Engineer
Annual CTC: ₹12,00,000
Monthly Take-home: ₹88,000
Date of Joining: March 2019
 
Bank Statement Summary (Last 6 months):
Average Monthly Balance: ₹1,45,000
Salary Credits: ₹88,000/month (regular)
No cheque bounces.
 
Fixed Assets:
- Residential plot (Pune): ₹18,00,000
- Savings/FD: ₹4,50,000
 
Outstanding Liabilities:
- Car loan balance: ₹1,20,000 (remaining tenure 12 months, EMI ₹10,000)
        """,
    },
 
    "Priya Patel (Refer - HITL)": {
        "applicant_name": "Priya Patel",
        "applicant_age": 28,
        "annual_income": 600_000.0,          # ₹6 LPA
        "loan_amount_requested": 2_500_000.0, # ₹25L
        "loan_purpose": "Business Expansion",
        "employment_type": "Self-Employed",
        "existing_obligations": 18_000.0,    # ₹18k/month obligations
        # cibil_score_override: bureau mock gives Priya CIBIL=486 (2 flags → Reject).
        # Override to 700 so only FOIR fires (79% > 55% limit) → exactly 1 flag → Refer → HITL.
        # LTI=4.2x passes (< 5x limit). FOIR is the sole flag that triggers the HITL panel.
        "cibil_score_override": 700,
        "document_text": """
ITR Summary — Priya Patel (PAN: ABCPP1234P)
Assessment Year: 2023-24
Net Taxable Income: ₹5,80,000
Business Income: ₹6,20,000 (Gross)
Expenses Claimed: ₹40,000
 
GST Registration: Active (GSTIN 27ABCDE1234F1Z5)
Annual Turnover (FY23-24): ₹22,00,000
 
Assets Declared:
- Business equipment: ₹3,00,000
- Savings account balance: ₹75,000
 
Liabilities:
- Informal borrowing from family: ₹1,50,000
- Credit card outstanding: ₹45,000 (EMI ₹8,000)
- Personal loan: ₹1,00,000 (EMI ₹10,000)
 
Note: Business commenced January 2022 — limited credit history.
        """,
    },
 
    "Vikram Singh (Reject)": {
        "applicant_name": "Vikram Singh",
        "applicant_age": 24,
        "annual_income": 360_000.0,          # ₹3.6 LPA
        "loan_amount_requested": 5_000_000.0, # ₹50L — extremely high LTI
        "loan_purpose": "Personal Loan",
        "employment_type": "Self-Employed",
        "existing_obligations": 25_000.0,    # ₹25k/month — very high FOIR
        # cibil_score_override: bureau mock gives Vikram CIBIL=643 (just below 650 threshold).
        # Override to 580 (Poor) to make the rejection story cleaner:
        # FOIR=228%, LTI=13.9x, CIBIL=580 → 3 rule flags → High risk → Reject.
        "cibil_score_override": 580,
        "document_text": """
Income Declaration — Vikram Singh
Self-employed: Freelance photography
Estimated annual income: ₹3,60,000 (approximate)
No ITR filed for current year.
No formal employment records.
 
Bank Statement (last 3 months):
Average balance: ₹8,000
Irregular credits ranging ₹5,000–₹40,000
2 cheque bounces recorded.
 
Assets: None declared.
Liabilities:
- Personal loan (friend): ₹2,00,000
- Credit card (2 cards): ₹1,20,000 outstanding
- Monthly obligations: approx ₹25,000
 
Note: Applicant recently graduated (2022). Limited credit history.
Age: 24 years.
        """,
    },
}