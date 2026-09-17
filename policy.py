"""Underwriting policy: risk bands and hard rules.

The policy result is kept separate from the model outputs and from customer-facing reason codes.
Every threshold here is a demonstration assumption, versioned so it can be traced in the audit trail.
"""
from __future__ import annotations

from calculations import ApplicationInput, DerivedMetrics, MAX_CLTV

POLICY_VERSION = "policy-v1"
BANDS_VERSION = "bands-v1"

# Calibrated probability of default -> risk band -> recommendation. Upper bounds are exclusive.
BANDS = [
    {"band": "A", "upper": 0.05, "recommendation": "approve", "label": "Very low"},
    {"band": "B", "upper": 0.10, "recommendation": "approve", "label": "Low"},
    {"band": "C", "upper": 0.20, "recommendation": "conditional_approval", "label": "Moderate"},
    {"band": "D", "upper": 0.35, "recommendation": "manual_review", "label": "Elevated"},
    {"band": "E", "upper": 1.01, "recommendation": "decline", "label": "High"},
]
DECISION_THRESHOLD = 0.35          # probability at or above which the band recommendation is a decline
BAND_CUTOFFS = {"approve": 0.10, "conditional_approval": 0.20, "manual_review": 0.35}
RECOMMENDATION_ORDER = ["approve", "conditional_approval", "manual_review", "decline"]
RECOMMENDATION_LABELS = {"approve": "Approve", "conditional_approval": "Conditional approval",
                         "manual_review": "Manual review", "decline": "Decline recommendation"}


def band_for(probability: float) -> dict:
    for band in BANDS:
        if probability < band["upper"]:
            return band
    return BANDS[-1]


def severity(recommendation: str) -> int:
    return RECOMMENDATION_ORDER.index(recommendation)


def most_conservative(*recommendations: str) -> str:
    return max(recommendations, key=severity)


# Rule table: (id, name, feature/derived name, customer-reason code or None)
RULES_DOC = [
    ("R1", "Combined LTV within product maximum", "cltv_after", "EQ01"),
    ("R2", "Mortgage balance below property value", "gross_equity", "EQ11"),
    ("R3", "Debt-to-income after the loan", "dti_after", "EQ02"),
    ("R4", "Credit score program floor", "credit_score", "EQ03"),
    ("R5", "Recent delinquencies", "delinquencies_24m", "EQ04"),
    ("R6", "Income verification", "income_basis", "EQ09"),
    ("R7", "Required documents", "doc_completeness", "EQ10"),
    ("R8", "Cash reserves after closing", "reserves_months", "EQ08"),
    ("R9", "Employment", "employment_status", "EQ07"),
    ("R10", "Investment property exposure", "occupancy_type", "EQ12"),
    ("R11", "Recent credit inquiries", "inquiries_6m", "EQ05"),
]


def evaluate_policy(app: ApplicationInput, derived: DerivedMetrics) -> dict:
    """Return {"result", "rules": [fired rules], "version"}. Rules only fire when they change the outcome."""
    fired = []

    def fire(rule_id, outcome, observed, threshold, message, code, feature):
        fired.append({"rule_id": rule_id, "outcome": outcome, "observed": observed, "threshold": threshold,
                      "message": message, "reason_code": code, "feature": feature})

    cap = MAX_CLTV[app.occupancy_type]
    if derived.cltv_after > cap:
        fire("R1", "decline", derived.cltv_after, cap,
             f"Combined LTV after the loan is {derived.cltv_after:.1%}; the maximum for a {app.occupancy_type.replace('_', ' ')} is {cap:.0%}.",
             "EQ01", "cltv_after")
    if derived.gross_equity < 0:
        fire("R2", "decline", derived.gross_equity, 0, "The mortgage balance exceeds the property value.", "EQ11", "gross_equity")
    if derived.dti_after is None:
        fire("R3", "decline", None, 0.50, "Debt-to-income cannot be calculated without income.", "EQ02", "dti_after")
    elif derived.dti_after > 0.50:
        fire("R3", "decline", derived.dti_after, 0.50, f"Debt-to-income after the loan is {derived.dti_after:.1%}, above the 50% maximum.", "EQ02", "dti_after")
    elif derived.dti_after > 0.43:
        fire("R3", "manual_review", derived.dti_after, 0.43, f"Debt-to-income after the loan is {derived.dti_after:.1%}, above the 43% review level.", "EQ02", "dti_after")
    if app.credit_score < 620:
        fire("R4", "decline", app.credit_score, 620, f"Credit score {app.credit_score} is below the program floor of 620.", "EQ03", "credit_score")
    elif app.credit_score < 660:
        fire("R4", "manual_review", app.credit_score, 660, f"Credit score {app.credit_score} is below the 660 review level.", "EQ03", "credit_score")
    if app.delinquencies_24m >= 3:
        fire("R5", "decline", app.delinquencies_24m, 3, f"{app.delinquencies_24m} delinquencies in the last 24 months (maximum 2).", "EQ04", "delinquencies_24m")
    elif app.delinquencies_24m >= 1:
        fire("R5", "manual_review", app.delinquencies_24m, 1, f"{app.delinquencies_24m} delinquency in the last 24 months requires review.", "EQ04", "delinquencies_24m")
    if derived.income_basis == "stated":
        fire("R6", "conditional_approval", "stated", "verified", "Income must be verified before closing.", "EQ09", "income_basis")
    if derived.missing_documents:
        fire("R7", "conditional_approval", derived.doc_completeness, 1.0,
             f"{len(derived.missing_documents)} required document(s) outstanding.", "EQ10", "doc_completeness")
    if derived.reserves_months is not None and derived.reserves_months < 2:
        fire("R8", "conditional_approval", derived.reserves_months, 2,
             f"Reserves cover {derived.reserves_months:.1f} months of obligations; two months are required.", "EQ08", "reserves_months")
    if app.employment_status == "unemployed":
        fire("R9", "manual_review", "unemployed", "employed", "Applicant is not currently employed.", "EQ07", "employment_status")
    elif app.employment_status in ("employed", "self_employed") and app.employment_years < 2:
        fire("R9", "conditional_approval", app.employment_years, 2, "Less than two years in current employment; verify employment history.", "EQ07", "employment_years")
    if app.occupancy_type == "investment" and derived.cltv_after > 0.70:
        fire("R10", "manual_review", derived.cltv_after, 0.70, "Investment property above 70% CLTV requires review.", "EQ12", "occupancy_type")
    if app.inquiries_6m >= 6:
        fire("R11", "manual_review", app.inquiries_6m, 6, f"{app.inquiries_6m} credit inquiries in six months.", "EQ05", "inquiries_6m")

    result = most_conservative("approve", *[rule["outcome"] for rule in fired]) if fired else "approve"
    return {"version": POLICY_VERSION, "result": result, "rules": fired}
