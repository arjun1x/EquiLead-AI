"""Adverse-action reason codes.

Raw SHAP output is an engineering explanation, not a customer-facing reason. Before real lending use, each
principal reason on a notice must come from a code table that legal and compliance have approved
(ECOA / Regulation B, 12 CFR 1002.9). This module maps model features to that table and refuses to
promote anything that is not on it.

The statements below are working drafts. `APPROVED_BY` stays None until compliance signs off, and the
UI labels every code as pending until then.
"""

REASON_CODE_VERSION = "aa-codes-2026.09"
APPROVED_BY = None          # e.g. {"name": "...", "role": "Compliance", "date": "2026-10-01"}
MAX_PRINCIPAL_REASONS = 4   # Regulation B guidance: no more than four principal reasons

# feature -> (code, statement, direction that counts as adverse)
# direction "+" means a positive SHAP value (raises the default score) is the adverse case.
CODE_TABLE = {
    "DEBTINC": ("AA-01", "Excessive obligations in relation to income", "+"),
    "DELINQ":  ("AA-02", "Delinquent past or present credit obligations with others", "+"),
    "DEROG":   ("AA-03", "Serious delinquency, derogatory public record, or collection", "+"),
    "CLAGE":   ("AA-04", "Length of time accounts have been established", "+"),
    "NINQ":    ("AA-05", "Number of recent inquiries on credit report", "+"),
    "CLNO":    ("AA-06", "Number of established credit accounts", "+"),
    "YOJ":     ("AA-07", "Length of employment", "+"),
    "LOAN":    ("AA-08", "Value or type of collateral not sufficient for amount requested", "+"),
    "VALUE":   ("AA-08", "Value or type of collateral not sufficient for amount requested", "+"),
    "MORTDUE": ("AA-08", "Value or type of collateral not sufficient for amount requested", "+"),
}

# Features that must never appear on a notice without a compliance decision.
# JOB is an occupation category (a potential proxy); REASON is the loan purpose.
NOT_NOTICE_ELIGIBLE = {"JOB": "occupation category requires a compliance decision before use as a reason",
                       "REASON": "loan purpose is not an approved adverse-action reason"}


def adverse_action_reasons(shap_reasons, decision, max_reasons=MAX_PRINCIPAL_REASONS):
    """Return the principal reasons for a decline, in order of model impact.

    Only features that raised the default score qualify. Features outside the code table are reported
    with review_required=True and no code, so a reviewer can see what the model leaned on without that
    reason ever reaching a customer."""
    if decision != "denied":
        return []
    ordered = sorted((r for r in shap_reasons if float(r.get("shap_value", 0)) > 0),
                     key=lambda r: float(r["shap_value"]), reverse=True)
    reasons, seen = [], set()
    for row in ordered:
        feature = row["feature"]
        if feature in CODE_TABLE:
            code, statement, _ = CODE_TABLE[feature]
            if code in seen:
                continue
            seen.add(code)
            reasons.append({"code": code, "statement": statement, "feature": feature,
                            "weight": round(float(row["shap_value"]), 4), "review_required": False})
        else:
            reasons.append({"code": None, "statement": NOT_NOTICE_ELIGIBLE.get(feature, "no approved reason code"),
                            "feature": feature, "weight": round(float(row["shap_value"]), 4), "review_required": True})
        if len([r for r in reasons if r["code"]]) >= max_reasons:
            break
    return reasons


def notice_statements(reasons):
    """Statements safe to place in a customer notice: coded reasons only, never review_required rows."""
    return [r["statement"] for r in reasons if r.get("code")]


def approval_status(card=None):
    approved = bool(card.get("reason_codes_approved")) if card else APPROVED_BY is not None
    approved_on = (card or {}).get("reason_codes_approved_on") or (APPROVED_BY or {}).get("date")
    return {"version": REASON_CODE_VERSION, "approved": approved, "approved_on": approved_on}
