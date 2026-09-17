"""Controlled adverse-action reason codes.

Three things are kept apart on purpose:
  1. the internal model explanation (SHAP / coefficient contributions),
  2. the underwriting-policy result (rules that fired),
  3. the customer-facing reasons below.

A reason only reaches a customer when it is on this reviewed table, is approved for customer use, and the
observed value really sits on the adverse side of a documented comparison. The largest SHAP value is never
promoted to a legal reason by itself.
"""
from __future__ import annotations

from calculations import FEATURE_LABELS, format_value
from policy import POLICY_VERSION

REASON_CODE_VERSION = "reason-codes-v2"
MAX_PRINCIPAL_REASONS = 4

# code -> customer explanation, internal explanation, source feature, approved for customer-facing use
REASON_CODES = {
    "EQ01": {"customer": "The combined amount of your existing mortgage and the requested credit is too high relative to the value of the property.",
             "internal": "Combined loan-to-value after the proposed loan exceeds the product maximum or comfort level.",
             "feature": "cltv_after", "approved": True},
    "EQ02": {"customer": "Your monthly debt obligations, including the proposed payment, are too high relative to your income.",
             "internal": "Debt-to-income after the proposed payment exceeds the policy or comfort level.",
             "feature": "dti_after", "approved": True},
    "EQ03": {"customer": "Your credit score does not meet the minimum for this program.",
             "internal": "Credit score below the policy floor or below the model comfort level.",
             "feature": "credit_score", "approved": True},
    "EQ04": {"customer": "Your credit report shows recent late payments or delinquent accounts.",
             "internal": "Delinquencies in the last 24 months.", "feature": "delinquencies_24m", "approved": True},
    "EQ05": {"customer": "There are too many recent inquiries on your credit report.",
             "internal": "Credit inquiries in the last six months.", "feature": "inquiries_6m", "approved": True},
    "EQ06": {"customer": "Your credit history is not long enough.",
             "internal": "Length of credit history below comfort level.", "feature": "credit_history_years", "approved": True},
    "EQ07": {"customer": "Your length of employment is too short, or your employment could not be verified.",
             "internal": "Employment length or status.", "feature": "employment_years", "approved": True},
    "EQ08": {"customer": "You do not have sufficient cash reserves after closing.",
             "internal": "Reserves in months of obligations below policy or comfort level.", "feature": "reserves_months", "approved": True},
    "EQ09": {"customer": "We were unable to verify your income.",
             "internal": "Ratios were calculated on stated income; no verification received.", "feature": "income_basis", "approved": True},
    "EQ10": {"customer": "We did not receive all of the documents needed to complete your application.",
             "internal": "Required documents outstanding.", "feature": "doc_completeness", "approved": True},
    "EQ11": {"customer": "The value of the property is less than the balance of your existing mortgage.",
             "internal": "Negative gross equity.", "feature": "gross_equity", "approved": True},
    "EQ12": {"customer": "The property is not owner-occupied and the requested amount exceeds the limit for that property type.",
             "internal": "Investment-property CLTV above the review level.", "feature": "occupancy_type", "approved": True},
}

# Model-driver eligibility: a risk-raising driver on one of these features becomes a reason only when the
# observed value is on the adverse side of the comparison. Comparisons are documented comfort levels.
DRIVER_COMPARISONS = {
    "cltv_after": {"code": "EQ01", "adverse_when": "above", "value": 0.70, "text": "combined LTV above 70%"},
    "dti_after": {"code": "EQ02", "adverse_when": "above", "value": 0.36, "text": "debt-to-income above 36%"},
    "credit_score": {"code": "EQ03", "adverse_when": "below", "value": 700, "text": "credit score below 700"},
    "delinquencies_24m": {"code": "EQ04", "adverse_when": "above", "value": 0, "text": "one or more delinquencies"},
    "inquiries_6m": {"code": "EQ05", "adverse_when": "above", "value": 2, "text": "more than two inquiries in six months"},
    "credit_history_years": {"code": "EQ06", "adverse_when": "below", "value": 3, "text": "credit history shorter than three years"},
    "employment_years": {"code": "EQ07", "adverse_when": "below", "value": 2, "text": "employment shorter than two years"},
    "reserves_months": {"code": "EQ08", "adverse_when": "below", "value": 3, "text": "reserves below three months"},
}
# Features that can never become a customer-facing reason on their own.
NEVER_CUSTOMER_FACING = {"loan_purpose", "property_type", "term_years", "interest_rate", "product_type",
                         "monthly_income", "requested_amount", "property_value", "mortgage_balance", "ltv_current",
                         "monthly_debt", "occupancy_type", "doc_completeness"}


def _entry(code, feature, observed, comparison, basis, source, model_version):
    spec = REASON_CODES[code]
    return {"code": code, "customer_explanation": spec["customer"], "internal_explanation": spec["internal"],
            "source_feature": feature, "source": source, "observed_value": format_value(feature, observed) if observed is not None else "not available",
            "observed_raw": observed, "comparison": comparison, "basis": basis, "version": model_version,
            "approved_for_customer_use": bool(spec["approved"])}


def assign_reason_codes(policy: dict, drivers: list[dict], recommendation: str, feature_values: dict, model_version: str) -> dict:
    """Build the reason-code set for a recommendation.

    policy: output of policy.evaluate_policy; drivers: risk-raising drivers [{feature, contribution, observed}]
    ordered by magnitude, from the model consensus; feature_values: current feature vector.
    Returns {"principal": [...], "excluded": [...], "version": ...}."""
    if recommendation == "approve":
        return {"version": REASON_CODE_VERSION, "principal": [], "excluded": [], "note": "No adverse action: the recommendation is an approval."}
    principal, seen, excluded = [], set(), []

    # 1. Policy rules are definitive and go first.
    for rule in policy.get("rules", []):
        code = rule.get("reason_code")
        if not code or code in seen or code not in REASON_CODES:
            continue
        seen.add(code)
        threshold = rule["threshold"]
        comparison = f"{rule['message']}" if isinstance(threshold, str) else f"observed {format_value(rule['feature'], rule['observed'])} vs limit {format_value(rule['feature'], threshold)}"
        principal.append(_entry(code, rule["feature"], rule["observed"], comparison, "policy", f"{POLICY_VERSION}:{rule['rule_id']}", POLICY_VERSION))

    # 2. Model drivers, only when mapped and genuinely adverse.
    for driver in drivers:
        feature = driver["feature"]
        if feature in NEVER_CUSTOMER_FACING or feature not in DRIVER_COMPARISONS:
            excluded.append({"feature": feature, "label": FEATURE_LABELS.get(feature, feature),
                             "reason": "not on the reviewed reason table" if feature not in DRIVER_COMPARISONS else "never customer-facing",
                             "contribution": driver.get("contribution")})
            continue
        spec = DRIVER_COMPARISONS[feature]
        observed = feature_values.get(feature)
        if observed is None:
            excluded.append({"feature": feature, "label": FEATURE_LABELS.get(feature, feature), "reason": "no observed value", "contribution": driver.get("contribution")})
            continue
        adverse = observed > spec["value"] if spec["adverse_when"] == "above" else observed < spec["value"]
        if not adverse:
            excluded.append({"feature": feature, "label": FEATURE_LABELS.get(feature, feature),
                             "reason": f"observed {format_value(feature, observed)} is not {spec['text']}", "contribution": driver.get("contribution")})
            continue
        if spec["code"] in seen:
            continue
        seen.add(spec["code"])
        principal.append(_entry(spec["code"], feature, observed, f"observed {format_value(feature, observed)}; {spec['text']}", "model", f"driver:{feature}", model_version))
        if len(principal) >= MAX_PRINCIPAL_REASONS:
            break
    return {"version": REASON_CODE_VERSION, "principal": principal[:MAX_PRINCIPAL_REASONS], "excluded": excluded,
            "note": "Principal reasons, most significant first. Policy rules take precedence over model drivers."}


def customer_statements(assignment: dict) -> list[str]:
    return [r["customer_explanation"] for r in assignment.get("principal", []) if r.get("approved_for_customer_use")]
