"""Controlled reason codes. Run: python -m pytest -q -p no:cacheprovider test_reason_codes.py"""
import pytest

import scoring
from conftest import HIGH_RISK_APPLICATION, application
from reason_codes import MAX_PRINCIPAL_REASONS, NEVER_CUSTOMER_FACING, REASON_CODE_VERSION, REASON_CODES, assign_reason_codes, customer_statements

REQUIRED_FIELDS = {"code", "customer_explanation", "internal_explanation", "source_feature", "source", "observed_value", "comparison",
                   "basis", "version", "approved_for_customer_use"}


@pytest.fixture(scope="module", autouse=True)
def models():
    scoring.load_models(auto_train=True)


def test_reason_code_table_is_complete_and_specific():
    assert REASON_CODE_VERSION.startswith("reason-codes-")
    for code, entry in REASON_CODES.items():
        assert code.startswith("EQ") and len(code) == 4
        assert entry["customer"].endswith(".") and len(entry["customer"]) > 25
        assert entry["internal"] and entry["feature"]
        assert "internal standards" not in entry["customer"].lower() and "criteria" not in entry["customer"].lower()


def test_approval_has_no_adverse_reasons():
    assigned = assign_reason_codes({"rules": []}, [], "approve", {}, {})
    assert assigned["principal"] == [] and customer_statements(assigned) == []


def test_high_risk_case_gets_policy_backed_codes_first():
    response = scoring.score_application(application(**HIGH_RISK_APPLICATION), with_counterfactuals=False)
    principal = response.reason_codes["principal"]
    assert 1 <= len(principal) <= MAX_PRINCIPAL_REASONS
    assert principal[0]["basis"] == "policy"
    for entry in principal:
        assert REQUIRED_FIELDS <= set(entry)
        assert entry["code"] in REASON_CODES
        assert entry["customer_explanation"] == REASON_CODES[entry["code"]]["customer"]
        assert entry["version"]           # version of the rule or model that produced the reason
        assert entry["observed_value"] not in ("", None)
    assert response.reason_codes["version"] == REASON_CODE_VERSION
    assert len({e["code"] for e in principal}) == len(principal)          # no duplicates


def test_customer_statements_only_use_approved_wording():
    response = scoring.score_application(application(**HIGH_RISK_APPLICATION), with_counterfactuals=False)
    statements = customer_statements(response.reason_codes)
    approved = {e["customer"] for e in REASON_CODES.values() if e["approved"]}
    assert statements and all(s in approved for s in statements)
    for s in statements:
        assert "SHAP" not in s and "model" not in s.lower()


def test_model_drivers_never_promote_forbidden_features():
    drivers = [{"feature": "loan_purpose", "label": "Loan purpose", "contribution": 2.0, "observed": "debt_consolidation"},
               {"feature": "interest_rate", "label": "Interest rate", "contribution": 1.5, "observed": 9.0},
               {"feature": "cltv_after", "label": "Combined LTV after loan", "contribution": 1.0, "observed": 0.84}]
    assigned = assign_reason_codes({"rules": []}, drivers, "decline", {"cltv_after": 0.84, "loan_purpose": "debt_consolidation", "interest_rate": 9.0}, {"scorecard": "v"})
    codes = {e["code"] for e in assigned["principal"]}
    assert "EQ01" in codes
    assert all(e["source_feature"] not in NEVER_CUSTOMER_FACING for e in assigned["principal"])
    excluded = {x["feature"] for x in assigned["excluded"]}
    assert "loan_purpose" in excluded and "interest_rate" in excluded


def test_model_driver_below_comparison_threshold_is_excluded_with_reason():
    drivers = [{"feature": "credit_score", "label": "Credit score", "contribution": 0.9, "observed": 780}]
    assigned = assign_reason_codes({"rules": []}, drivers, "manual_review", {"credit_score": 780}, {})
    assert assigned["principal"] == []
    assert assigned["excluded"] and "credit_score" == assigned["excluded"][0]["feature"] and assigned["excluded"][0]["reason"]
