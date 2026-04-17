"""Tests for email generator (unit tests, no API calls)."""

from src.llm.email_generator import _translate_shap_reasons, SHAP_TO_ENGLISH


def test_translate_shap_positive():
    reasons = [{"feature": "DEBTINC", "shap_value": 1.5, "value": 45.0}]
    result = _translate_shap_reasons(reasons)
    assert len(result) == 1
    assert result[0]["explanation"] == "debt-to-income ratio"
    assert result[0]["direction"] == "increases default risk"


def test_translate_shap_negative():
    reasons = [{"feature": "VALUE", "shap_value": -0.8, "value": 150000}]
    result = _translate_shap_reasons(reasons)
    assert result[0]["direction"] == "decreases default risk"


def test_translate_all_features_mapped():
    for feature in SHAP_TO_ENGLISH:
        reasons = [{"feature": feature, "shap_value": 0.1, "value": 1.0}]
        result = _translate_shap_reasons(reasons)
        assert result[0]["explanation"] == SHAP_TO_ENGLISH[feature]


def test_translate_unknown_feature():
    reasons = [{"feature": "UNKNOWN_COL", "shap_value": 0.5, "value": 1.0}]
    result = _translate_shap_reasons(reasons)
    assert result[0]["explanation"] == "UNKNOWN_COL"


def test_translate_preserves_shap_value():
    reasons = [{"feature": "LOAN", "shap_value": 2.345, "value": 25000}]
    result = _translate_shap_reasons(reasons)
    assert result[0]["shap_value"] == 2.345
