"""Financial calculations and input validation. Run: python -m pytest -q -p no:cacheprovider test_calculations.py"""
import pytest
from pydantic import ValidationError

from calculations import (DOCUMENT_ITEMS, MODEL_FEATURES, ApplicationInput, amortized_payment, derive_metrics, feature_row, format_value,
                          interest_only_payment, safe_ratio)
from conftest import BASE_APPLICATION, application


def flags(derived):
    return {f.code: f.severity for f in derived.flags}


def test_amortized_payment_matches_standard_formula():
    assert amortized_payment(100000, 6.0, 30) == pytest.approx(599.55, abs=0.01)
    assert amortized_payment(90000, 8.5, 20) == pytest.approx(781.05, abs=0.05)


def test_amortized_payment_edge_cases():
    assert amortized_payment(120000, 0.0, 10) == pytest.approx(1000.0)      # zero rate: straight line
    assert amortized_payment(0, 8.0, 20) == 0.0
    assert interest_only_payment(90000, 8.5) == pytest.approx(637.5)
    assert interest_only_payment(-5, 8.5) == 0.0


def test_safe_ratio_never_divides_by_zero():
    assert safe_ratio(1, 0) is None
    assert safe_ratio(None, 5) is None
    assert safe_ratio(1, float("inf")) is None
    assert safe_ratio(3, 4) == 0.75


def test_derived_metrics_for_a_typical_case():
    d = derive_metrics(application())
    assert d.income_basis == "verified" and d.monthly_income == 9500
    assert d.gross_equity == 340000
    assert d.max_cltv == 0.85
    assert d.available_equity == pytest.approx(650000 * 0.85 - 310000)
    assert d.ltv_current == pytest.approx(310000 / 650000)
    assert d.cltv_after == pytest.approx(400000 / 650000)
    assert d.payment_basis.startswith("interest-only") or "interest-only" in d.payment_basis
    assert d.dti_before == pytest.approx(2400 / 9500)
    assert d.dti_after == pytest.approx((2400 + d.estimated_payment) / 9500)
    assert d.reserves_months == pytest.approx(40000 / (2400 + d.estimated_payment))
    assert d.doc_completeness == 1.0 and d.missing_documents == []
    assert not d.requested_exceeds_equity
    assert "NO_INCOME" not in flags(d)


def test_home_equity_loan_uses_amortized_payment():
    d = derive_metrics(application(product_type="home_equity_loan"))
    assert d.estimated_payment == pytest.approx(amortized_payment(90000, 8.5, 20), abs=0.01)
    assert "amortiz" in d.payment_basis


def test_stated_income_is_flagged_and_used_when_no_verification():
    d = derive_metrics(application(verified_monthly_income=None))
    assert d.income_basis == "stated" and d.monthly_income == pytest.approx(10000)
    assert "INCOME_UNVERIFIED" in flags(d)


def test_zero_income_blocks_and_dti_is_undefined_not_infinite():
    d = derive_metrics(application(annual_income=0, verified_monthly_income=None, employment_status="unemployed"))
    assert d.dti_after is None and d.dti_before is None
    assert flags(d).get("NO_INCOME") == "blocking"
    assert d.reserves_months is not None


def test_underwater_property_and_requested_over_equity_block():
    d = derive_metrics(application(property_value=300000, mortgage_balance=320000, requested_amount=10000))
    assert d.available_equity == 0
    assert flags(d).get("UNDERWATER") == "blocking"
    assert d.requested_exceeds_equity
    d2 = derive_metrics(application(requested_amount=300000))
    assert d2.requested_exceeds_equity and flags(d2).get("EXCEEDS_EQUITY") == "blocking"


def test_document_completeness_and_missing_list():
    d = derive_metrics(application(documents=["identity", "insurance"]))
    assert d.doc_completeness == pytest.approx(2 / len(DOCUMENT_ITEMS))
    assert set(d.missing_documents) == set(DOCUMENT_ITEMS) - {"identity", "insurance"}
    assert "DOCS_INCOMPLETE" in flags(d)


def test_occupancy_caps_available_equity():
    primary = derive_metrics(application(occupancy_type="primary")).available_equity
    investment = derive_metrics(application(occupancy_type="investment")).available_equity
    assert investment < primary
    assert derive_metrics(application(occupancy_type="investment")).max_cltv == 0.75


def test_outlier_and_impossible_inputs_are_flagged_or_rejected():
    assert "INCOME_OUTLIER" in flags(derive_metrics(application(verified_monthly_income=250000)))
    assert "VALUE_OUTLIER" in flags(derive_metrics(application(property_value=40000000, mortgage_balance=1000000, requested_amount=100000)))
    assert "UNEMPLOYED_WITH_INCOME" in flags(derive_metrics(application(employment_status="unemployed")))


@pytest.mark.parametrize("field,value", [
    ("credit_score", 200), ("credit_score", 900), ("monthly_debt", -1), ("property_value", 0), ("requested_amount", 0),
    ("annual_income", -5), ("term_years", 2), ("interest_rate", 0), ("delinquencies_24m", -1), ("employment_status", "retired-ish"),
    ("documents", ["passport"]), ("mortgage_balance", 5_000_000), ("requested_amount", 2_000_000), ("applicant_name", "x" * 121),
])
def test_validation_rejects_bad_values(field, value):
    with pytest.raises(ValidationError):
        ApplicationInput(**{**BASE_APPLICATION, field: value})


def test_validation_accepts_strings_from_forms_and_blank_optionals():
    app = ApplicationInput(**{**{k: str(v) for k, v in BASE_APPLICATION.items() if k != "documents"}, "documents": BASE_APPLICATION["documents"], "verified_monthly_income": None})
    assert app.credit_score == 735 and app.verified_monthly_income == 0


def test_feature_row_matches_model_contract():
    row = feature_row(application())
    assert set(row) == set(MODEL_FEATURES)
    assert row["cltv_after"] == pytest.approx(400000 / 650000)
    assert row["product_type"] == "heloc"


def test_format_value_is_human_readable():
    assert format_value("cltv_after", 0.6154) == "61.5%"
    assert format_value("requested_amount", 90000) == "$90,000"
    assert format_value("interest_rate", 8.5) == "8.50%"
    assert format_value("credit_score", 735) == "735"
    assert format_value("credit_score", None) == "—"
