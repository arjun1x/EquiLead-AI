"""Model outputs, risk bands, consensus and fallbacks. Run: python -m pytest -q -p no:cacheprovider test_scoring.py"""
import pytest

import explain
import scoring
from conftest import HIGH_RISK_APPLICATION, application
from policy import BANDS, DECISION_THRESHOLD, RECOMMENDATION_ORDER, band_for, evaluate_policy, most_conservative
from calculations import derive_metrics


@pytest.fixture(scope="module", autouse=True)
def models():
    status = scoring.load_models(auto_train=True)
    assert not status["errors"], status["errors"]
    assert set(status["loaded"]) == set(scoring.MODEL_KEYS)
    return status


def result(key, probability, recommendation=None, error=None):
    if probability is None:
        return scoring.ModelResult(model_key=key, model_name=key, model_version="test", error=error or "unavailable")
    band = band_for(probability)
    return scoring.ModelResult(model_key=key, model_name=key, model_version="test", probability=probability, raw_probability=probability,
                               risk_band=band["band"], recommendation=recommendation or band["recommendation"],
                               recommendation_label=recommendation or band["recommendation"], error=error)


def test_three_models_return_the_full_output_schema():
    response = scoring.score_application(application())
    assert [m.model_key for m in response.models] == ["scorecard", "forest", "gbm"]
    for m in response.models:
        assert m.error is None
        assert m.model_name and m.model_version and len(m.model_hash) == 64
        assert 0 < m.probability < 1 and 0 <= m.raw_probability <= 1
        assert m.risk_band in "ABCDE" and m.recommendation in RECOMMENDATION_ORDER
        assert m.decision_threshold == DECISION_THRESHOLD and set(m.band_cutoffs) == {"approve", "conditional_approval", "manual_review"}
        assert isinstance(m.in_range, bool) and isinstance(m.out_of_range, list)
        assert m.processing_ms > 0 and m.scoring_ms >= 0
        assert m.explainer_method
        assert all(d.contribution > 0 for d in m.drivers_positive) and all(d.contribution < 0 for d in m.drivers_negative)
    assert response.consensus.final_recommendation in RECOMMENDATION_ORDER
    assert len(response.scoring_hash) == 64 and response.total_ms > 0
    assert response.derived["calculation_version"] == "calc-v1"


def test_explainers_are_model_specific():
    methods = {m.model_key: m.explainer_method for m in scoring.score_application(application()).models}
    assert "coefficient" in methods["scorecard"]
    assert "SHAP" in methods["forest"] and "SHAP" in methods["gbm"]


def test_scoring_is_deterministic():
    a = scoring.score_application(application(), with_counterfactuals=False)
    b = scoring.score_application(application(), with_counterfactuals=False)
    assert [m.probability for m in a.models] == [m.probability for m in b.models]
    assert a.scoring_hash == b.scoring_hash


def test_risk_bands_are_contiguous_and_map_to_recommendations():
    assert band_for(0.0)["band"] == "A" and band_for(0.049)["band"] == "A"
    assert band_for(0.05)["band"] == "B" and band_for(0.10)["band"] == "C"
    assert band_for(0.20)["band"] == "D" and band_for(0.35)["band"] == "E" and band_for(0.99)["band"] == "E"
    assert band_for(0.02)["recommendation"] == "approve" and band_for(0.15)["recommendation"] == "conditional_approval"
    assert band_for(0.30)["recommendation"] == "manual_review" and band_for(0.50)["recommendation"] == "decline"
    assert [b["band"] for b in BANDS] == ["A", "B", "C", "D", "E"]


def test_high_risk_case_is_not_approved():
    response = scoring.score_application(application(**HIGH_RISK_APPLICATION))
    assert response.consensus.final_recommendation in ("manual_review", "decline")
    assert response.policy["result"] == "decline"
    assert response.reason_codes["principal"]


def test_consensus_unanimous():
    c = scoring.consensus([result("scorecard", 0.02), result("forest", 0.03), result("gbm", 0.04)], "approve")
    assert c.agreement == "unanimous" and c.final_recommendation == "approve" and not c.routed_to_manual_review
    assert c.median_probability == pytest.approx(0.03) and c.spread == pytest.approx(0.02)


def test_consensus_minor_disagreement_takes_the_conservative_side():
    c = scoring.consensus([result("scorecard", 0.04), result("forest", 0.06), result("gbm", 0.12)], "approve")
    assert c.agreement == "minor" and c.model_recommendation == "conditional_approval" and c.final_recommendation == "conditional_approval"


def test_material_disagreement_routes_to_manual_review():
    c = scoring.consensus([result("scorecard", 0.03), result("forest", 0.45), result("gbm", 0.05)], "approve")
    assert c.agreement == "material" and c.final_recommendation == "manual_review" and c.routed_to_manual_review
    assert any("Material disagreement" in line for line in c.explanation)


def test_policy_result_can_only_make_the_outcome_more_conservative():
    c = scoring.consensus([result("scorecard", 0.02), result("forest", 0.03), result("gbm", 0.02)], "decline")
    assert c.model_recommendation == "approve" and c.final_recommendation == "decline"
    assert any("takes precedence" in line for line in c.explanation)
    c2 = scoring.consensus([result("scorecard", 0.5), result("forest", 0.6), result("gbm", 0.55)], "approve")
    assert c2.final_recommendation == "decline"
    assert most_conservative("approve", "manual_review", "conditional_approval") == "manual_review"


def test_single_model_is_insufficient_for_an_automated_recommendation():
    c = scoring.consensus([result("scorecard", 0.02), result("forest", None, error="missing"), result("gbm", None, error="missing")], "approve")
    assert c.agreement == "insufficient" and c.final_recommendation == "manual_review" and c.models_failed == ["forest", "gbm"]


def test_missing_model_artifact_falls_back_gracefully(monkeypatch):
    state = dict(scoring._STATE)
    models_ = dict(state["models"]); models_.pop("gbm")
    monkeypatch.setitem(scoring._STATE, "models", models_)
    monkeypatch.setitem(scoring._STATE, "errors", {**state["errors"], "gbm": "artifact missing (test)"})
    response = scoring.score_application(application(), with_counterfactuals=False)
    gbm = next(m for m in response.models if m.model_key == "gbm")
    assert gbm.error and gbm.probability is None
    assert response.consensus.models_failed == ["gbm"] and set(response.consensus.models_used) == {"scorecard", "forest"}
    assert response.consensus.final_recommendation in RECOMMENDATION_ORDER
    assert "gbm" in scoring.status()["errors"]


def test_out_of_range_inputs_are_flagged_not_rejected():
    response = scoring.score_application(application(annual_income=6_000_000, verified_monthly_income=480000), with_counterfactuals=False)
    assert any(not m.in_range and "monthly_income" in m.out_of_range for m in response.models)


def test_shap_unavailable_falls_back_to_perturbation(monkeypatch):
    monkeypatch.setattr(explain, "SHAP_AVAILABLE", False)
    from calculations import feature_row
    r = scoring.score_model("forest", feature_row(application()))
    assert "perturbation" in r.explainer_method and r.error is None and (r.drivers_positive or r.drivers_negative)


def test_policy_rules_are_explicit_and_versioned():
    app = application(**HIGH_RISK_APPLICATION)
    policy = evaluate_policy(app, derive_metrics(app))
    assert policy["version"] == "policy-v1" and policy["result"] == "decline"
    ids = {r["rule_id"] for r in policy["rules"]}
    assert {"R2", "R4", "R5"} & ids
    for rule in policy["rules"]:
        assert {"rule_id", "outcome", "observed", "threshold", "message", "reason_code", "feature"} <= set(rule)


def test_counterfactuals_are_labelled_estimates_and_only_move_one_lever():
    response = scoring.score_application(application(credit_score=690, monthly_debt=3600, cash_reserves=3000, documents=["identity", "income_verification"]))
    if response.consensus.final_recommendation == "approve":
        assert response.counterfactuals == []
    for cf in response.counterfactuals:
        assert {"lever", "field", "from", "to", "estimated_probability", "estimated_band", "current_band"} <= set(cf)
        assert cf["from"] != cf["to"] and 0 <= cf["estimated_probability"] <= 1
