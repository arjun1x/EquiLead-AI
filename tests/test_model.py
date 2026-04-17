"""Tests for XGBoost model training and SHAP explanations."""

import pytest
import numpy as np

from src.ml.train import train, explain_single
from src.data.schema import MODEL_FEATURES


@pytest.fixture(scope="module")
def trained():
    return train(save=False)


def test_model_trains(trained):
    assert trained["model"] is not None
    assert trained["explainer"] is not None


def test_metrics_exist(trained):
    metrics = trained["metrics"]
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics
        assert 0 <= metrics[key] <= 1


def test_accuracy_above_threshold(trained):
    assert trained["metrics"]["accuracy"] >= 0.80


def test_auc_above_threshold(trained):
    assert trained["metrics"]["roc_auc"] >= 0.85


def test_confusion_matrix_shape(trained):
    cm = trained["metrics"]["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2


def test_shap_values_shape(trained):
    shap_values = trained["shap_values"]
    X_test = trained["X_test"]
    assert shap_values.shape == X_test.shape


def test_explain_single_returns_top5(trained):
    row = trained["X_test"].iloc[0].to_dict()
    for col in MODEL_FEATURES:
        if isinstance(row[col], (int, float, np.integer, np.floating)):
            row[col] = float(row[col])

    reasons = explain_single(
        trained["model"], trained["explainer"],
        row, trained["label_encoders"],
    )
    assert len(reasons) == 5
    for r in reasons:
        assert "feature" in r
        assert "shap_value" in r
        assert "value" in r
        assert r["feature"] in MODEL_FEATURES


def test_explain_single_sorted_by_importance(trained):
    row = trained["X_test"].iloc[0].to_dict()
    for col in MODEL_FEATURES:
        if isinstance(row[col], (int, float, np.integer, np.floating)):
            row[col] = float(row[col])

    reasons = explain_single(
        trained["model"], trained["explainer"],
        row, trained["label_encoders"],
    )
    shap_abs = [abs(r["shap_value"]) for r in reasons]
    assert shap_abs == sorted(shap_abs, reverse=True)


def test_explain_single_handles_nulls(trained):
    row = {f: None for f in MODEL_FEATURES}
    row["LOAN"] = 10000
    row["REASON"] = "HomeImp"
    row["JOB"] = "Other"

    reasons = explain_single(
        trained["model"], trained["explainer"],
        row, trained["label_encoders"],
    )
    assert len(reasons) == 5
