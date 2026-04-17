"""
XGBoost training pipeline with SHAP explanations.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)

from src.ml.preprocessing import load_raw, preprocess, split

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def train(save: bool = True) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df, label_encoders = preprocess(df)
    X_train, X_test, y_train, y_test = split(df)

    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if save:
        model.save_model(str(MODELS_DIR / "xgb_model.json"))
        with open(MODELS_DIR / "label_encoders.pkl", "wb") as f:
            pickle.dump(label_encoders, f)
        with open(MODELS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Repaid", "Default"]))
    print(f"ROC-AUC: {metrics['roc_auc']}")

    return {
        "model": model,
        "explainer": explainer,
        "shap_values": shap_values,
        "X_test": X_test,
        "y_test": y_test,
        "metrics": metrics,
        "label_encoders": label_encoders,
    }


def explain_single(model, explainer, row: dict, label_encoders: dict) -> list[dict]:
    """Return top SHAP reasons for a single loan application."""
    import pandas as pd
    from src.data.schema import MODEL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

    input_df = pd.DataFrame([row])[MODEL_FEATURES]

    for col in CATEGORICAL_FEATURES:
        input_df[col] = input_df[col].fillna("Unknown")
        if col in label_encoders:
            le = label_encoders[col]
            input_df[col] = input_df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    for col in NUMERIC_FEATURES:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

    sv = explainer.shap_values(input_df)[0]
    feature_impacts = sorted(
        zip(MODEL_FEATURES, sv, input_df.iloc[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return [
        {"feature": name, "shap_value": round(float(val), 4), "value": float(fval)}
        for name, val, fval in feature_impacts[:5]
    ]


if __name__ == "__main__":
    results = train()
    print(f"\nModel saved to {MODELS_DIR}")
    print(f"Metrics: {results['metrics']}")
