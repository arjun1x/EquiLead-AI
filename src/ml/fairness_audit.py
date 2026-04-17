"""
Phase 6: Fairness audit report.

Analyzes model predictions for disparate impact across categorical groups
and generates a comprehensive fairness report.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.schema import MODEL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.ml.preprocessing import load_raw, preprocess, split
from src.ml.train import train


def run_fairness_audit():
    print("=" * 60)
    print("EquiLend AI -- Fairness Audit Report")
    print("=" * 60)

    print("\n[1/4] Training model...")
    results = train(save=False)
    model = results["model"]
    X_test = results["X_test"]
    y_test = results["y_test"]
    metrics = results["metrics"]
    label_encoders = results["label_encoders"]

    raw_df = load_raw()
    processed_df, _ = preprocess(raw_df)
    _, X_test_full, _, y_test_full = split(processed_df)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n[2/4] Overall Model Performance")
    print("-" * 40)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted Repaid  Predicted Default")
    print(f"  Actual Repaid      {cm[0][0]:>6}            {cm[0][1]:>6}")
    print(f"  Actual Default     {cm[1][0]:>6}            {cm[1][1]:>6}")

    fpr = cm[0][1] / (cm[0][0] + cm[0][1])
    fnr = cm[1][0] / (cm[1][0] + cm[1][1])
    print(f"\n  False Positive Rate (good loans denied): {fpr:.2%}")
    print(f"  False Negative Rate (bad loans approved): {fnr:.2%}")

    print("\n[3/4] Disparate Impact Analysis by Group")
    print("-" * 40)

    raw_test = raw_df.iloc[X_test.index]

    for col in CATEGORICAL_FEATURES:
        print(f"\n  === {col} ===")
        groups = raw_test[col].fillna("Missing")
        denial_rates = {}

        for group_name in sorted(groups.unique()):
            mask = groups == group_name
            group_pred = y_pred[mask.values]
            group_actual = y_test.values[mask.values]
            group_proba = y_proba[mask.values]
            n = len(group_pred)

            if n < 5:
                continue

            denial_rate = group_pred.mean()
            actual_default_rate = group_actual.mean()
            avg_risk_score = group_proba.mean()

            denial_rates[group_name] = denial_rate

            print(f"    {group_name:>12}: n={n:>4}, "
                  f"denial_rate={denial_rate:.2%}, "
                  f"actual_default={actual_default_rate:.2%}, "
                  f"avg_risk={avg_risk_score:.3f}")

        if len(denial_rates) >= 2:
            min_rate = min(denial_rates.values())
            max_rate = max(denial_rates.values())
            if max_rate > 0:
                di_ratio = min_rate / max_rate
                min_group = [k for k, v in denial_rates.items() if v == min_rate][0]
                max_group = [k for k, v in denial_rates.items() if v == max_rate][0]
                status = "PASS" if di_ratio >= 0.8 else "REVIEW"
                print(f"    Disparate Impact Ratio: {di_ratio:.3f} "
                      f"({min_group} vs {max_group}) [{status}]")
                if di_ratio < 0.8:
                    print(f"    ** WARNING: Below 4/5ths rule threshold (0.80) **")

    print("\n[4/4] Feature Importance Ranking")
    print("-" * 40)

    importances = model.feature_importances_
    feat_imp = sorted(zip(MODEL_FEATURES, importances), key=lambda x: x[1], reverse=True)

    for rank, (feat, imp) in enumerate(feat_imp, 1):
        bar = "#" * int(imp * 50)
        print(f"  {rank:>2}. {feat:>10}: {imp:.4f} {bar}")

    print("\n" + "=" * 60)
    print("FAIRNESS AUDIT SUMMARY")
    print("=" * 60)
    print(f"  Model: XGBoost ({len(MODEL_FEATURES)} features)")
    print(f"  Dataset: HMEQ ({len(raw_df)} records, {y_test.mean():.1%} default rate)")
    print(f"  Protected features used in model: NONE")
    print(f"  Categorical groups analyzed: {', '.join(CATEGORICAL_FEATURES)}")
    print(f"  Overall AUC: {metrics['roc_auc']:.4f}")
    print(f"\n  NOTE: This dataset contains no protected class features")
    print(f"  (race, gender, age). Fairness audit focuses on:")
    print(f"    1. Disparate impact across JOB and REASON groups")
    print(f"    2. Email language bias (handled by Phase 4 agent)")
    print(f"    3. Feature importance to verify no proxy discrimination")
    print("=" * 60)


if __name__ == "__main__":
    run_fairness_audit()
