"""
Demo: Full EquiLend AI pipeline (Phases 2-4).

XGBoost -> SHAP -> Claude email -> Bias detection agent
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data.schema import MODEL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.ml.preprocessing import load_raw
from src.ml.train import train, explain_single
from src.llm.email_generator import generate_email
from src.agents.bias_detector import run_bias_agent


def predict_single(model, applicant_data, label_encoders):
    input_df = pd.DataFrame([applicant_data])[MODEL_FEATURES]
    for col in CATEGORICAL_FEATURES:
        input_df[col] = input_df[col].fillna("Unknown")
        if col in label_encoders:
            le = label_encoders[col]
            input_df[col] = input_df[col].astype(str).apply(
                lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else 0
            )
    for col in NUMERIC_FEATURES:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

    proba = model.predict_proba(input_df)[0]
    return proba[1], max(proba) * 100


def run_demo():
    print("=" * 60)
    print("EquiLend AI -- Full Pipeline Demo (Phases 2-4)")
    print("XGBoost -> SHAP -> Claude Email -> Bias Agent")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    print("\n[1/4] Training model...")
    results = train(save=False)
    model = results["model"]
    explainer = results["explainer"]
    label_encoders = results["label_encoders"]

    raw_df = load_raw()

    samples = [
        {"index": 0, "name": "Sarah Mitchell"},
        {"index": 10, "name": "James Rodriguez"},
        {"index": 100, "name": "Emily Chen"},
    ]

    for sample in samples:
        idx = sample["index"]
        row = raw_df.iloc[idx]
        applicant_data = row.to_dict()
        actual = "DEFAULT" if row["BAD"] == 1 else "REPAID"

        print(f"\n{'=' * 60}")
        print(f"Applicant: {sample['name']} (row {idx}, actual: {actual})")
        loan_val = row.get('LOAN', 0)
        prop_val = row.get('VALUE', 0)
        job_val = row.get('JOB', 'N/A')
        print(f"  LOAN=${loan_val:,.0f}  VALUE=${prop_val:,.0f}  JOB={job_val}")
        print("-" * 60)

        print("\n[2/4] SHAP explanation...")
        shap_reasons = explain_single(model, explainer, applicant_data, label_encoders)
        print("Top reasons:")
        for r in shap_reasons:
            direction = "+ risk" if r["shap_value"] > 0 else "- risk"
            print(f"  {r['feature']:>10}: SHAP={r['shap_value']:+.4f} ({direction})")

        default_prob, confidence = predict_single(model, applicant_data, label_encoders)
        decision = "denied" if default_prob >= 0.5 else "approved"
        print(f"\nDecision: {decision.upper()} (default prob: {default_prob:.1%}, confidence: {confidence:.1f}%)")

        print("\n[3/4] Generating email via Claude...")
        email_result = generate_email(
            decision=decision,
            confidence=confidence,
            applicant_data=applicant_data,
            shap_reasons=shap_reasons,
            applicant_name=sample["name"],
        )
        draft_email = email_result["email"]
        print(f"  Draft generated ({email_result['input_tokens']}+{email_result['output_tokens']} tokens)")

        print("\n[4/4] Bias detection agent...")
        agent_result = run_bias_agent(draft_email, api_key=api_key)

        print(f"\n  Agent decision: {agent_result.decision.upper()}")
        print(f"  Bias scores: {' -> '.join(f'{s:.2f}' for s in agent_result.bias_scores)}")
        print(f"  Rewrites: {agent_result.rewrites}")

        print(f"\n--- FINAL EMAIL ---")
        print(agent_result.final_email)
        print(f"--- END ({agent_result.decision.upper()}) ---")


if __name__ == "__main__":
    run_demo()
