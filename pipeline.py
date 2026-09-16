"""Research-mode bridge for the EquiLead XGBoost + SHAP workflow.

Demo mode is the default because the repository does not include training data or model artifacts.
Research mode (DEMO_MODE=0) fails closed: every prerequisite is checked before any computation or API call.

Required beside app.py (or via env): xgb_model.json, label_encoders.pkl, model_card.json, ANTHROPIC_API_KEY.
Optional: calibrator.pkl (a fitted sklearn calibrator with predict_proba or predict) so the decision uses a
calibrated probability instead of the raw class-weighted score.
"""
import json
import os
import pickle
from pathlib import Path

from model_card import load_model_card, decision_metadata
from reason_codes import adverse_action_reasons, notice_statements, REASON_CODE_VERSION

MODEL_FEATURES = ["LOAN", "MORTDUE", "VALUE", "REASON", "JOB", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]
CATEGORICAL = ["REASON", "JOB"]
NUMERIC = [f for f in MODEL_FEATURES if f not in CATEGORICAL]
ROOT = Path(__file__).resolve().parent


def _path(env_name, default):
    path = Path(os.getenv(env_name, default))
    return path if path.is_absolute() else ROOT / path


def check_prerequisites():
    """Raise a clear RuntimeError for the first missing prerequisite. Nothing is computed or sent before this passes."""
    model_path = _path("MODEL_PATH", "xgb_model.json")
    encoders_path = _path("LABEL_ENCODERS_PATH", "label_encoders.pkl")
    missing = [p.name for p in (model_path, encoders_path) if not p.exists()]
    if missing:
        raise RuntimeError(f"Model artifacts are missing: {', '.join(missing)}. Add them, then rerun with DEMO_MODE=0.")
    card = load_model_card(demo=False)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required when DEMO_MODE=0.")
    try:
        import shap  # noqa: F401
        import xgboost  # noqa: F401
        import anthropic  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install the research-mode ML dependencies from requirements.txt before disabling demo mode.") from exc
    return model_path, encoders_path, card


def _prepare(data, encoders):
    import pandas as pd
    frame = pd.DataFrame([data])[MODEL_FEATURES].copy()
    for col in CATEGORICAL:
        frame[col] = frame[col].fillna("Unknown").astype(str)
        if col in encoders:
            encoder = encoders[col]
            frame[col] = frame[col].map(lambda value: encoder.transform([value])[0] if value in encoder.classes_ else 0)
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    return frame


def _reasons(explainer, frame):
    values = explainer.shap_values(frame)[0]
    rows = sorted(zip(MODEL_FEATURES, values, frame.iloc[0]), key=lambda row: abs(row[1]), reverse=True)[:5]
    return [{"feature": name, "shap_value": round(float(value), 4), "value": float(raw)} for name, value, raw in rows]


def _calibrate(raw_score):
    path = _path("CALIBRATOR_PATH", "calibrator.pkl")
    if not path.exists():
        return None
    with path.open("rb") as handle:
        calibrator = pickle.load(handle)
    if hasattr(calibrator, "predict_proba"):
        return float(calibrator.predict_proba([[raw_score]])[0][1])
    return float(calibrator.predict([raw_score])[0])


def _draft_email(decision, data, reasons, applicant_name):
    import anthropic
    statements = notice_statements(reasons)
    if decision == "denied" and not statements:
        raise RuntimeError("No approved adverse-action reason is available for this decline; the draft cannot be written.")
    reason_block = "\n".join(f"- {s}" for s in statements) if statements else "- Not applicable (approval)"
    prompt = f"""Write a concise, professional customer-facing home-equity loan decision email (150-250 words).
Decision: {'approval recommendation' if decision == 'approved' else 'decline recommendation'}
Applicant name: {applicant_name or 'Dear Applicant'}
Loan amount: ${data['LOAN']:,.0f}; property value: ${data['VALUE']:,.0f}; existing mortgage: ${data['MORTDUE']:,.0f}
Purpose: {data['REASON']}
Principal reasons (use these statements verbatim, in this order, and add nothing else as a reason):
{reason_block}
Do not mention AI, algorithms, models, scores, credit scores, occupation, or protected characteristics.
For a decline, give respectful next steps. For approval, outline verification steps.
Sign off exactly as: EquiLead Financial, Loan Services Team."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    result = client.messages.create(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"), max_tokens=700,
                                    system="You are a careful loan services writer. Return only the email.",
                                    messages=[{"role": "user", "content": prompt}])
    return result.content[0].text


def _language_review(email):
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    result = client.messages.create(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"), max_tokens=400,
        system='Return only JSON: {"score":0.0,"issues":[],"summary":"..."}. Score customer-facing lending language for discriminatory or discouraging wording from 0 to 1.',
        messages=[{"role": "user", "content": f"Review this draft for bias and tone:\n\n{email}"}])
    raw = result.content[0].text
    try:
        return json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return {"score": 0.5, "issues": [], "summary": "The language review response could not be parsed; human review is required."}


def run_pipeline(data, applicant_name=""):
    """Return fields consumed by app.py and models.LoanDecision."""
    model_path, encoders_path, card = check_prerequisites()
    import shap
    import xgboost as xgb
    with encoders_path.open("rb") as handle:
        encoders = pickle.load(handle)
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    frame = _prepare(data, encoders)
    raw_score = float(model.predict_proba(frame)[0][1])
    calibrated = _calibrate(raw_score)
    score = calibrated if calibrated is not None else raw_score
    threshold = float(card["threshold"])
    decision = "denied" if score >= threshold else "approved"
    confidence = float(max(score, 1 - score) * 100)
    reasons = _reasons(shap.TreeExplainer(model), frame)
    reason_codes = adverse_action_reasons(reasons, decision)
    email = _draft_email(decision, data, reason_codes, applicant_name)
    review = _language_review(email)
    bias_score = float(review.get("score", 0.5))
    action = "send" if bias_score < 0.3 else ("escalate" if bias_score > 0.7 else "rewritten")
    if any(r.get("review_required") for r in reason_codes):
        action = "escalate"  # a reason without an approved code must be seen by a person
    result = {"decision": decision, "default_probability": score, "raw_score": raw_score,
              "calibrated_probability": calibrated, "confidence": confidence,
              "shap_reasons": reasons, "reason_codes": reason_codes, "reason_code_version": REASON_CODE_VERSION,
              "email_draft": email, "final_email": email,
              "bias_score": bias_score, "bias_action": action, "bias_rewrites": 0,
              "bias_audit_trail": [{"attempt": 0, "bias_score": bias_score, "issues": review.get("issues", []), "summary": review.get("summary", "")}],
              "is_demo": 0}
    result.update(decision_metadata(card, calibrated))
    result["model_version"] = f"{card['model_version']} ({model_path.name})"
    return result
