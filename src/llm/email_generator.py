"""
Phase 3: Claude-powered email generator.

Takes an XGBoost decision + SHAP explanations and generates a
professional customer-facing decision email via Claude.
"""

from pathlib import Path

import anthropic
from jinja2 import Template

PROMPTS_DIR = Path(__file__).parent / "prompts"

SHAP_TO_ENGLISH = {
    "LOAN": "loan amount requested",
    "MORTDUE": "existing mortgage balance",
    "VALUE": "property value",
    "REASON": "stated loan purpose",
    "JOB": "occupation type",
    "YOJ": "employment tenure",
    "DEROG": "derogatory marks on credit history",
    "DELINQ": "delinquent credit lines",
    "CLAGE": "length of credit history",
    "NINQ": "recent credit inquiries",
    "CLNO": "number of open credit lines",
    "DEBTINC": "debt-to-income ratio",
}


def _load_template(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def _translate_shap_reasons(shap_reasons: list[dict]) -> list[dict]:
    translated = []
    for r in shap_reasons:
        feature = r["feature"]
        shap_val = r["shap_value"]
        translated.append({
            "feature": feature,
            "explanation": SHAP_TO_ENGLISH.get(feature, feature),
            "direction": "increases default risk" if shap_val > 0 else "decreases default risk",
            "shap_value": shap_val,
        })
    return translated


def generate_email(
    decision: str,
    confidence: float,
    applicant_data: dict,
    shap_reasons: list[dict],
    applicant_name: str | None = None,
) -> dict:
    """
    Generate a decision email using Claude.

    Args:
        decision: "approved" or "denied"
        confidence: model confidence 0-100
        applicant_data: raw feature values for the applicant
        shap_reasons: top SHAP reasons from explain_single()
        applicant_name: optional applicant name

    Returns:
        dict with "email", "model_used", "input_tokens", "output_tokens"
    """
    system_prompt = _load_template("email_system.txt")
    user_template = Template(_load_template("email_user.txt"))

    reasons = _translate_shap_reasons(shap_reasons)

    user_prompt = user_template.render(
        decision="denial" if decision == "denied" else "approval",
        decision_label=decision.upper(),
        confidence=round(confidence, 1),
        loan_amount=f"{applicant_data.get('LOAN', 'N/A'):,.0f}" if isinstance(applicant_data.get('LOAN'), (int, float)) else "N/A",
        property_value=f"{applicant_data.get('VALUE', 'N/A'):,.0f}" if isinstance(applicant_data.get('VALUE'), (int, float)) else "N/A",
        mortgage_due=f"{applicant_data.get('MORTDUE', 'N/A'):,.0f}" if isinstance(applicant_data.get('MORTDUE'), (int, float)) else "N/A",
        reason=applicant_data.get("REASON", "N/A"),
        job=applicant_data.get("JOB", "N/A"),
        yoj=applicant_data.get("YOJ", "N/A"),
        debtinc=f"{applicant_data.get('DEBTINC', 'N/A'):.1f}" if isinstance(applicant_data.get('DEBTINC'), (int, float)) else "N/A",
        reasons=reasons,
    )

    if applicant_name:
        user_prompt = f"Applicant name: {applicant_name}\n\n" + user_prompt

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return {
        "email": response.content[0].text,
        "model_used": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
