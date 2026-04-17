"""
FastAPI web application — EquiLend AI loan decisioning platform.
"""

import os
import json
from pathlib import Path
from contextlib import asynccontextmanager

import xgboost as xgb
import shap
import pickle
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.data.schema import (
    MODEL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
)
from src.ml.train import explain_single
from src.llm.email_generator import generate_email
from src.agents.bias_detector import run_bias_agent
from src.audit.models import init_db, Session, LoanDecision

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
TEMPLATES_DIR = PROJECT_ROOT / "src" / "api" / "templates"

model = None
explainer = None
label_encoders = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, explainer, label_encoders

    init_db()

    model_path = MODELS_DIR / "xgb_model.json"
    if not model_path.exists():
        from src.ml.train import train
        print("No saved model found, training...")
        results = train(save=True)
        model = results["model"]
        explainer = results["explainer"]
        label_encoders = results["label_encoders"]
    else:
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        explainer = shap.TreeExplainer(model)
        with open(MODELS_DIR / "label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)

    print("EquiLend AI ready.")
    yield


app = FastAPI(title="EquiLend AI", lifespan=lifespan)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def predict_single(applicant_data: dict) -> tuple[float, float]:
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
    return float(proba[1]), float(max(proba) * 100)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "form.html")


@app.post("/apply", response_class=HTMLResponse)
async def apply_loan(
    request: Request,
    applicant_name: str = Form(""),
    loan: float = Form(...),
    mortdue: float = Form(0),
    value: float = Form(0),
    reason: str = Form("HomeImp"),
    job: str = Form("Other"),
    yoj: float = Form(0),
    derog: float = Form(0),
    delinq: float = Form(0),
    clage: float = Form(0),
    ninq: float = Form(0),
    clno: float = Form(0),
    debtinc: float = Form(0),
):
    applicant_data = {
        "LOAN": loan, "MORTDUE": mortdue, "VALUE": value,
        "REASON": reason, "JOB": job, "YOJ": yoj,
        "DEROG": derog, "DELINQ": delinq, "CLAGE": clage,
        "NINQ": ninq, "CLNO": clno, "DEBTINC": debtinc,
    }

    default_prob, confidence = predict_single(applicant_data)
    decision = "denied" if default_prob >= 0.5 else "approved"

    shap_reasons = explain_single(model, explainer, applicant_data, label_encoders)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    email_result = generate_email(
        decision=decision,
        confidence=confidence,
        applicant_data=applicant_data,
        shap_reasons=shap_reasons,
        applicant_name=applicant_name or None,
    )

    agent_result = run_bias_agent(email_result["email"], api_key=api_key)

    session = Session()
    record = LoanDecision(
        applicant_name=applicant_name or None,
        loan_amount=loan, mortgage_due=mortdue, property_value=value,
        reason=reason, job=job, yoj=yoj, derog=derog, delinq=delinq,
        clage=clage, ninq=ninq, clno=clno, debtinc=debtinc,
        decision=decision,
        default_probability=round(default_prob, 4),
        confidence=round(confidence, 1),
        email_draft=email_result["email"],
        final_email=agent_result.final_email,
        bias_score=agent_result.bias_scores[-1] if agent_result.bias_scores else 0,
        bias_action=agent_result.decision,
        bias_rewrites=agent_result.rewrites,
    )
    record.set_shap(shap_reasons)
    record.set_bias_trail(agent_result.audit_trail)
    session.add(record)
    session.commit()
    record_id = record.id
    session.close()

    return templates.TemplateResponse(request, "result.html", {
        "applicant_name": applicant_name or "Applicant",
        "decision": decision,
        "default_prob": f"{default_prob:.1%}",
        "confidence": f"{confidence:.1f}",
        "shap_reasons": shap_reasons,
        "email": agent_result.final_email,
        "bias_score": f"{agent_result.bias_scores[-1]:.2f}" if agent_result.bias_scores else "N/A",
        "bias_action": agent_result.decision,
        "bias_rewrites": agent_result.rewrites,
        "record_id": record_id,
        "applicant_data": applicant_data,
    })


@app.get("/audit", response_class=HTMLResponse)
async def audit_log(request: Request):
    session = Session()
    records = session.query(LoanDecision).order_by(LoanDecision.created_at.desc()).limit(50).all()
    session.close()
    return templates.TemplateResponse(request, "audit.html", {
        "records": records,
    })
