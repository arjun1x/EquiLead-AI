"""
FastAPI web application — EquiLend AI loan decisioning platform.
"""

import os
import json
import hashlib
import secrets
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import xgboost as xgb
import shap
import pickle
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

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
USERS_FILE = PROJECT_ROOT / "data" / "users.json"

model = None
explainer = None
label_encoders = None


# ── Simple user store ────────────────────────────────────

def _load_users() -> dict:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text())
    return {}


def _save_users(users: dict):
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2))


def _hash_password(password: str, salt: str = "") -> tuple[str, str]:
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return hashed, salt


def _verify_password(password: str, hashed: str, salt: str) -> bool:
    check, _ = _hash_password(password, salt)
    return check == hashed


# ── App setup ────────────────────────────────────────────

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

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET", "equilend-dev-secret-change-me"),
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def get_user(request: Request) -> dict | None:
    return request.session.get("user")


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


# ── Auth routes ──────────────────────────────────────────


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if get_user(request):
        return RedirectResponse(url="/", status_code=302)
    error = request.query_params.get("error")
    return templates.TemplateResponse(request, "login.html", {"error": error})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, email: str = Form(...), password: str = Form(...)):
    users = _load_users()
    user = users.get(email)
    if not user or not _verify_password(password, user["password"], user["salt"]):
        return templates.TemplateResponse(request, "login.html", {
            "error": "Invalid email or password.",
        })
    request.session["user"] = {"name": user["name"], "email": email}
    return RedirectResponse(url="/", status_code=302)


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    if get_user(request):
        return RedirectResponse(url="/", status_code=302)
    error = request.query_params.get("error")
    return templates.TemplateResponse(request, "register.html", {"error": error})


@app.post("/register", response_class=HTMLResponse)
async def register_submit(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
):
    users = _load_users()
    if email in users:
        return templates.TemplateResponse(request, "register.html", {
            "error": "An account with this email already exists.",
        })
    if len(password) < 6:
        return templates.TemplateResponse(request, "register.html", {
            "error": "Password must be at least 6 characters.",
        })
    hashed, salt = _hash_password(password)
    users[email] = {"name": name, "password": hashed, "salt": salt}
    _save_users(users)
    request.session["user"] = {"name": name, "email": email}
    return RedirectResponse(url="/", status_code=302)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


# ── App routes (protected) ───────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "form.html", {"user": user})


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
    user = get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

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
        "user": user,
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
    user = get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    session = Session()
    records = session.query(LoanDecision).order_by(LoanDecision.created_at.desc()).limit(50).all()
    session.close()
    return templates.TemplateResponse(request, "audit.html", {
        "user": user,
        "records": records,
    })
