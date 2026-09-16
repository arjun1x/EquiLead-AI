"""EquiLead: flat FastAPI/Jinja project. Run python run.py. No build step."""
import csv
import datetime as dt
import hashlib
import io
import logging
import os
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from security import load_secrets, limiter, client_key, has_role
load_secrets()  # decrypts secrets.enc into the environment when SECRETS_KEY is set; fails closed otherwise

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from sqlalchemy import select, func, case, or_
from sqlalchemy.exc import IntegrityError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware
from starlette.concurrency import run_in_threadpool
from models import Session, User, LoanDecision, init_db, DEMO_MODE, ROLES, append_audit, utcnow
from demo_data import DEFAULTS, DEMO_OWNER, DEMO_REVIEWER, FEATURE_NAMES, fixture, make_record, store_record, seed_demo
from model_card import load_model_card
from monitoring import build_report, row_from_record
from reason_codes import approval_status

log = logging.getLogger("equilead")
secret = os.getenv("SESSION_SECRET", "")
if not secret and not DEMO_MODE:
    raise RuntimeError("Set SESSION_SECRET to a random value before running with DEMO_MODE=0.")
MODEL_CARD = load_model_card(DEMO_MODE)  # research mode refuses to start without a complete model card

@asynccontextmanager
async def lifespan(app):
    init_db()
    if DEMO_MODE:
        seed_demo()
    yield

app = FastAPI(title="EquiLead", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(SessionMiddleware, secret_key=secret or secrets.token_hex(32),
    same_site="lax", https_only=os.getenv("COOKIE_SECURE", "0") == "1", max_age=8*3600)
templates = Jinja2Templates(directory=str(ROOT))
templates.env.globals.update(feature_names=FEATURE_NAMES)
templates.env.filters["money"] = lambda v: f"${float(v or 0):,.0f}"
templates.env.filters["pct"] = lambda v: "—" if v is None else f"{float(v)*100:.1f}%"

ASSETS = {"styles.css": "text/css", "app.js": "text/javascript", "scene.js": "text/javascript",
          "three.module.min.js": "text/javascript", "logo.svg": "image/svg+xml", "house-fallback.svg": "image/svg+xml"}
ERROR_TITLES = {403: "A fresh start is needed.", 404: "That page isn’t here.", 422: "That address didn’t look right.",
                429: "A short pause is needed.", 500: "Something went wrong on our side."}


def error_page(request, code, message):
    return templates.TemplateResponse(request, "error.html", {"code": code, "message": message,
        "title": ERROR_TITLES.get(code, "Something needs attention.")}, status_code=code)


@app.middleware("http")
async def guard(request, call_next):
    if request.method == "POST" and not limiter.allow(client_key(request)):
        response = error_page(request, 429, "Too many requests from this connection. Wait a minute and try again.")
    else:
        response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store" if not request.url.path.startswith("/assets/") else "public, max-age=3600"
    return response


@app.get("/assets/{filename}")
def asset(filename: str):
    if filename not in ASSETS:
        raise HTTPException(404)
    return FileResponse(ROOT / filename, media_type=ASSETS[filename])


@app.get("/healthz")
def healthz():
    return JSONResponse({"status": "ok", "demo_mode": DEMO_MODE, "model_version": MODEL_CARD["model_version"]})


# ── Helpers ──────────────────────────────────────────────

def csrf(request):
    if "csrf" not in request.session:
        request.session["csrf"] = secrets.token_urlsafe(32)
    return request.session["csrf"]


def check_csrf(request, form):
    token = request.session.get("csrf", "")
    if not token or not secrets.compare_digest(token, str(form.get("csrf_token", ""))):
        raise HTTPException(403, "Your session expired. Reload the form and try again.")


def user(request):
    account = request.session.get("user")
    if account and "role" not in account:
        account["role"] = "analyst"
    return account


def render(request, name, status_code=200, **context):
    account = user(request)
    base = dict(user=account, csrf_token=csrf(request), demo_mode=DEMO_MODE, active="",
                can_review=has_role(account, "reviewer"), model_card=MODEL_CARD,
                reason_code_status=approval_status(MODEL_CARD))
    base.update(context)
    return templates.TemplateResponse(request, name, base, status_code=status_code)


def go(url):
    return RedirectResponse(url, status_code=303)


def scoped_query(request):
    """Analysts see their own records; reviewers and admins see the whole workspace."""
    account = user(request)
    query = select(LoanDecision)
    return query if has_role(account, "reviewer") else query.where(LoanDecision.owner_email == account["email"])


def password_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600_000).hex()


def audit(actor, action, record_id=None, detail=None):
    with Session() as db:
        append_audit(db, actor, action, record_id, detail)
        db.commit()


# ── Authentication ───────────────────────────────────────

@app.get("/login")
def login(request: Request):
    return go("/") if user(request) else render(request, "login.html", mode="login", error=None, values={})


@app.get("/register")
def register(request: Request):
    return go("/") if user(request) else render(request, "login.html", mode="register", error=None, values={})


@app.post("/demo")
async def demo(request: Request):
    form = await request.form()
    check_csrf(request, form)
    if not DEMO_MODE:
        raise HTTPException(404)
    role = "reviewer" if str(form.get("role", "")) == "reviewer" else "analyst"
    request.session.clear()
    if role == "reviewer":
        request.session["user"] = {"name": "Sam Okafor", "email": DEMO_REVIEWER, "role": "reviewer"}
    else:
        request.session["user"] = {"name": "Alex Morgan", "email": DEMO_OWNER, "role": "analyst"}
    audit(request.session["user"]["email"], "demo_login", None, {"role": role})
    return go("/")


@app.post("/login")
@app.post("/register")
async def auth(request: Request):
    form = await request.form()
    check_csrf(request, form)
    mode = "register" if request.url.path == "/register" else "login"
    email = str(form.get("email", "")).strip().lower()
    name = str(form.get("name", "")).strip()
    password = str(form.get("password", ""))
    error = None
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email) or len(email) > 254:
        error = "Enter a valid email address."
    elif len(password) > 256:
        error = "Password must be 256 characters or fewer."
    elif mode == "register" and (len(password) < 12 or not 1 <= len(name) <= 100):
        error = "Enter your name and a password with at least 12 characters."
    if error:
        return render(request, "login.html", 422, mode=mode, error=error, values={"name": name, "email": email})
    with Session() as db:
        account = db.get(User, email)
        if mode == "register":
            if account:
                error = "This email is already registered. Sign in to continue."
            else:
                salt = secrets.token_hex(24)
                hashed = await run_in_threadpool(password_hash, password, salt)
                account = User(email=email, name=name, salt=salt, password_hash=hashed, role="analyst")
                db.add(account)
                try:
                    db.commit()
                except IntegrityError:
                    db.rollback()
                    error = "This email is already registered. Sign in to continue."
        else:
            check = await run_in_threadpool(password_hash, password, account.salt if account else "dummy-salt")
            if not account or not secrets.compare_digest(check, account.password_hash):
                error = "Email or password is incorrect. Please try again."
        if error:
            if mode == "login":
                append_audit(db, "anonymous", "login_failed", None, {"client": client_key(request)})
                db.commit()
            return render(request, "login.html", 422, mode=mode, error=error, values={"name": name, "email": email})
        request.session.clear()
        request.session["user"] = {"name": account.name, "email": account.email, "role": account.role or "analyst"}
        append_audit(db, account.email, "registered" if mode == "register" else "login", None, {})
        db.commit()
    return go("/")


@app.post("/logout")
async def logout(request: Request):
    check_csrf(request, await request.form())
    account = user(request)
    request.session.clear()
    if account:
        audit(account["email"], "logout")
    return go("/login")


# ── Workspace ────────────────────────────────────────────

@app.get("/")
def home(request: Request):
    if not user(request):
        return go("/login")
    with Session() as db:
        scope = scoped_query(request)
        records = db.scalars(scope.order_by(LoanDecision.created_at.desc()).limit(5)).all()
        stats_query = select(func.count(LoanDecision.id), func.sum(LoanDecision.loan_amount),
            func.sum(case((LoanDecision.decision == "approved", 1), else_=0)),
            func.sum(case((LoanDecision.bias_action == "escalate", 1), else_=0)))
        if not has_role(user(request), "reviewer"):
            stats_query = stats_query.where(LoanDecision.owner_email == user(request)["email"])
        stats = db.execute(stats_query).one()
    return render(request, "dashboard.html", active="overview", records=records,
                  total=stats[0], volume=stats[1] or 0, approved=stats[2] or 0, review=stats[3] or 0)


@app.get("/apply")
def application(request: Request):
    if not user(request):
        return go("/login")
    values = DEFAULTS.copy() if DEMO_MODE else {key: "" for key in DEFAULTS}
    values.update(reason="HomeImp", job="Other", demo_outcome="approved")
    return render(request, "form.html", active="apply", values=values, errors={})


class LoanInput(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, str_strip_whitespace=True, extra="ignore")
    applicant_name: str = Field(default="", max_length=200)
    loan: float = Field(gt=0, le=10_000_000)
    mortdue: float = Field(ge=0, le=100_000_000)
    value: float = Field(gt=0, le=100_000_000)
    reason: Literal["HomeImp", "DebtCon"]
    job: Literal["Mgr", "Office", "Other", "ProfExe", "Sales", "Self"]
    yoj: float = Field(ge=0, le=80)
    derog: int = Field(ge=0, le=1000)
    delinq: int = Field(ge=0, le=1000)
    clage: float = Field(ge=0, le=1200)
    ninq: int = Field(ge=0, le=1000)
    clno: int = Field(ge=0, le=1000)
    debtinc: float = Field(ge=0, le=1000)
    demo_outcome: Literal["approved", "denied", "review"] = "approved"


@app.post("/apply")
async def apply_loan(request: Request):
    account = user(request)
    if not account:
        return go("/login")
    form = await request.form()
    check_csrf(request, form)
    values = {k: str(v) for k, v in form.items()}
    try:
        parsed = LoanInput.model_validate(values)
    except ValidationError as exc:
        errors = {".".join(str(part) for part in e["loc"]) or "form": e["msg"] for e in exc.errors()}
        return render(request, "form.html", 422, active="apply", values=values, errors=errors)
    data = {k.upper(): v for k, v in parsed.model_dump().items() if k not in ("applicant_name", "demo_outcome")}
    if DEMO_MODE:
        result = fixture(data, parsed.demo_outcome)
    else:
        try:
            from pipeline import run_pipeline
            result = await run_in_threadpool(run_pipeline, data, parsed.applicant_name)
        except Exception:
            log.exception("Application pipeline could not complete")
            return render(request, "form.html", 503, active="apply", values=values,
                errors={"pipeline": "Analysis could not finish. Your entries are preserved. Check the server configuration and try again."})
    with Session() as db:
        record = store_record(db, make_record(account["email"], parsed.applicant_name, data, result))
        append_audit(db, account["email"], "decision_created", record.id,
                     {"model_version": record.model_version, "is_demo": record.is_demo, "record_hash": record.record_hash})
        db.commit()
        record_id = record.id
    return go(f"/decisions/{record_id}")


@app.get("/decisions/{record_id}")
def result(request: Request, record_id: int):
    if not user(request):
        return go("/login")
    with Session() as db:
        record = db.scalar(scoped_query(request).where(LoanDecision.id == record_id))
    if not record:
        raise HTTPException(404, "Decision not found")
    reasons = record.get_shap()
    return render(request, "result.html", active="audit", r=record, reasons=reasons, codes=record.get_reason_codes(),
                  max_impact=max([abs(r["shap_value"]) for r in reasons] or [1]) or 1)


@app.post("/decisions/{record_id}/review")
async def review(request: Request, record_id: int):
    account = user(request)
    if not account:
        return go("/login")
    form = await request.form()
    check_csrf(request, form)
    if not has_role(account, "reviewer"):
        raise HTTPException(403, "Only a reviewer can record a human decision.")
    decision = str(form.get("human_decision", ""))
    note = str(form.get("note", "")).strip()
    outcome = str(form.get("outcome", ""))
    if decision not in ("pending", "approved", "declined") or len(note) > 500 or outcome not in ("", "0", "1"):
        raise HTTPException(422, "The review form contained a value that is not allowed.")
    with Session() as db:
        record = db.get(LoanDecision, record_id)
        if not record:
            raise HTTPException(404, "Decision not found")
        record.human_decision = decision
        record.human_note = note
        record.reviewed_by = account["email"]
        record.reviewed_at = utcnow()
        record.outcome = int(outcome) if outcome else None
        append_audit(db, account["email"], "human_review_recorded", record.id,
                     {"human_decision": decision, "outcome": record.outcome, "note_sha256": hashlib.sha256(note.encode()).hexdigest()})
        db.commit()
    return go(f"/decisions/{record_id}")


def filtered_query(request, q, status):
    query = scoped_query(request)
    if q.strip():
        search = q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        query = query.where(LoanDecision.applicant_name.ilike(f"%{search}%", escape="\\"))
    if status == "review":
        query = query.where(LoanDecision.bias_action == "escalate")
    elif status == "awaiting":
        query = query.where(or_(LoanDecision.human_decision.is_(None), LoanDecision.human_decision == "pending"))
    elif status in ("approved", "denied"):
        query = query.where(LoanDecision.decision == status)
    return query


@app.get("/audit")
def audit_page(request: Request, q: str = "", status: str = "all", page: int = 1):
    if not user(request):
        return go("/login")
    q = q[:200]
    query = filtered_query(request, q, status)
    page = max(1, page)
    with Session() as db:
        total = db.scalar(select(func.count()).select_from(query.subquery()))
        pages = max(1, (total + 9) // 10)
        page = min(page, pages)
        records = db.scalars(query.order_by(LoanDecision.created_at.desc()).offset((page - 1) * 10).limit(10)).all()
    return render(request, "audit.html", active="audit", records=records, q=q, status=status,
                  total=total, page=page, pages=pages)


def csv_safe(value):
    text = str(value if value is not None else "")
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else text


@app.get("/audit.csv")
def export(request: Request, q: str = "", status: str = "all"):
    account = user(request)
    if not account:
        return go("/login")
    with Session() as db:
        records = db.scalars(filtered_query(request, q[:200], status).order_by(LoanDecision.created_at.desc())).all()
        append_audit(db, account["email"], "csv_exported", None, {"rows": len(records), "q": bool(q.strip()), "status": status})
        db.commit()
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["ID", "Applicant", "Loan amount", "Recommendation", "Default score", "Calibration", "Threshold",
                     "Language review", "Human decision", "Model version", "Dataset version", "Training date",
                     "Created UTC", "Synthetic data", "Record hash"])
    for r in records:
        writer.writerow([r.id, csv_safe(r.applicant_name), r.loan_amount, r.decision, r.default_probability,
                         csv_safe(r.calibration), r.threshold, r.bias_action, r.human_decision or "pending",
                         csv_safe(r.model_version), csv_safe(r.dataset_version), csv_safe(r.training_date),
                         r.created_at.isoformat() if r.created_at else "", bool(r.is_demo), r.record_hash or ""])
    return Response(output.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=equilead-decisions.csv"})


@app.get("/monitoring")
def monitoring_page(request: Request, days: int = 90):
    account = user(request)
    if not account:
        return go("/login")
    if not has_role(account, "reviewer"):
        raise HTTPException(403, "Monitoring is available to reviewers and administrators.")
    days = min(max(days, 1), 3650)
    since = utcnow().replace(tzinfo=None) - dt.timedelta(days=days)
    with Session() as db:
        rows = [row_from_record(r) for r in db.scalars(select(LoanDecision).where(LoanDecision.created_at >= since).order_by(LoanDecision.id))]
    return render(request, "monitoring.html", active="monitoring", report=build_report(rows, MODEL_CARD, days), days=days)


@app.get("/guide")
def guide(request: Request):
    if not user(request):
        return go("/login")
    return render(request, "guide.html", active="guide")


# ── Errors ───────────────────────────────────────────────

@app.exception_handler(StarletteHTTPException)  # also covers FastAPI's subclass and unmatched routes
async def http_error(request: Request, exc):
    return error_page(request, exc.status_code, exc.detail or "Nothing was found at that address.")


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc):
    return error_page(request, 422, "Part of that address or query was not in the expected format.")


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc):
    log.exception("Unhandled error on %s", request.url.path)
    return error_page(request, 500, "The request could not be completed. Nothing was sent to any applicant.")
