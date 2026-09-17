"""EquiLead — single-folder FastAPI/Jinja decision-support workspace for home-equity loans and HELOCs.

Run:  python run.py   (or: python -m uvicorn app:app --port 8000)
Every template, stylesheet, script and model artifact is served from this folder. No build step.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import logging
import os
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from security import load_secrets, limiter, client_key, has_role
load_secrets()

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

import scoring
import workflow
from calculations import (ApplicationInput, DOCUMENT_ITEMS, EMPLOYMENT_STATUS, FEATURE_LABELS, LOAN_PURPOSES, OCCUPANCY_TYPES,
                          PRODUCT_TYPES, PROPERTY_TYPES, derive_metrics, format_value)
from demo_data import DEMO_OWNER, DEMO_REVIEWER, seed_demo, store_scoring
from fairness import GOVERNANCE, PROTECTED_CLASSES, compare_models
from letters import LETTER_TEMPLATE_VERSION, LETTER_TYPES, check_language, correct_draft, generate_letter, letter_type_for
from models import (DEMO_MODE, Application, AuditEvent, Letter, ModelRun, Session, User, append_audit, init_db, next_reference,
                    record_change, utcnow, verify_audit_chain)
from monitoring import build_report, row_from_application
from train import MODEL_DOCS
from policy import BANDS, RECOMMENDATION_LABELS
from reason_codes import REASON_CODES, REASON_CODE_VERSION, customer_statements

log = logging.getLogger("equilead")
secret = os.getenv("SESSION_SECRET", "")
if not secret and not DEMO_MODE:
    raise RuntimeError("Set SESSION_SECRET to a random value before running with DEMO_MODE=0.")

REC_TO_DECISION = {"approve": "approved", "conditional_approval": "conditionally_approved", "manual_review": "manual_review", "decline": "declined"}
DECISION_LABELS = {"approved": "Approved", "conditionally_approved": "Conditionally approved", "declined": "Declined", "manual_review": "Manual review"}
LETTER_TYPES_FOR_DECISION = {"approved": ["approval"], "conditionally_approved": ["conditional_approval", "incomplete"],
                             "declined": ["adverse_action"], "manual_review": ["manual_review", "incomplete"], None: ["manual_review", "incomplete"]}
QUEUE_FILTERS = {"active": "Active", "needs_decision": "Needs decision", "manual_review": "Manual review", "needs_letter": "Needs letter",
                 "draft": "Drafts", "decided": "Decided", "finalized": "Finalized", "withdrawn": "Withdrawn", "all": "All"}
ERROR_TITLES = {403: "That action is not allowed.", 404: "That page isn’t here.", 422: "Something in that request didn’t look right.",
                429: "A short pause is needed.", 500: "Something went wrong on our side."}
_TRAINING_REPORT = {"data": None, "mtime": None}


@asynccontextmanager
async def lifespan(app):
    init_db()
    status = await run_in_threadpool(scoring.load_models, True)
    if status["errors"]:
        log.warning("Model status: %s", status["errors"])
    if DEMO_MODE:
        created = await run_in_threadpool(seed_demo, scoring.score_application)
        if created:
            log.info("Seeded %d demo applications", created)
    yield


app = FastAPI(title="EquiLead", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(SessionMiddleware, secret_key=secret or secrets.token_hex(32), same_site="lax",
                   https_only=os.getenv("COOKIE_SECURE", "0") == "1", max_age=8 * 3600)
templates = Jinja2Templates(directory=str(ROOT))
templates.env.globals.update(feature_labels=FEATURE_LABELS, product_types=PRODUCT_TYPES, employment_status=EMPLOYMENT_STATUS,
                             occupancy_types=OCCUPANCY_TYPES, property_types=PROPERTY_TYPES, loan_purposes=LOAN_PURPOSES,
                             document_items=DOCUMENT_ITEMS, state_labels=workflow.LABELS, rec_labels=RECOMMENDATION_LABELS,
                             decision_labels=DECISION_LABELS, letter_types=LETTER_TYPES, bands=BANDS, format_value=format_value,
                             queue_filters=QUEUE_FILTERS, rec_to_decision=REC_TO_DECISION)
templates.env.filters["money"] = lambda v: "—" if v is None else f"${float(v):,.0f}"
templates.env.filters["pct"] = lambda v, digits=1: "—" if v is None else f"{float(v) * 100:.{digits}f}%"
templates.env.filters["num"] = lambda v, digits=1: "—" if v is None else f"{float(v):,.{digits}f}"
templates.env.filters["when"] = lambda v: "—" if not v else v.strftime("%b %d, %Y · %H:%M UTC")
templates.env.filters["day"] = lambda v: "—" if not v else v.strftime("%b %d, %Y")

ASSETS = {"styles.css": "text/css", "app.js": "text/javascript", "scene.js": "text/javascript",
          "three.module.min.js": "text/javascript", "logo.svg": "image/svg+xml", "house-fallback.svg": "image/svg+xml"}


# ── Plumbing ─────────────────────────────────────────────

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
    if filename not in ASSETS:          # allow-list: no path traversal, no serving of models or the database
        raise HTTPException(404)
    return FileResponse(ROOT / filename, media_type=ASSETS[filename])


@app.get("/healthz")
def healthz():
    return JSONResponse({"status": "ok", "demo_mode": DEMO_MODE, "models": scoring.status()})


def csrf(request):
    if "csrf" not in request.session:
        request.session["csrf"] = secrets.token_urlsafe(32)
    return request.session["csrf"]


def check_csrf(request, form_or_token):
    token = request.session.get("csrf", "")
    supplied = form_or_token if isinstance(form_or_token, str) else str(form_or_token.get("csrf_token", ""))
    if not token or not secrets.compare_digest(token, supplied):
        raise HTTPException(403, "Your session expired or the form token did not match. Reload the page and try again.")


def user(request):
    account = request.session.get("user")
    if account and "role" not in account:
        account["role"] = "analyst"
    return account


def require_user(request):
    account = user(request)
    if not account:
        raise HTTPException(403, "Sign in to continue.")
    return account


def flash(request, message, kind="info"):
    request.session["flash"] = {"message": message, "kind": kind}


def render(request, name, status_code=200, **context):
    account = user(request)
    base = dict(user=account, csrf_token=csrf(request), demo_mode=DEMO_MODE, active="", can_review=has_role(account, "reviewer"),
                model_status=scoring.status(), flash=request.session.pop("flash", None))
    base.update(context)
    return templates.TemplateResponse(request, name, base, status_code=status_code)


def go(url):
    return RedirectResponse(url, status_code=303)


def scoped(request):
    account = user(request)
    query = select(Application)
    return query if has_role(account, "reviewer") else query.where(Application.owner_email == account["email"])


def get_application(db, request, application_id):
    application = db.scalar(scoped(request).where(Application.id == application_id))
    if not application:
        raise HTTPException(404, "Application not found.")
    return application


def password_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600_000).hex()


def audit(actor, action, application_id=None, detail=None):
    with Session() as db:
        append_audit(db, actor, action, application_id, detail)
        db.commit()


def training_report():
    path = ROOT / "training_report.json"
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _TRAINING_REPORT["mtime"] != mtime:
        _TRAINING_REPORT["data"], _TRAINING_REPORT["mtime"] = json.loads(path.read_text(encoding="utf-8")), mtime
    return _TRAINING_REPORT["data"]


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
    request.session["user"] = ({"name": "Sam Okafor", "email": DEMO_REVIEWER, "role": "reviewer"} if role == "reviewer"
                               else {"name": "Alex Morgan", "email": DEMO_OWNER, "role": "analyst"})
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


# ── Dashboard and queue ──────────────────────────────────

def queue_query(request, filter_key, q):
    query = scoped(request)
    if filter_key == "active":
        query = query.where(Application.state.notin_(["finalized", "withdrawn"]))
    elif filter_key == "needs_decision":
        query = query.where(Application.state.in_(["scored", "manual_review"]))
    elif filter_key == "manual_review":
        query = query.where(Application.state == "manual_review")
    elif filter_key == "needs_letter":
        query = query.where(Application.state.in_(["approved", "conditionally_approved", "declined"]))
    elif filter_key == "draft":
        query = query.where(Application.state.in_(["draft", "submitted"]))
    elif filter_key == "decided":
        query = query.where(Application.state.in_(["approved", "conditionally_approved", "declined", "letter_generated"]))
    elif filter_key in ("finalized", "withdrawn"):
        query = query.where(Application.state == filter_key)
    if q.strip():
        search = q.strip()[:80].replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        query = query.where(or_(Application.applicant_name.ilike(f"%{search}%", escape="\\"), Application.reference.ilike(f"%{search}%", escape="\\")))
    return query


@app.get("/")
def dashboard(request: Request):
    if not user(request):
        return go("/login")
    with Session() as db:
        base = scoped(request).subquery()
        counts = dict(db.execute(select(base.c.state, func.count()).group_by(base.c.state)).all())
        attention = db.scalars(queue_query(request, "needs_decision", "").order_by(Application.created_at.asc()).limit(6)).all()
        letters_due = db.scalars(queue_query(request, "needs_letter", "").order_by(Application.updated_at.desc()).limit(6)).all()
        recent = db.scalars(select(AuditEvent).where(AuditEvent.application_id.isnot(None)).order_by(AuditEvent.id.desc()).limit(8)).all()
        refs = {a.id: a.reference for a in db.scalars(select(Application).where(Application.id.in_([e.application_id for e in recent])))} if recent else {}
    total = sum(counts.values())
    return render(request, "dashboard.html", active="dashboard", counts=counts, total=total, attention=attention, letters_due=letters_due,
                  recent=recent, refs=refs, registry=scoring.registry())


@app.get("/applications")
def queue(request: Request, filter: str = "active", q: str = "", page: int = 1):
    if not user(request):
        return go("/login")
    filter_key = filter if filter in QUEUE_FILTERS else "active"
    query = queue_query(request, filter_key, q)
    page = max(1, page)
    with Session() as db:
        total = db.scalar(select(func.count()).select_from(query.subquery()))
        pages = max(1, (total + 14) // 15)
        page = min(page, pages)
        rows = db.scalars(query.order_by(Application.updated_at.desc()).offset((page - 1) * 15).limit(15)).all()
    return render(request, "queue.html", active="applications", rows=rows, filter=filter_key, q=q[:80], total=total, page=page, pages=pages)


# ── Application form ─────────────────────────────────────

def form_values(form):
    values = {k: str(v).strip() for k, v in form.items() if k not in ("csrf_token", "action", "documents")}
    values = {k: v[:200] for k, v in values.items() if v != ""}          # blank optional fields become "not provided"
    values["documents"] = [str(v) for v in form.getlist("documents") if str(v) in DOCUMENT_ITEMS]
    return values


def default_values():
    return {"product_type": "heloc", "employment_status": "employed", "occupancy_type": "primary", "property_type": "single_family",
            "loan_purpose": "home_improvement", "term_years": "20", "interest_rate": "8.50", "documents": []}


def validation_errors(exc: ValidationError):
    return {".".join(str(p) for p in e["loc"]) or "form": e["msg"].replace("Value error, ", "") for e in exc.errors()}


def parse_submission(values):
    """Validate a form submission. Returns (ApplicationInput, {}) or (None, errors)."""
    errors = {} if values.get("applicant_name") else {"applicant_name": "Field required"}
    try:
        app_input = ApplicationInput(**values)
    except ValidationError as exc:
        errors.update(validation_errors(exc))
    return (None, errors) if errors else (app_input, {})


@app.get("/applications/new")
def new_application(request: Request):
    if not user(request):
        return go("/login")
    return render(request, "application_form.html", active="new", values=default_values(), errors={}, application=None)


@app.post("/applications/new")
async def create_application(request: Request):
    account = user(request)
    if not account:
        return go("/login")
    form = await request.form()
    check_csrf(request, form)
    values, action = form_values(form), str(form.get("action", "submit"))
    with Session() as db:
        application = Application(reference=next_reference(db), owner_email=account["email"], applicant_name=values.get("applicant_name", "")[:120],
                                  product_type=values.get("product_type", "heloc"), inputs=values, state="draft")
        if action == "save_draft":
            db.add(application); db.flush()
            append_audit(db, account["email"], "created", application.id, {"reference": application.reference, "state": "draft"})
            db.commit()
            flash(request, f"Draft {application.reference} saved.", "ok")
            return go(f"/applications/{application.id}/edit")
        app_input, errors = parse_submission(values)
        if errors:
            return render(request, "application_form.html", 422, active="new", values=values, errors=errors, application=None)
        db.add(application); db.flush()
        append_audit(db, account["email"], "created", application.id, {"reference": application.reference, "state": "draft"})
        await submit_and_score(db, request, application, app_input)
        db.commit()
        return go(f"/applications/{application.id}")


async def submit_and_score(db, request, application, app_input: ApplicationInput):
    """Shared by create / edit / re-score: validate -> submitted -> score -> scored or manual_review."""
    account = user(request)
    application.inputs = app_input.model_dump()
    application.applicant_name = app_input.applicant_name
    application.product_type = app_input.product_type
    if application.state != "submitted":
        workflow.assert_transition(application.state, "submitted")
        application.state = "submitted"
        append_audit(db, account["email"], "submitted", application.id, {"state": "submitted", "input_sha256": hashlib.sha256(json.dumps(application.inputs, sort_keys=True).encode()).hexdigest()})
    batch = (db.scalar(select(func.max(ModelRun.batch)).where(ModelRun.application_id == application.id)) or 0) + 1
    response = await run_in_threadpool(scoring.score_application, app_input)
    store_scoring(db, application, response, account["email"], batch)
    target = "manual_review" if response.consensus.routed_to_manual_review else "scored"
    workflow.assert_transition(application.state, target)
    application.state = target
    application.validation = {"flags": response.derived.get("flags", [])}
    return response


@app.get("/applications/{application_id}/edit")
def edit_application(request: Request, application_id: int):
    if not user(request):
        return go("/login")
    with Session() as db:
        application = get_application(db, request, application_id)
    if application.state not in ("draft", "submitted", "scored", "manual_review"):
        flash(request, "Inputs can no longer be edited once a decision has been recorded. Withdraw the case or reopen it by re-scoring.", "warn")
        return go(f"/applications/{application_id}")
    values = default_values()
    values.update({k: (v if isinstance(v, list) else str(v)) for k, v in (application.inputs or {}).items()})
    return render(request, "application_form.html", active="applications", values=values, errors={}, application=application)


@app.post("/applications/{application_id}/edit")
async def update_application(request: Request, application_id: int):
    account = user(request)
    if not account:
        return go("/login")
    form = await request.form()
    check_csrf(request, form)
    values, action = form_values(form), str(form.get("action", "submit"))
    with Session() as db:
        application = get_application(db, request, application_id)
        if application.state not in ("draft", "submitted", "scored", "manual_review"):
            raise HTTPException(403, "This application can no longer be edited.")
        if action == "save_draft":
            if application.state != "draft":
                raise HTTPException(403, "Only drafts can be saved without scoring. Submit to re-score.")
            record_change(db, account["email"], "draft_saved", application, {"inputs": values, "applicant_name": values.get("applicant_name", "")[:120]})
            db.commit()
            flash(request, "Draft saved.", "ok")
            return go(f"/applications/{application.id}/edit")
        app_input, errors = parse_submission(values)
        if errors:
            return render(request, "application_form.html", 422, active="applications", values=values, errors=errors, application=application)
        if application.state != "draft":
            record_change(db, account["email"], "inputs_updated", application, {"inputs": app_input.model_dump()})
            application.state = "draft"   # resubmission path: draft -> submitted -> scored
            append_audit(db, account["email"], "reopened", application.id, {"reason": "inputs changed; previous scoring superseded"})
        await submit_and_score(db, request, application, app_input)
        db.commit()
        return go(f"/applications/{application.id}")


@app.post("/applications/{application_id}/score")
async def rescore(request: Request, application_id: int):
    account = user(request)
    if not account:
        return go("/login")
    check_csrf(request, await request.form())
    with Session() as db:
        application = get_application(db, request, application_id)
        if not workflow.can_score(application.state):
            raise HTTPException(403, f"An application in state '{workflow.LABELS[application.state]}' cannot be re-scored.")
        try:
            app_input = ApplicationInput(**(application.inputs or {}))
        except ValidationError as exc:
            raise HTTPException(422, "Stored inputs are incomplete; edit the application first.") from exc
        if application.state in ("approved", "conditionally_approved", "declined"):
            append_audit(db, account["email"], "reopened", application.id, {"previous_state": application.state, "reason": "re-score requested"})
            application.state = "scored"
        elif application.state != "submitted":
            application.state = "submitted"
            append_audit(db, account["email"], "submitted", application.id, {"state": "submitted", "reason": "re-score"})
        await submit_and_score(db, request, application, app_input)
        db.commit()
    flash(request, "All three models were re-run. Previous runs are kept in the history.", "ok")
    return go(f"/applications/{application_id}")


# ── Detail, decision, letter, finalize ───────────────────

def latest_runs(application):
    runs = application.latest_runs()
    return [runs[k] for k in ("scorecard", "forest", "gbm") if k in runs]


@app.get("/applications/{application_id}")
def application_detail(request: Request, application_id: int):
    if not user(request):
        return go("/login")
    with Session() as db:
        application = get_application(db, request, application_id)
        runs = latest_runs(application)
        letter = application.current_letter()
        events = db.scalar(select(func.count(AuditEvent.id)).where(AuditEvent.application_id == application.id))
    consensus = application.consensus or {}
    recommended_decision = REC_TO_DECISION.get(consensus.get("final_recommendation"))
    return render(request, "application.html", active="applications", a=application, runs=runs, letter=letter, events=events,
                  recommended_decision=recommended_decision, blockers=[] if application.state == "finalized" else workflow.finalize_blockers(application),
                  can_decide=workflow.can_decide(application.state), can_score=workflow.can_score(application.state),
                  can_letter=workflow.can_generate_letter(application.state), customer_reasons=customer_statements(application.reason_codes or {}))


@app.post("/applications/{application_id}/decision")
async def record_decision(request: Request, application_id: int):
    account = user(request)
    if not account:
        return go("/login")
    form = await request.form()
    check_csrf(request, form)
    if not has_role(account, "reviewer"):
        raise HTTPException(403, "Only a loan officer (reviewer role) can record a decision.")
    decision = str(form.get("human_decision", ""))
    note = str(form.get("note", "")).strip()[:2000]
    override_reason = str(form.get("override_reason", "")).strip()[:2000]
    signoff_name = str(form.get("signoff_name", "")).strip()[:120]
    signoff_role = str(form.get("signoff_role", "")).strip()[:80]
    conditions = [c.strip()[:300] for c in str(form.get("conditions", "")).splitlines() if c.strip()][:10]
    amount_text = str(form.get("approved_amount", "")).strip().replace(",", "").replace("$", "")
    if decision not in DECISION_LABELS:
        raise HTTPException(422, "Choose a decision.")
    with Session() as db:
        application = get_application(db, request, application_id)
        if not workflow.can_decide(application.state):
            raise HTTPException(403, f"A decision cannot be recorded while the application is '{workflow.LABELS[application.state]}'.")
        consensus = application.consensus or {}
        recommended = REC_TO_DECISION.get(consensus.get("final_recommendation"))
        # Any final decision that differs from the combined recommendation (including resolving a manual-review
        # routing) must be explained. Sending a case back to manual review is a deferral, not an override.
        override = decision != "manual_review" and recommended is not None and decision != recommended
        errors = []
        if override and len(override_reason) < 20:
            errors.append("This decision differs from the combined recommendation. Explain why in at least 20 characters.")
        if decision != "manual_review" and not (signoff_name and signoff_role):
            errors.append("Sign-off name and role are required to record a decision.")
        approved_amount = None
        if decision in ("approved", "conditionally_approved"):
            try:
                approved_amount = float(amount_text) if amount_text else float(application.inputs.get("requested_amount", 0))
            except ValueError:
                errors.append("Approved amount must be a number.")
            if approved_amount is not None and (approved_amount <= 0 or approved_amount > float(application.inputs.get("requested_amount", 0)) * 1.000001):
                errors.append("Approved amount must be positive and no more than the requested amount.")
        if decision == "conditionally_approved" and not conditions:
            conditions = ["Provide the outstanding items listed on the application."]
        if errors:
            flash(request, " ".join(errors), "warn")
            return go(f"/applications/{application_id}#decision")
        if application.state != decision:
            workflow.assert_transition(application.state, decision)
        previous_letter = application.current_letter()
        changes = {"human_decision": decision, "decision_note": note, "approved_amount": approved_amount, "conditions": conditions,
                   "override": override, "override_reason": override_reason if override else None, "decided_by": account["email"],
                   "decided_at": utcnow(), "signoff_name": signoff_name or None, "signoff_role": signoff_role or None,
                   "state": decision, "letter_approved_at": None}
        record_change(db, account["email"], "decision_recorded", application, changes,
                      {"recommended": recommended, "override": override, "previous_state": application.state})
        if previous_letter and previous_letter.status != "superseded":
            previous_letter.status = "superseded"
            append_audit(db, account["email"], "letter_superseded", application.id, {"letter_id": previous_letter.id, "reason": "decision changed"})
        db.commit()
    flash(request, "Decision recorded." + (" Override explanation stored." if override else ""), "ok")
    return go(f"/applications/{application_id}")


@app.get("/applications/{application_id}/letter")
def letter_page(request: Request, application_id: int):
    if not user(request):
        return go("/login")
    with Session() as db:
        application = get_application(db, request, application_id)
        letter = application.current_letter()
        history = list(application.letters)
    allowed = LETTER_TYPES_FOR_DECISION.get(application.human_decision, LETTER_TYPES_FOR_DECISION[None])
    default_type = letter_type_for(application.human_decision, (application.derived or {}).get("missing_documents")) if application.human_decision else "manual_review"
    remaining = check_language(letter.corrected_text, stored_decision=application.human_decision, stored_reasons=letter.decision_snapshot.get("reasons"), letter_type=letter.letter_type) if letter else []
    return render(request, "letter.html", active="applications", a=application, letter=letter, history=history, allowed=allowed,
                  default_type=default_type, remaining=remaining, can_letter=workflow.can_generate_letter(application.state),
                  customer_reasons=customer_statements(application.reason_codes or {}),
                  blockers=[] if application.state == "finalized" else workflow.finalize_blockers(application))


@app.post("/applications/{application_id}/letter")
async def generate_letter_route(request: Request, application_id: int):
    account = user(request)
    if not account:
        return go("/login")
    form = await request.form()
    check_csrf(request, form)
    if not has_role(account, "reviewer"):
        raise HTTPException(403, "Only a loan officer can generate a decision letter.")
    letter_type = str(form.get("letter_type", ""))
    with Session() as db:
        application = get_application(db, request, application_id)
        if not workflow.can_generate_letter(application.state):
            raise HTTPException(403, f"A letter cannot be generated while the application is '{workflow.LABELS[application.state]}'.")
        allowed = LETTER_TYPES_FOR_DECISION.get(application.human_decision, LETTER_TYPES_FOR_DECISION[None])
        if letter_type not in allowed:
            raise HTTPException(422, f"Letter type '{letter_type}' does not match the recorded decision.")
        derived = application.derived or {}
        reasons = customer_statements(application.reason_codes or {}) if letter_type == "adverse_action" else []
        conditions = list(application.conditions or [])
        if letter_type == "incomplete":
            conditions = [DOCUMENT_ITEMS[d] for d in derived.get("missing_documents", [])] + [c for c in conditions if c not in ("Provide the outstanding items listed on the application.",)]
        draft = generate_letter(letter_type, applicant_name=application.applicant_name, reference=application.reference,
                                product_label=PRODUCT_TYPES.get(application.product_type, "Home equity product"),
                                approved_amount=application.approved_amount, term_years=(application.inputs or {}).get("term_years"),
                                conditions=conditions, reasons=reasons, signoff_name=application.signoff_name, signoff_role=application.signoff_role)
        issues = check_language(draft, stored_decision=application.human_decision, stored_reasons=reasons, letter_type=letter_type)
        corrected = correct_draft(draft, [dict(i) for i in issues])
        for previous in application.letters:
            if previous.status != "superseded":
                previous.status = "superseded"
        version = len(application.letters) + 1
        letter = Letter(application_id=application.id, version=version, template_version=LETTER_TEMPLATE_VERSION, letter_type=letter_type,
                        draft_text=draft, issues=issues, corrected_text=corrected, created_by=account["email"],
                        decision_snapshot={"human_decision": application.human_decision, "reasons": reasons, "approved_amount": application.approved_amount,
                                           "conditions": conditions, "reason_code_version": (application.reason_codes or {}).get("version")})
        db.add(letter); db.flush()
        if application.state != "letter_generated":
            workflow.assert_transition(application.state, "letter_generated")
            application.state = "letter_generated"
        application.letter_approved_at = None
        append_audit(db, account["email"], "letter_generated", application.id, {"letter_id": letter.id, "version": version, "type": letter_type,
                     "template_version": LETTER_TEMPLATE_VERSION, "issues": [i["category"] for i in issues], "reasons": reasons})
        db.commit()
    flash(request, f"Letter version {version} generated. Review the detected issues before approving.", "ok")
    return go(f"/applications/{application_id}/letter")


@app.post("/applications/{application_id}/letter/{letter_id}/approve")
async def approve_letter(request: Request, application_id: int, letter_id: int):
    account = user(request)
    if not account:
        return go("/login")
    check_csrf(request, await request.form())
    if not has_role(account, "reviewer"):
        raise HTTPException(403, "Only a loan officer can approve a letter.")
    with Session() as db:
        application = get_application(db, request, application_id)
        letter = db.get(Letter, letter_id)
        if not letter or letter.application_id != application.id or letter.status == "superseded":
            raise HTTPException(404, "Letter not found or superseded.")
        remaining = check_language(letter.corrected_text, stored_decision=application.human_decision,
                                   stored_reasons=letter.decision_snapshot.get("reasons"), letter_type=letter.letter_type)
        blocking = [i for i in remaining if i["category"] in ("protected_class", "mismatch", "vague", "unsupported", "guarantee")]
        if blocking:
            flash(request, "The corrected draft still has blocking issues: " + "; ".join(sorted({i['message'] for i in blocking})) + " Regenerate or correct the decision first.", "warn")
            return go(f"/applications/{application_id}/letter")
        letter.status, letter.approved_by, letter.approved_at = "approved", account["email"], utcnow()
        application.letter_approved_at = letter.approved_at
        append_audit(db, account["email"], "letter_approved", application.id, {"letter_id": letter.id, "version": letter.version, "sha256": hashlib.sha256(letter.corrected_text.encode()).hexdigest()})
        db.commit()
    flash(request, "Letter approved. The case can now be finalized.", "ok")
    return go(f"/applications/{application_id}/letter")


@app.post("/applications/{application_id}/finalize")
async def finalize(request: Request, application_id: int):
    account = user(request)
    if not account:
        return go("/login")
    check_csrf(request, await request.form())
    if not has_role(account, "reviewer"):
        raise HTTPException(403, "Only a loan officer can finalize a case.")
    with Session() as db:
        application = get_application(db, request, application_id)
        blockers = workflow.finalize_blockers(application)
        if blockers:
            flash(request, "Cannot finalize: " + " ".join(blockers), "warn")
            return go(f"/applications/{application_id}")
        workflow.assert_transition(application.state, "finalized")
        record_change(db, account["email"], "finalized", application, {"state": "finalized", "finalized_at": utcnow()},
                      {"human_decision": application.human_decision, "signoff": f"{application.signoff_name}, {application.signoff_role}", "letter_id": application.current_letter().id})
        db.commit()
    flash(request, "Case finalized. The record is now read-only.", "ok")
    return go(f"/applications/{application_id}")


@app.post("/applications/{application_id}/withdraw")
async def withdraw(request: Request, application_id: int):
    account = user(request)
    if not account:
        return go("/login")
    form = await request.form()
    check_csrf(request, form)
    reason = str(form.get("reason", "")).strip()[:500]
    with Session() as db:
        application = get_application(db, request, application_id)
        workflow.assert_transition(application.state, "withdrawn")
        if not reason:
            flash(request, "Give a short reason for withdrawing the application.", "warn")
            return go(f"/applications/{application_id}")
        record_change(db, account["email"], "withdrawn", application, {"state": "withdrawn", "withdrawn_reason": reason})
        db.commit()
    flash(request, "Application withdrawn.", "ok")
    return go(f"/applications/{application_id}")


@app.get("/applications/{application_id}/history")
def history(request: Request, application_id: int):
    if not user(request):
        return go("/login")
    with Session() as db:
        application = get_application(db, request, application_id)
        events = db.scalars(select(AuditEvent).where(AuditEvent.application_id == application.id).order_by(AuditEvent.id)).all()
        runs = list(application.model_runs)
        letters_ = list(application.letters)
    batches = {}
    for run in runs:
        batches.setdefault(run.batch, []).append(run)
    return render(request, "history.html", active="applications", a=application, events=events, batches=batches, letters=letters_)


# ── Governance pages ─────────────────────────────────────

@app.get("/fairness")
def fairness_page(request: Request):
    if not user(request):
        return go("/login")
    report = training_report()
    rows = compare_models(report["fairness"]) if report else []
    return render(request, "fairness.html", active="fairness", report=report, rows=rows, governance=GOVERNANCE, protected_classes=PROTECTED_CLASSES,
                  registry=scoring.registry())


@app.get("/models")
def models_page(request: Request):
    if not user(request):
        return go("/login")
    return render(request, "models.html", active="models", registry=scoring.registry(), report=training_report(), reason_codes=REASON_CODES, model_docs=MODEL_DOCS,
                  reason_code_version=REASON_CODE_VERSION)


@app.get("/monitoring")
def monitoring_page(request: Request, days: int = 90):
    account = user(request)
    if not account:
        return go("/login")
    if not has_role(account, "reviewer"):
        raise HTTPException(403, "Monitoring is available to loan officers and administrators.")
    days = min(max(days, 1), 3650)
    since = utcnow().replace(tzinfo=None) - dt.timedelta(days=days)
    with Session() as db:
        rows = [row_from_application(a) for a in db.scalars(select(Application).where(Application.state != "draft", Application.created_at >= since).order_by(Application.id))]
    return render(request, "monitoring.html", active="monitoring", report=build_report(rows, scoring.registry(), days), days=days)


@app.get("/audit")
def audit_page(request: Request, action: str = "", page: int = 1):
    account = user(request)
    if not account:
        return go("/login")
    with Session() as db:
        query = select(AuditEvent)
        if not has_role(account, "reviewer"):
            own = select(Application.id).where(Application.owner_email == account["email"])
            query = query.where(or_(AuditEvent.actor == account["email"], AuditEvent.application_id.in_(own)))
        if action:
            query = query.where(AuditEvent.action == action[:40])
        total = db.scalar(select(func.count()).select_from(query.subquery()))
        pages = max(1, (total + 24) // 25)
        page = min(max(1, page), pages)
        events = db.scalars(query.order_by(AuditEvent.id.desc()).offset((page - 1) * 25).limit(25)).all()
        refs = {a.id: a.reference for a in db.scalars(select(Application).where(Application.id.in_([e.application_id for e in events if e.application_id])))} if events else {}
        actions = [r[0] for r in db.execute(select(AuditEvent.action).distinct().order_by(AuditEvent.action)).all()]
        chain = verify_audit_chain(db) if has_role(account, "reviewer") else None
    return render(request, "audit.html", active="audit", events=events, refs=refs, actions=actions, action=action, total=total, page=page, pages=pages, chain=chain)


def csv_safe(value):
    text = str(value if value is not None else "")
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else text


@app.get("/audit.csv")
def export_csv(request: Request, filter: str = "all", q: str = ""):
    account = user(request)
    if not account:
        return go("/login")
    with Session() as db:
        rows = db.scalars(queue_query(request, filter if filter in QUEUE_FILTERS else "all", q).order_by(Application.id)).all()
        append_audit(db, account["email"], "csv_exported", None, {"rows": len(rows), "filter": filter})
        db.commit()
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(["Reference", "Applicant", "Product", "State", "Created UTC", "Requested", "CLTV after", "DTI after", "Policy result",
                     "Model consensus", "Agreement", "Median PD", "Human decision", "Override", "Signoff", "Reason codes", "Scoring hash", "Demo"])
    for a in rows:
        c, d = a.consensus or {}, a.derived or {}
        writer.writerow([a.reference, csv_safe(a.applicant_name), a.product_type, a.state, a.created_at.isoformat() if a.created_at else "",
                         (a.inputs or {}).get("requested_amount", ""), d.get("cltv_after", ""), d.get("dti_after", ""), (a.policy or {}).get("result", ""),
                         c.get("final_recommendation", ""), c.get("agreement", ""), c.get("median_probability", ""), a.human_decision or "",
                         bool(a.override), csv_safe(f"{a.signoff_name or ''} {a.signoff_role or ''}".strip()),
                         " ".join(r["code"] for r in (a.reason_codes or {}).get("principal", [])), a.scoring_hash or "", bool(a.is_demo)])
    return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=equilead-applications.csv"})


@app.get("/guide")
def guide(request: Request):
    if not user(request):
        return go("/login")
    return render(request, "guide.html", active="guide")


# ── JSON API (typed) ─────────────────────────────────────

async def json_body(request):
    try:
        return await request.json()
    except Exception as exc:
        raise HTTPException(422, "Request body must be JSON.") from exc


def api_user(request):
    account = user(request)
    if not account:
        raise HTTPException(403, "Sign in first; the API uses the workspace session.")
    return account


@app.post("/api/calculate")
async def api_calculate(request: Request):
    """Live figures for the form. Returns field errors instead of numbers when the inputs are incomplete."""
    api_user(request)
    check_csrf(request, request.headers.get("x-csrf-token", ""))
    body = await json_body(request)
    try:
        app_input = ApplicationInput(**body)
    except ValidationError as exc:
        return JSONResponse({"ok": False, "errors": validation_errors(exc)})     # 200: an incomplete form is not an error
    derived = derive_metrics(app_input)
    return JSONResponse({"ok": True, "derived": derived.model_dump()})


@app.post("/api/score")
async def api_score(request: Request):
    """Score without persisting. Body: {"application": {...ApplicationInput fields...}}."""
    api_user(request)
    check_csrf(request, request.headers.get("x-csrf-token", ""))
    body = await json_body(request)
    try:
        req = scoring.ScoreRequest(**body)
    except ValidationError as exc:
        return JSONResponse({"ok": False, "errors": validation_errors(exc)}, status_code=422)
    response = await run_in_threadpool(scoring.score_application, req.application)
    return JSONResponse({"ok": True, "result": response.model_dump()})


@app.get("/api/applications/{application_id}")
def api_application(request: Request, application_id: int):
    api_user(request)
    with Session() as db:
        application = get_application(db, request, application_id)
        runs = [r.as_dict() for r in latest_runs(application)]
    payload = application.public_summary()
    payload.update(derived=application.derived, models=runs, explanations=application.explanations)
    return JSONResponse(payload)


@app.get("/api/models")
def api_models(request: Request):
    api_user(request)
    registry = scoring.registry()
    return JSONResponse({"status": scoring.status(), "models": {k: {f: v.get(f) for f in ("name", "version", "sha256", "algorithm", "trained_at", "calibration")}
                                                              for k, v in registry.get("models", {}).items()},
                         "bands": BANDS, "dataset": registry.get("dataset")})


# ── Errors ───────────────────────────────────────────────

@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"ok": False, "error": exc.detail or "error"}, status_code=exc.status_code)
    return error_page(request, exc.status_code, exc.detail or "Nothing was found at that address.")


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"ok": False, "error": "validation", "detail": exc.errors()}, status_code=422)
    return error_page(request, 422, "Part of that address or query was not in the expected format.")


@app.exception_handler(workflow.TransitionError)
async def transition_error(request: Request, exc):
    return error_page(request, 403, str(exc))


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc):
    log.exception("Unhandled error on %s", request.url.path)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"ok": False, "error": "internal"}, status_code=500)
    return error_page(request, 500, "The request could not be completed. No decision or letter was changed.")
