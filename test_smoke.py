"""Route, persistence, security and governance tests. Run: python -m pytest -q -p no:cacheprovider test_smoke.py

conftest.py points DATABASE_URL at a temp file, so the demo database next to app.py is never touched."""
import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from conftest import form_data

pytestmark = pytest.mark.usefixtures("client")


@pytest.fixture(scope="module")
def client():
    from app import app
    with TestClient(app, follow_redirects=False) as c:      # lifespan: init_db, load models, seed demo
        yield c


@pytest.fixture(scope="module")
def reviewer(client):
    demo(client, "reviewer")
    return client


@pytest.fixture(scope="module")
def analyst():
    from app import app
    c = TestClient(app, follow_redirects=False)
    demo(c, "analyst")
    return c


def csrf(c, path="/"):
    html = c.get(path).text
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def demo(c, role):
    r = c.post("/demo", data={"csrf_token": csrf(c, "/login"), "role": role})
    assert r.status_code == 303 and r.headers["location"] == "/"


def h1(html):
    m = re.search(r"<h1>([^<]+)</h1>", html)
    return m.group(1) if m else ""


def flash(html):
    m = re.search(r'role="status">.*?<p>([^<]+)</p>', html, re.S)
    return m.group(1) if m else ""


def state_of(html):
    return re.search(r"badge state-(\w+)", html).group(1)


def create(c, **overrides):
    r = c.post("/applications/new", data={"csrf_token": csrf(c, "/applications/new"), "action": "submit", **form_data(**overrides)})
    assert r.status_code == 303, r.text[:300]
    return int(re.search(r"/applications/(\d+)", r.headers["location"]).group(1))


# ── Access and safety ────────────────────────────────────

def test_anonymous_is_redirected_and_health_reports_models(client):
    assert client.get("/").status_code == 303 and client.get("/applications/1").status_code == 303
    health = client.get("/healthz").json()
    assert health["status"] == "ok" and health["demo_mode"] is True and set(health["models"]["loaded"]) == {"scorecard", "forest", "gbm"}


def test_csrf_is_enforced_on_forms_and_json(client):
    assert client.post("/demo", data={"role": "reviewer"}).status_code == 403
    assert client.post("/demo", data={"role": "reviewer", "csrf_token": "wrong"}).status_code == 403
    assert client.post("/api/calculate", json=form_data()).status_code == 403


def test_assets_are_allow_listed_and_headers_are_set(client):
    assert client.get("/assets/styles.css").status_code == 200
    assert client.get("/assets/app.py").status_code == 404
    assert client.get("/assets/model_registry.json").status_code == 404
    assert client.get("/assets/..%2Fapp.py").status_code == 404
    r = client.get("/login")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"


def test_error_pages_are_html_not_json(reviewer):
    r = reviewer.get("/nope"); assert r.status_code == 404 and "<h1>" in r.text
    r = reviewer.get("/applications/abc"); assert r.status_code == 422 and "<h1>" in r.text
    r = reviewer.get("/api/applications/999999"); assert r.status_code == 404 and r.json()["ok"] is False


def test_rate_limiter_counts_per_key():
    from security import RateLimiter
    limiter = RateLimiter(2)
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a") and limiter.allow("b")


# ── Demo seed and dashboard ──────────────────────────────

def test_demo_seed_is_labelled_synthetic(reviewer):
    from models import Application, Session
    with Session() as db:
        seeded = db.scalars(select(Application).where(Application.is_demo.is_(True))).all()
    assert len(seeded) >= 8
    assert "Demonstration" in reviewer.get("/").text and "synthetic" in reviewer.get("/").text.lower()


def test_pages_render_for_reviewer(reviewer):
    for path in ("/", "/applications", "/applications?filter=needs_decision&q=EQ", "/applications/new", "/fairness", "/models", "/monitoring", "/audit", "/guide"):
        r = reviewer.get(path)
        assert r.status_code == 200, path
    assert reviewer.get("/audit.csv").headers["content-type"].startswith("text/csv")


# ── Application lifecycle ────────────────────────────────

def test_validation_errors_rerender_with_values_preserved(reviewer):
    r = reviewer.post("/applications/new", data={"csrf_token": csrf(reviewer, "/applications/new"), "action": "submit", **form_data(credit_score="200", applicant_name="Keep Me")})
    assert r.status_code == 422 and "needs attention" in r.text and 'value="Keep Me"' in r.text and "Credit score" in r.text
    r = reviewer.post("/applications/new", data={"csrf_token": csrf(reviewer, "/applications/new"), "action": "submit", **form_data(applicant_name="")})
    assert r.status_code == 422 and "Applicant name" in r.text


def test_submit_scores_with_three_models_and_persists(reviewer):
    app_id = create(reviewer)
    html = reviewer.get(f"/applications/{app_id}").text
    assert state_of(html) in ("scored", "manual_review")
    assert html.count("Policy scorecard") >= 1 and "Random forest" in html and "Gradient boosting" in html
    assert "Consensus" in html and "Policy rules" in html and "Reason codes" in html
    from models import Application, AuditEvent, Session
    with Session() as db:
        a = db.get(Application, app_id)
        assert not a.is_demo and len(a.model_runs) == 3 and a.consensus["final_recommendation"] and a.scoring_hash
        actions = [e.action for e in db.scalars(select(AuditEvent).where(AuditEvent.application_id == app_id).order_by(AuditEvent.id))]
    assert actions[:3] == ["created", "submitted", "scored"]
    api = reviewer.get(f"/api/applications/{app_id}").json()
    assert api["reference"].startswith("EQ-") and len(api["models"]) == 3


def test_draft_is_saved_without_scoring_and_can_be_withdrawn(reviewer):
    r = reviewer.post("/applications/new", data={"csrf_token": csrf(reviewer, "/applications/new"), "action": "save_draft", "applicant_name": "Draft Only", "annual_income": "50000"})
    assert r.status_code == 303 and r.headers["location"].endswith("/edit")
    app_id = int(re.search(r"/applications/(\d+)", r.headers["location"]).group(1))
    assert 'value="50000"' in reviewer.get(f"/applications/{app_id}/edit").text
    assert state_of(reviewer.get(f"/applications/{app_id}").text) == "draft"
    reviewer.post(f"/applications/{app_id}/withdraw", data={"csrf_token": csrf(reviewer), "reason": "Applicant withdrew"})
    assert state_of(reviewer.get(f"/applications/{app_id}").text) == "withdrawn"


def test_owner_scoping(reviewer, analyst):
    mine = create(analyst, applicant_name="Analyst Case")
    theirs = create(reviewer, applicant_name="Reviewer Case")
    assert analyst.get(f"/applications/{mine}").status_code == 200
    assert analyst.get(f"/applications/{theirs}").status_code == 404
    assert analyst.get(f"/api/applications/{theirs}").status_code == 404
    assert reviewer.get(f"/applications/{mine}").status_code == 200
    assert "Reviewer Case" not in analyst.get("/applications?filter=all").text
    assert analyst.get("/monitoring").status_code == 403


def test_analyst_cannot_decide(analyst):
    app_id = create(analyst)
    r = analyst.post(f"/applications/{app_id}/decision", data={"csrf_token": csrf(analyst), "human_decision": "approved", "signoff_name": "A", "signoff_role": "B"})
    assert r.status_code == 403


def test_override_requires_written_explanation_and_is_audited(reviewer):
    app_id = create(reviewer, credit_score=650, delinquencies_24m=1, inquiries_6m=4, monthly_debt=3200, cash_reserves=4000, documents=["identity"])
    html = reviewer.get(f"/applications/{app_id}").text
    recommended = re.search(r'data-recommended="(\w*)"', html).group(1)
    assert recommended and recommended != "approved"
    token = csrf(reviewer, f"/applications/{app_id}")
    r = reviewer.post(f"/applications/{app_id}/decision", data={"csrf_token": token, "human_decision": "approved", "signoff_name": "Sam", "signoff_role": "Loan officer"})
    assert r.status_code == 303 and "Explain why" in flash(reviewer.get(f"/applications/{app_id}").text)
    assert state_of(reviewer.get(f"/applications/{app_id}").text) in ("scored", "manual_review")
    r = reviewer.post(f"/applications/{app_id}/decision", data={"csrf_token": token, "human_decision": "approved", "signoff_name": "Sam", "signoff_role": "Loan officer",
                                                               "override_reason": "Employer confirmed a documented raise and the delinquency was a disputed medical bill.", "approved_amount": "80000"})
    html = reviewer.get(f"/applications/{app_id}").text
    assert state_of(html) == "approved" and "differs from recommendation" in html
    from models import Application, AuditEvent, Session
    with Session() as db:
        a = db.get(Application, app_id)
        assert a.override is True and a.override_reason.startswith("Employer") and a.approved_amount == 80000 and a.signoff_name == "Sam"
        event = db.scalars(select(AuditEvent).where(AuditEvent.application_id == app_id, AuditEvent.action == "decision_recorded")).one()
        detail = event.detail_dict()
    assert detail["previous"]["human_decision"] is None and detail["updated"]["human_decision"] == "approved"
    assert detail["previous"]["state"] in ("scored", "manual_review") and detail["updated"]["state"] == "approved" and detail["override"] is True
    history = reviewer.get(f"/applications/{app_id}/history").text
    assert "Decision recorded" in history and "override reason" in history


def test_cannot_finalize_without_approved_letter_then_full_path(reviewer):
    app_id = create(reviewer)
    token = csrf(reviewer, f"/applications/{app_id}")
    html = reviewer.get(f"/applications/{app_id}").text
    recommended = re.search(r'data-recommended="(\w*)"', html).group(1)
    decision = "approved" if recommended in ("approved", "manual_review") else "conditionally_approved"
    data = {"csrf_token": token, "human_decision": decision, "signoff_name": "Sam", "signoff_role": "Loan officer", "approved_amount": "90000",
            "override_reason": "Resolved after reviewing the complete file with the applicant."}
    reviewer.post(f"/applications/{app_id}/decision", data=data)
    assert state_of(reviewer.get(f"/applications/{app_id}").text) == decision
    reviewer.post(f"/applications/{app_id}/finalize", data={"csrf_token": token})
    page = reviewer.get(f"/applications/{app_id}").text
    assert "Cannot finalize" in flash(page) and state_of(page) == decision
    # wrong letter type for the decision is refused
    assert reviewer.post(f"/applications/{app_id}/letter", data={"csrf_token": token, "letter_type": "adverse_action"}).status_code == 422
    letter_type = "approval" if decision == "approved" else "conditional_approval"
    reviewer.post(f"/applications/{app_id}/letter", data={"csrf_token": token, "letter_type": letter_type})
    page = reviewer.get(f"/applications/{app_id}/letter").text
    assert "Original draft" in page and "Corrected draft" in page and "No issues detected" in page
    letter_id = int(re.search(r"/letter/(\d+)/approve", page).group(1))
    reviewer.post(f"/applications/{app_id}/letter/{letter_id}/approve", data={"csrf_token": token})
    reviewer.post(f"/applications/{app_id}/finalize", data={"csrf_token": token})
    assert state_of(reviewer.get(f"/applications/{app_id}").text) == "finalized"
    assert reviewer.post(f"/applications/{app_id}/decision", data=data).status_code == 403
    assert reviewer.post(f"/applications/{app_id}/withdraw", data={"csrf_token": token, "reason": "x"}).status_code == 403
    assert reviewer.get(f"/applications/{app_id}/edit").status_code == 303
    from models import Application, Session
    with Session() as db:
        a = db.get(Application, app_id)
        assert a.state == "finalized" and a.finalized_at and a.letter_approved_at and a.current_letter().status == "approved"


def test_declined_case_letter_lists_stored_reasons_only(reviewer):
    app_id = create(reviewer, credit_score=600, delinquencies_24m=3, monthly_debt=4200, property_value=320000, mortgage_balance=250000, requested_amount=40000, cash_reserves=200, documents=["identity"])
    token = csrf(reviewer, f"/applications/{app_id}")
    reviewer.post(f"/applications/{app_id}/decision", data={"csrf_token": token, "human_decision": "declined", "signoff_name": "Sam", "signoff_role": "Loan officer",
                                                           "override_reason": "Decline confirmed after reviewing the full credit file."})
    reviewer.post(f"/applications/{app_id}/letter", data={"csrf_token": token, "letter_type": "adverse_action"})
    page = reviewer.get(f"/applications/{app_id}/letter").text
    from models import Application, Session
    from reason_codes import customer_statements
    with Session() as db:
        a = db.get(Application, app_id)
        reasons = customer_statements(a.reason_codes)
        letter = a.current_letter()
    assert reasons and all(r in letter.corrected_text for r in reasons)
    assert "xgboost" not in letter.corrected_text.lower() and "shap" not in letter.corrected_text.lower()
    assert letter.decision_snapshot["reasons"] == reasons and "No issues detected" in page


def test_rescore_keeps_previous_runs_in_history(reviewer):
    app_id = create(reviewer)
    reviewer.post(f"/applications/{app_id}/score", data={"csrf_token": csrf(reviewer)})
    from models import Application, Session
    with Session() as db:
        a = db.get(Application, app_id)
        assert len(a.model_runs) == 6 and {r.batch for r in a.model_runs} == {1, 2}
    assert "Model run 2" in reviewer.get(f"/applications/{app_id}/history").text


def test_edit_after_scoring_supersedes_and_audits_previous_values(reviewer):
    app_id = create(reviewer)
    r = reviewer.post(f"/applications/{app_id}/edit", data={"csrf_token": csrf(reviewer), "action": "submit", **form_data(requested_amount="120000")})
    assert r.status_code == 303
    from models import Application, AuditEvent, Session
    with Session() as db:
        a = db.get(Application, app_id)
        assert a.inputs["requested_amount"] == 120000
        event = db.scalars(select(AuditEvent).where(AuditEvent.application_id == app_id, AuditEvent.action == "inputs_updated")).one()
    detail = event.detail_dict()
    assert detail["previous"]["inputs"]["requested_amount"] == 90000 and detail["updated"]["inputs"]["requested_amount"] == 120000


# ── Audit chain, export, API ─────────────────────────────

def test_audit_chain_verifies_and_detects_tampering(reviewer):
    from models import AuditEvent, Session, verify_audit_chain
    with Session() as db:
        assert verify_audit_chain(db)["ok"]
        event = db.scalars(select(AuditEvent).where(AuditEvent.action == "scored").order_by(AuditEvent.id)).first()
        original = event.detail
        event.detail = original.replace("scored", "tampered", 1) if "scored" in original else original + " "
        db.commit()
        result = verify_audit_chain(db)
        event.detail = original
        db.commit()
        assert not result["ok"] and result["first_bad_id"] == event.id
        assert verify_audit_chain(db)["ok"]


def test_csv_export_neutralises_spreadsheet_formulas(reviewer):
    create(reviewer, applicant_name="=HYPERLINK(\"x\")")
    body = reviewer.get("/audit.csv?filter=all").text
    assert "'=HYPERLINK" in body and "\n=HYPERLINK" not in body
    assert "Scoring hash" in body.splitlines()[0]


def test_json_score_api_is_typed_and_does_not_persist(reviewer):
    token = csrf(reviewer)
    r = reviewer.post("/api/score", json={"application": form_data()}, headers={"X-CSRF-Token": token})
    assert r.status_code == 200
    result = r.json()["result"]
    assert len(result["models"]) == 3 and result["consensus"]["final_recommendation"] and result["derived"]["cltv_after"]
    r = reviewer.post("/api/score", json={"application": {"applicant_name": "x"}}, headers={"X-CSRF-Token": token})
    assert r.status_code == 422 and "application.credit_score" in r.json()["errors"]
    r = reviewer.post("/api/calculate", json={"applicant_name": "x"}, headers={"X-CSRF-Token": token})
    assert r.status_code == 200 and r.json()["ok"] is False and "credit_score" in r.json()["errors"]
    assert reviewer.get("/api/models").json()["models"]["scorecard"]["sha256"]


def test_logout_clears_session(client):
    from app import app
    c = TestClient(app, follow_redirects=False)
    demo(c, "analyst")
    assert c.get("/").status_code == 200
    c.post("/logout", data={"csrf_token": csrf(c)})
    assert c.get("/").status_code == 303
