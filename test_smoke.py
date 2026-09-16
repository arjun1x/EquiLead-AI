"""Route, safety and governance tests for the flat handoff. Run: python -m pytest -q test_smoke.py

The suite uses its own SQLite file in the system temp directory, so the demo database next to app.py is
never touched and no folder is created inside the project."""
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import pytest

TMP = pathlib.Path(tempfile.mkdtemp(prefix="equilead-test-"))
os.environ["DEMO_MODE"] = "1"
os.environ["DATABASE_URL"] = f"sqlite:///{(TMP / 'test.sqlite3').as_posix()}"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000"
os.environ.pop("SECRETS_KEY", None)

from fastapi.testclient import TestClient
from sqlalchemy import select, text

import app as app_module
import fairness
import model_card
import models
import monitoring
import pipeline
import reason_codes
import security
from app import app
from models import Session, LoanDecision, AuditEvent, verify_decision_chain, verify_audit_chain

ROOT = pathlib.Path(__file__).resolve().parent
VALID = {"applicant_name": "Smoke Test", "loan": "45000", "mortdue": "185000", "value": "420000", "reason": "HomeImp",
         "job": "ProfExe", "yoj": "6", "derog": "0", "delinq": "0", "clage": "156", "ninq": "1", "clno": "12",
         "debtinc": "28.5", "demo_outcome": "approved"}


# ── helpers ──────────────────────────────────────────────

def token(client, path="/login"):
    return re.search(r'name="csrf_token" value="([^"]+)', client.get(path).text).group(1)


def demo_login(client, role="analyst"):
    response = client.post("/demo", data={"csrf_token": token(client), "role": role}, follow_redirects=False)
    assert response.status_code == 303
    return client


def register(client, email, name="Test User", password="correct-horse-battery"):
    response = client.post("/register", data={"csrf_token": token(client, "/register"), "name": name,
                                              "email": email, "password": password}, follow_redirects=False)
    assert response.status_code == 303, response.text
    return client


def apply(client, **overrides):
    values = {**VALID, **overrides, "csrf_token": token(client, "/apply")}
    return client.post("/apply", data=values, follow_redirects=False)


def count_records():
    with Session() as db:
        return db.scalar(select(models.func.count(LoanDecision.id))) if hasattr(models, "func") else len(db.scalars(select(LoanDecision.id)).all())


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── routes and headers ───────────────────────────────────

def test_public_routes_assets_and_security_headers(client):
    assert client.get("/login").status_code == 200
    assert client.get("/register").status_code == 200
    for name in ("styles.css", "app.js", "scene.js", "three.module.min.js", "logo.svg", "house-fallback.svg"):
        assert client.get(f"/assets/{name}").status_code == 200
    assert client.get("/assets/app.py").status_code == 404
    home = client.get("/", follow_redirects=False)
    assert home.status_code == 303 and home.headers["location"] == "/login"
    assert client.get("/healthz").json()["demo_mode"] is True
    page = client.get("/login")
    assert page.headers["x-frame-options"] == "DENY"
    assert page.headers["x-content-type-options"] == "nosniff"
    assert page.headers["cache-control"] == "no-store"
    assert client.get("/assets/styles.css").headers["cache-control"].startswith("public")


def test_error_pages_are_html_not_json(client):
    missing = client.get("/does-not-exist")
    assert missing.status_code == 404 and "That page isn" in missing.text
    demo_login(client)
    bad_path = client.get("/decisions/abc")
    assert bad_path.status_code == 422 and "didn’t look right" in bad_path.text and "text/html" in bad_path.headers["content-type"]
    bad_query = client.get("/audit?page=abc")
    assert bad_query.status_code == 422 and "text/html" in bad_query.headers["content-type"]


# ── CSRF ─────────────────────────────────────────────────

def test_csrf_protection_on_every_post(client):
    for path, data in (("/demo", {}), ("/login", {"email": "a@b.co", "password": "x"}), ("/logout", {})):
        response = client.post(path, data={**data, "csrf_token": "forged"})
        assert response.status_code == 403 and "fresh start" in response.text, path
    demo_login(client)
    assert client.post("/apply", data={**VALID, "csrf_token": "forged"}).status_code == 403
    assert client.post("/decisions/1/review", data={"csrf_token": "", "human_decision": "approved"}).status_code == 403
    assert client.post("/apply", data=VALID).status_code == 403  # missing token entirely


# ── validation limits ────────────────────────────────────

@pytest.mark.parametrize("field,value,fragment", [
    ("loan", "0", "greater than 0"), ("loan", "-5", "greater than 0"), ("loan", "10000001", "less than or equal"),
    ("loan", "inf", "finite"), ("loan", "nan", "finite"), ("loan", "abc", "valid number"),
    ("value", "0", "greater than 0"), ("mortdue", "-1", "greater than or equal"),
    ("reason", "Nope", "HomeImp"), ("job", "CEO", "Input should be"),
    ("debtinc", "1001", "less than or equal"), ("yoj", "81", "less than or equal"),
    ("derog", "1.5", "valid integer"), ("clage", "1201", "less than or equal"),
    ("applicant_name", "x" * 201, "at most 200"), ("demo_outcome", "chaos", "Input should be"),
])
def test_validation_limits_reject_and_preserve_entries(client, field, value, fragment):
    demo_login(client)
    before = count_records()
    response = apply(client, **{field: value})
    assert response.status_code == 422
    assert fragment in response.text
    assert count_records() == before
    preserved = 'value="45000"' if field == "value" else 'value="420000"'
    assert preserved in response.text  # other entries are preserved


def test_missing_fields_are_reported(client):
    demo_login(client)
    response = client.post("/apply", data={"csrf_token": token(client, "/apply"), "loan": "1"})
    assert response.status_code == 422 and "Field required" in response.text


# ── application flow, demo separation and owner scoping ──

def test_demo_application_and_owner_scoped_archive(client):
    demo_login(client)
    created = apply(client, applicant_name="Scoped Applicant")
    assert created.status_code == 303
    location = created.headers["location"]
    record_id = int(location.rsplit("/", 1)[1])
    page = client.get(location)
    assert page.status_code == 200
    assert "Sample analysis" in page.text and "synthetic fixture (not a probability)" in page.text
    assert "Threshold 50.0%" in page.text and "Training date" in page.text and "synthetic-fixture-v1" in page.text
    assert "Awaiting a reviewer." in page.text and "Record decision" not in page.text  # analysts cannot review
    with Session() as db:
        record = db.get(LoanDecision, record_id)
        assert record.is_demo == 1 and record.record_hash and len(record.record_hash) == 64
        assert record.human_decision == "pending" and record.dataset_version == "synthetic-demo"
    assert client.get("/audit?q=Scoped%20Applicant").text.count("Scoped Applicant") >= 1
    # another analyst cannot see it
    other = TestClient(app)
    register(other, "other-analyst@example.com")
    assert other.get(location).status_code == 404
    assert "Scoped Applicant" not in other.get("/audit?q=Scoped").text
    assert "Scoped Applicant" not in other.get("/audit.csv").text
    assert other.get("/monitoring").status_code == 403
    # a reviewer sees the whole workspace
    reviewer = demo_login(TestClient(app), role="reviewer")
    assert reviewer.get(location).status_code == 200
    assert "Scoped Applicant" in reviewer.get("/audit?q=Scoped").text


def test_all_three_demo_outcomes(client):
    demo_login(client)
    for outcome, heading, warning, codes in (("approved", "Recommend approval.", False, False),
                                             ("denied", "Recommend decline.", False, True),
                                             ("review", "Recommend approval.", True, False)):
        page = client.get(apply(client, demo_outcome=outcome).headers["location"]).text
        assert heading in page
        assert ("Human review required" in page) is warning
        assert ("AA-01" in page) is codes
        if outcome == "denied":
            assert "pending legal and compliance approval" in page
        else:
            assert "No adverse-action reasons apply" in page


def test_research_mode_fails_closed(client, monkeypatch, tmp_path):
    # /demo disappears outside demo mode
    monkeypatch.setattr(app_module, "DEMO_MODE", False)
    assert client.post("/demo", data={"csrf_token": token(client)}).status_code == 404
    monkeypatch.setattr(app_module, "DEMO_MODE", True)
    # pipeline prerequisites, in order: artifacts, model card, API key
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setenv("LABEL_ENCODERS_PATH", str(tmp_path / "missing.pkl"))
    monkeypatch.setenv("MODEL_CARD_PATH", str(tmp_path / "missing_card.json"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="artifacts are missing"):
        pipeline.check_prerequisites()
    (tmp_path / "missing.json").write_text("{}"); (tmp_path / "missing.pkl").write_bytes(b"")
    with pytest.raises(RuntimeError, match="Model card missing"):
        pipeline.check_prerequisites()
    monkeypatch.setenv("MODEL_CARD_PATH", str(ROOT / "model_card.example.json"))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        pipeline.check_prerequisites()
    # the server refuses to start in research mode without a session secret
    env = {**os.environ, "DEMO_MODE": "0", "SESSION_SECRET": "", "DATABASE_URL": f"sqlite:///{(TMP / 'research.sqlite3').as_posix()}",
           "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", "import app"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "SESSION_SECRET" in result.stderr


def test_model_card_validation():
    card = model_card.load_model_card(demo=False, path=ROOT / "model_card.example.json")
    assert card["threshold"] == 0.5 and card["is_demo"] is False
    assert "uncalibrated" in model_card.calibration_label(card, None)
    assert "probability" not in model_card.calibration_label(card, None).replace("not a probability", "")
    assert model_card.calibration_label(card, 0.4).endswith("calibrated probability")
    with pytest.raises(RuntimeError, match="missing"):
        model_card.load_model_card(demo=False, path=ROOT / "nope.json")
    bad = TMP / "bad_card.json"
    bad.write_text(json.dumps({**card, "protected_attributes_in_model": ["sex"]}))
    with pytest.raises(RuntimeError, match="protected attributes"):
        model_card.load_model_card(demo=False, path=bad)
    bad.write_text(json.dumps({"model_version": "x"}))
    with pytest.raises(RuntimeError, match="required fields"):
        model_card.load_model_card(demo=False, path=bad)


# ── archive filters and CSV ──────────────────────────────

def test_archive_filters_and_pagination(client):
    fresh = TestClient(app)
    register(fresh, "archive-user@example.com")
    for i in range(12):
        assert apply(fresh, applicant_name=f"Paged {i:02d}", demo_outcome="denied" if i % 4 == 0 else "approved").status_code == 303
    first = fresh.get("/audit")
    assert 'count-pill">12' in first.text and "Page 1 of 2" in first.text and "Next →" in first.text
    second = fresh.get("/audit?page=2")
    assert "Page 2 of 2" in second.text and "← Previous" in second.text
    assert fresh.get("/audit?page=99").text.count("Page 2 of 2") == 1
    assert 'count-pill">3' in fresh.get("/audit?status=denied").text
    assert 'count-pill">9' in fresh.get("/audit?status=approved").text
    assert 'count-pill">0' in fresh.get("/audit?status=review").text
    assert 'count-pill">12' in fresh.get("/audit?status=awaiting").text
    assert 'count-pill">1' in fresh.get("/audit?q=paged%2007").text  # case-insensitive
    assert 'count-pill">0' in fresh.get("/audit?q=%25").text  # wildcard escaped
    assert "No matching applications" in fresh.get("/audit?q=zzz").text
    demo = demo_login(TestClient(app))
    review = demo.get("/audit?status=review").text
    assert int(re.search(r'count-pill">(\d+)', review).group(1)) >= 2  # two seeded escalations, plus any created by other tests
    assert "Jamie Rivera" in review and "Riley Morgan" in review and "Taylor Brooks" not in review


def test_csv_export_escapes_formulas_and_carries_metadata(client):
    fresh = TestClient(app)
    register(fresh, "csv-user@example.com")
    for name in ("=HACK(1)", "+SUM(A1)", "-2+3", "@cmd", "\tTab", "Plain Name"):
        assert apply(fresh, applicant_name=name).status_code == 303
    response = fresh.get("/audit.csv")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/csv")
    header, *rows = response.text.strip().splitlines()
    assert header.startswith("ID,Applicant,Loan amount,Recommendation,Default score,Calibration,Threshold")
    assert "Model version" in header and "Record hash" in header
    names = [row.split(",")[1] for row in rows]
    for dangerous in ("'=HACK(1)", "'+SUM(A1)", "'-2+3", "'@cmd"):
        assert dangerous in names
    assert "Plain Name" in names and not any(n.startswith("=") or n.startswith("+") for n in names)
    assert all(row.split(",")[1] != "\tTab" for row in rows)
    assert "synthetic-fixture-v1" in response.text and "synthetic-demo" in response.text
    with Session() as db:
        assert db.scalar(select(AuditEvent).where(AuditEvent.action == "csv_exported", AuditEvent.actor == "csv-user@example.com"))


# ── rate limiting ────────────────────────────────────────

def test_rate_limit_blocks_post_bursts(client):
    security.limiter.reset()
    original = security.limiter.limit
    security.limiter.limit = 2
    try:
        data = {"csrf_token": token(client), "email": "nobody@example.com", "password": "wrong-password"}
        assert client.post("/login", data=data).status_code == 422
        assert client.post("/login", data=data).status_code == 422
        blocked = client.post("/login", data=data)
        assert blocked.status_code == 429 and "short pause" in blocked.text
        assert client.get("/login").status_code == 200  # reads are not limited
    finally:
        security.limiter.limit = original
        security.limiter.reset()


# ── roles, human review and audit chain ──────────────────

def test_roles_human_review_and_audit_events(client):
    demo_login(client)
    location = apply(client, applicant_name="Review Me").headers["location"]
    record_id = int(location.rsplit("/", 1)[1])
    forbidden = client.post(f"/decisions/{record_id}/review", data={"csrf_token": token(client, location), "human_decision": "approved"})
    assert forbidden.status_code == 403 and "Only a reviewer" in forbidden.text
    reviewer = demo_login(TestClient(app), role="reviewer")
    page = reviewer.get(location)
    assert "Record decision" in page.text and "Monitoring" in page.text
    bad = reviewer.post(f"/decisions/{record_id}/review", data={"csrf_token": token(reviewer, location), "human_decision": "maybe"})
    assert bad.status_code == 422
    ok = reviewer.post(f"/decisions/{record_id}/review", data={"csrf_token": token(reviewer, location), "human_decision": "declined",
                                                               "note": "Income documents did not match.", "outcome": "0"}, follow_redirects=False)
    assert ok.status_code == 303
    page = reviewer.get(location).text
    assert "Recorded by a reviewer." in page and "Declined" in page and "Income documents did not match." in page and "Repaid" in page
    assert "Recommend approval." in page  # model recommendation is untouched and shown separately
    with Session() as db:
        record = db.get(LoanDecision, record_id)
        assert record.human_decision == "declined" and record.outcome == 0 and record.reviewed_by == "reviewer@equilead.local"
        event = db.scalar(select(AuditEvent).where(AuditEvent.action == "human_review_recorded", AuditEvent.record_id == record_id))
        assert event and json.loads(event.detail)["human_decision"] == "declined"
        assert "Income documents" not in (event.detail or "")  # the note itself is hashed, not copied
        assert verify_decision_chain(db)["ok"]  # human review does not break the sealed model output
    assert reviewer.get("/monitoring").status_code == 200
    assert "Synthetic figures" in reviewer.get("/monitoring?days=30").text


def test_hash_chains_detect_tampering(client):
    demo_login(client)
    apply(client, applicant_name="Chain One"); apply(client, applicant_name="Chain Two")
    with Session() as db:
        before = verify_decision_chain(db)
        assert before["ok"] and before["checked"] >= 2 and before["unsealed"] == 0
        assert verify_audit_chain(db)["ok"]
        victim = db.scalar(select(LoanDecision).where(LoanDecision.applicant_name == "Chain One"))
        original = victim.decision
        db.execute(text("UPDATE loan_decisions SET decision='denied' WHERE id=:id"), {"id": victim.id}); db.commit()
        db.expire_all()
        tampered = verify_decision_chain(db)
        assert not tampered["ok"] and tampered["first_bad_id"] == victim.id
        db.execute(text("UPDATE loan_decisions SET decision=:d WHERE id=:id"), {"d": original, "id": victim.id}); db.commit()
        db.expire_all()
        assert verify_decision_chain(db)["ok"]
        event = db.scalar(select(AuditEvent).order_by(AuditEvent.id.desc()).limit(1))
        event_id, original_actor = event.id, event.actor
        db.execute(text("UPDATE audit_events SET actor='mallory' WHERE id=:id"), {"id": event_id}); db.commit()
        db.expire_all()
        assert verify_audit_chain(db)["first_bad_id"] == event_id
        db.execute(text("UPDATE audit_events SET actor=:a WHERE id=:id"), {"a": original_actor, "id": event_id}); db.commit()
        db.expire_all()
        assert verify_audit_chain(db)["ok"]


def test_retention_purge_keeps_chain_verifiable(client):
    demo_login(client)
    location = apply(client, applicant_name="Old Record").headers["location"]
    record_id = int(location.rsplit("/", 1)[1])
    with Session() as db:
        db.execute(text("UPDATE loan_decisions SET created_at=:t WHERE id=:id"),
                   {"t": (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=400)).replace(tzinfo=None), "id": record_id}); db.commit()
    apply(client, applicant_name="Newer Record")  # chains onto the old record's hash
    assert security.purge_expired(0) == 0
    deleted = security.purge_expired(365, actor="test")
    assert deleted >= 1
    with Session() as db:
        assert db.get(LoanDecision, record_id) is None
        event = db.scalar(select(AuditEvent).where(AuditEvent.action == "retention_purge").order_by(AuditEvent.id.desc()))
        detail = json.loads(event.detail)
        assert record_id in detail["ids"] and len(detail["hashes"]) == len(detail["ids"])
        assert verify_decision_chain(db)["ok"] and verify_audit_chain(db)["ok"]


# ── secrets, reason codes, fairness, monitoring ──────────

def test_encrypted_secrets_roundtrip():
    pytest.importorskip("cryptography")
    key = security.keygen()
    path = TMP / "secrets.enc"
    security.write_secrets(key, path, {"ANTHROPIC_API_KEY": "sk-test", "SESSION_SECRET": "abc"})
    env = {"SECRETS_KEY": key, "SECRETS_FILE": str(path), "SESSION_SECRET": "already-set"}
    assert security.load_secrets(env) == {"ANTHROPIC_API_KEY": "sk-test", "SESSION_SECRET": "abc"}
    assert env["ANTHROPIC_API_KEY"] == "sk-test" and env["SESSION_SECRET"] == "already-set"
    with pytest.raises(RuntimeError, match="not found"):
        security.load_secrets({"SECRETS_KEY": key, "SECRETS_FILE": str(TMP / "missing.enc")})
    with pytest.raises(Exception):
        security.read_secrets(security.keygen(), path)  # wrong key never yields plaintext
    assert security.load_secrets({}) == {}


def test_adverse_action_reason_codes():
    shap = [{"feature": "DEBTINC", "shap_value": 0.9}, {"feature": "JOB", "shap_value": 0.6},
            {"feature": "CLAGE", "shap_value": -0.5}, {"feature": "LOAN", "shap_value": 0.3},
            {"feature": "VALUE", "shap_value": 0.2}, {"feature": "NINQ", "shap_value": 0.1},
            {"feature": "DELINQ", "shap_value": 0.05}, {"feature": "YOJ", "shap_value": 0.01}]
    assert reason_codes.adverse_action_reasons(shap, "approved") == []
    reasons = reason_codes.adverse_action_reasons(shap, "denied")
    coded = [r["code"] for r in reasons if r["code"]]
    assert coded == ["AA-01", "AA-08", "AA-05", "AA-02"]          # ordered by impact, collateral deduplicated, max four
    assert "CLAGE" not in [r["feature"] for r in reasons]          # protective factors are never reasons
    job = next(r for r in reasons if r["feature"] == "JOB")
    assert job["code"] is None and job["review_required"] and "compliance" in job["statement"]
    assert reason_codes.notice_statements(reasons) == ["Excessive obligations in relation to income",
        "Value or type of collateral not sufficient for amount requested", "Number of recent inquiries on credit report",
        "Delinquent past or present credit obligations with others"]
    assert reason_codes.approval_status(model_card.DEMO_CARD)["approved"] is False


def test_fairness_and_monitoring_maths():
    rows = ([{"job": "A", "decision": "approved", "outcome": 0}] * 40 + [{"job": "A", "decision": "denied", "outcome": 1}] * 10
            + [{"job": "B", "decision": "approved", "outcome": 0}] * 20 + [{"job": "B", "decision": "denied", "outcome": 0}] * 30
            + [{"job": "C", "decision": "approved"}] * 5)
    rates = fairness.group_rates(rows, "job")
    assert rates["A"]["approval_rate"] == 0.8 and rates["B"]["approval_rate"] == 0.4 and rates["A"]["fpr"] == 0.0 and rates["B"]["fpr"] == 0.6
    ff = fairness.four_fifths(rates)
    assert ff["status"] == "REVIEW" and ff["reference"] == "A" and abs(ff["min_ratio"] - 0.5) < 1e-9 and "C" not in ff["ratios"]
    assert fairness.four_fifths({"A": rates["A"]})["status"] == "INSUFFICIENT"
    assert fairness.error_rate_gaps(rates)["fpr"]["status"] == "REVIEW"
    balanced = fairness.group_rates([{"job": "A", "decision": "approved"}] * 30 + [{"job": "B", "decision": "approved"}] * 30, "job")
    assert fairness.four_fifths(balanced)["status"] == "PASS"
    assert monitoring.psi([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert monitoring.psi([0.9, 0.1], [0.1, 0.9]) > 0.25
    assert monitoring.bin_proportions([1, 5, 50], [0, 10, 100]) == [2 / 3, 1 / 3]
    assert monitoring.bin_proportions([], [0, 1]) is None
    card = model_card.DEMO_CARD
    few = monitoring.drift_report(card, [{"debtinc": 30, "score": 0.1}] * 5)
    assert not few["enough_rows"] and all(item["psi"] is None for item in few["items"]) and few["status"] == "muted"
    many = monitoring.drift_report(card, [{"debtinc": 30.0, "loan_amount": 25000, "reason": "HomeImp", "job": "Other", "score": 0.1}] * 40)
    assert many["enough_rows"] and any(item["psi"] is not None for item in many["items"])
    report = monitoring.build_report(monitoring.rows_from_db(), card, 90)
    assert report["synthetic"] is True and report["n"] >= 8 and "drift" in report and "job" in report["subgroups"]
    assert fairness.fair_lending_report(monitoring.rows_from_db())["governance"]["model_inputs"]
