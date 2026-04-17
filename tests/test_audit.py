"""Tests for audit log database models."""

import json
import tempfile
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.audit.models import Base, LoanDecision


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_create_table():
    session = _make_session()
    records = session.query(LoanDecision).all()
    assert records == []


def test_insert_and_query():
    session = _make_session()
    record = LoanDecision(
        applicant_name="Test User",
        loan_amount=25000,
        decision="approved",
        default_probability=0.15,
        confidence=85.0,
        bias_score=0.05,
        bias_action="send",
        bias_rewrites=0,
    )
    record.set_shap([{"feature": "LOAN", "shap_value": -0.5, "value": 25000}])
    session.add(record)
    session.commit()

    fetched = session.query(LoanDecision).first()
    assert fetched.applicant_name == "Test User"
    assert fetched.decision == "approved"
    assert fetched.loan_amount == 25000


def test_shap_serialization():
    session = _make_session()
    reasons = [
        {"feature": "DEBTINC", "shap_value": 1.5, "value": 45.0},
        {"feature": "LOAN", "shap_value": -0.3, "value": 10000},
    ]
    record = LoanDecision(
        loan_amount=10000, decision="denied",
        default_probability=0.8, confidence=80.0,
        bias_score=0.0, bias_action="send", bias_rewrites=0,
    )
    record.set_shap(reasons)
    session.add(record)
    session.commit()

    fetched = session.query(LoanDecision).first()
    parsed = fetched.get_shap()
    assert len(parsed) == 2
    assert parsed[0]["feature"] == "DEBTINC"


def test_bias_trail_serialization():
    session = _make_session()
    trail = [{"attempt": 0, "bias_score": 0.1, "summary": "Clean"}]
    record = LoanDecision(
        loan_amount=5000, decision="approved",
        default_probability=0.1, confidence=90.0,
        bias_score=0.1, bias_action="send", bias_rewrites=0,
    )
    record.set_bias_trail(trail)
    session.add(record)
    session.commit()

    fetched = session.query(LoanDecision).first()
    assert json.loads(fetched.bias_audit_trail)[0]["bias_score"] == 0.1


def test_multiple_records():
    session = _make_session()
    for i in range(5):
        record = LoanDecision(
            applicant_name=f"User {i}",
            loan_amount=10000 + i * 5000,
            decision="approved" if i % 2 == 0 else "denied",
            default_probability=0.1 * i,
            confidence=90 - i * 5,
            bias_score=0.0, bias_action="send", bias_rewrites=0,
        )
        session.add(record)
    session.commit()

    all_records = session.query(LoanDecision).all()
    assert len(all_records) == 5
    denied = session.query(LoanDecision).filter_by(decision="denied").all()
    assert len(denied) == 2
