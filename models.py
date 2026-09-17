"""Storage for EquiLead: applications, model runs, letters, users and an append-only hash-chained audit timeline.

Nothing about a decision is ever silently overwritten: every change to an application is written as an
audit event carrying the previous and updated values, and audit events chain to each other by hash.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

ROOT = Path(__file__).resolve().parent
DEMO_MODE = os.getenv("DEMO_MODE", "1") == "1"
DB_PATH = ROOT / ("equilead-demo.sqlite3" if DEMO_MODE else "equilead.sqlite3")
DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite:") else {})
Session = sessionmaker(bind=engine, expire_on_commit=False)

ROLES = {"analyst": 1, "reviewer": 2, "admin": 3}
GENESIS = "0" * 64


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def chain_hash(prev_hash, payload):
    return hashlib.sha256(((prev_hash or GENESIS) + "\n" + canonical(payload)).encode()).hexdigest()


def _iso(value):
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return value.isoformat()


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    email = Column(String(254), primary_key=True)
    name = Column(String(100), nullable=False)
    password_hash = Column(String(128), nullable=False)
    salt = Column(String(64), nullable=False)
    role = Column(String(20), default="analyst")
    created_at = Column(DateTime, default=utcnow)


class Application(Base):
    __tablename__ = "applications"
    id = Column(Integer, primary_key=True)
    reference = Column(String(24), unique=True, index=True)
    owner_email = Column(String(254), nullable=False, index=True)
    state = Column(String(30), nullable=False, default="draft", index=True)
    created_at = Column(DateTime, default=utcnow, index=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    applicant_name = Column(String(120), default="")
    product_type = Column(String(30), default="heloc")
    inputs = Column(JSON, default=dict)           # raw form values (draft may be partial)
    derived = Column(JSON, default=dict)          # DerivedMetrics
    validation = Column(JSON, default=dict)       # errors / flags at last submit
    scored_at = Column(DateTime)
    scoring_hash = Column(String(64))
    consensus = Column(JSON, default=dict)
    policy = Column(JSON, default=dict)
    reason_codes = Column(JSON, default=dict)
    explanations = Column(JSON, default=dict)     # per model drivers + counterfactuals
    human_decision = Column(String(30))
    decision_note = Column(Text)
    approved_amount = Column(Float)
    conditions = Column(JSON, default=list)
    override = Column(Boolean, default=False)
    override_reason = Column(Text)
    decided_by = Column(String(254))
    decided_at = Column(DateTime)
    signoff_name = Column(String(120))
    signoff_role = Column(String(80))
    letter_approved_at = Column(DateTime)
    finalized_at = Column(DateTime)
    withdrawn_reason = Column(Text)
    is_demo = Column(Integer, default=0)
    model_runs = relationship("ModelRun", back_populates="application", order_by="ModelRun.id")
    letters = relationship("Letter", back_populates="application", order_by="Letter.id")

    def latest_runs(self):
        """Most recent run for each model key."""
        latest = {}
        for run in self.model_runs:
            latest[run.model_key] = run
        return latest

    def current_letter(self):
        return self.letters[-1] if self.letters else None

    def public_summary(self):
        return {"id": self.id, "reference": self.reference, "state": self.state, "applicant_name": self.applicant_name,
                "product_type": self.product_type, "created_at": _iso(self.created_at), "scored_at": _iso(self.scored_at),
                "consensus": self.consensus, "policy": self.policy, "reason_codes": self.reason_codes,
                "human_decision": self.human_decision, "override": self.override, "is_demo": self.is_demo}


class ModelRun(Base):
    __tablename__ = "model_runs"
    id = Column(Integer, primary_key=True)
    application_id = Column(Integer, ForeignKey("applications.id"), nullable=False, index=True)
    batch = Column(Integer, default=1)
    created_at = Column(DateTime, default=utcnow)
    model_key = Column(String(30), nullable=False)
    model_name = Column(String(80))
    model_version = Column(String(60))
    model_hash = Column(String(64))
    preprocessing_version = Column(String(30))
    probability = Column(Float)
    raw_probability = Column(Float)
    risk_band = Column(String(2))
    recommendation = Column(String(30))
    decision_threshold = Column(Float)
    band_cutoffs = Column(JSON, default=dict)
    drivers = Column(JSON, default=dict)
    explainer_method = Column(String(80))
    processing_ms = Column(Float)
    scoring_ms = Column(Float)
    in_range = Column(Boolean)
    out_of_range = Column(JSON, default=list)
    error = Column(Text)
    application = relationship("Application", back_populates="model_runs")

    def as_dict(self):
        return {"model_key": self.model_key, "model_name": self.model_name, "model_version": self.model_version,
                "model_hash": self.model_hash, "preprocessing_version": self.preprocessing_version, "probability": self.probability,
                "raw_probability": self.raw_probability, "risk_band": self.risk_band, "recommendation": self.recommendation,
                "decision_threshold": self.decision_threshold, "band_cutoffs": self.band_cutoffs, "drivers": self.drivers,
                "explainer_method": self.explainer_method, "processing_ms": self.processing_ms, "scoring_ms": self.scoring_ms,
                "in_range": self.in_range, "out_of_range": self.out_of_range, "error": self.error, "batch": self.batch}


class Letter(Base):
    __tablename__ = "letters"
    id = Column(Integer, primary_key=True)
    application_id = Column(Integer, ForeignKey("applications.id"), nullable=False, index=True)
    version = Column(Integer, default=1)
    template_version = Column(String(30))
    letter_type = Column(String(30))
    draft_text = Column(Text)
    issues = Column(JSON, default=list)
    corrected_text = Column(Text)
    decision_snapshot = Column(JSON, default=dict)
    status = Column(String(20), default="draft")   # draft | approved | superseded
    created_by = Column(String(254))
    created_at = Column(DateTime, default=utcnow)
    approved_by = Column(String(254))
    approved_at = Column(DateTime)
    application = relationship("Application", back_populates="letters")


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=utcnow, index=True)
    actor = Column(String(254))
    action = Column(String(40), index=True)
    application_id = Column(Integer, index=True)
    detail = Column(Text)
    prev_hash = Column(String(64))
    hash = Column(String(64))

    def hash_payload(self):
        return {"actor": self.actor, "action": self.action, "application_id": self.application_id,
                "detail": self.detail, "created_at": _iso(self.created_at)}

    def detail_dict(self):
        try:
            return json.loads(self.detail or "{}")
        except json.JSONDecodeError:
            return {}


Index("ix_applications_owner_state", Application.owner_email, Application.state)


def append_audit(db, actor, action, application_id=None, detail=None):
    """Append one immutable event to the chain. Callers commit."""
    last = db.scalar(select(AuditEvent.hash).order_by(AuditEvent.id.desc()).limit(1))
    event = AuditEvent(actor=actor, action=action, application_id=application_id, detail=canonical(detail or {}), created_at=utcnow())
    event.prev_hash = last or GENESIS
    event.hash = chain_hash(event.prev_hash, event.hash_payload())
    db.add(event)
    return event


def record_change(db, actor, action, application, changes: dict, extra: dict | None = None):
    """Audit a field change with previous and updated values, then apply it."""
    previous = {field: getattr(application, field) for field in changes}
    for field, value in changes.items():
        setattr(application, field, value)
    detail = {"previous": previous, "updated": changes}
    if extra:
        detail.update(extra)
    return append_audit(db, actor, action, application.id, detail)


def verify_audit_chain(db):
    prev, checked, first_bad = GENESIS, 0, None
    for event in db.scalars(select(AuditEvent).order_by(AuditEvent.id)):
        if event.prev_hash != prev or chain_hash(event.prev_hash, event.hash_payload()) != event.hash:
            first_bad = first_bad or event.id
        prev = event.hash
        checked += 1
    return {"ok": first_bad is None, "checked": checked, "first_bad_id": first_bad}


def next_reference(db):
    year = utcnow().year
    count = db.scalar(select(Application.id).order_by(Application.id.desc()).limit(1)) or 0
    return f"EQ-{year}-{count + 1:06d}"


def _migrate():
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name not in existing:
                    conn.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {column.name} {column.type.compile(engine.dialect)}"))


def init_db():
    _migrate()
    Base.metadata.create_all(engine)
