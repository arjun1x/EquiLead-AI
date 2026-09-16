"""Flat-project storage. New database; existing repository data is never overwritten.

Two append-only hash chains live here:
  * loan_decisions.record_hash chains every model output (inputs, score, reasons, drafts) to the record before it.
  * audit_events.hash chains every workspace action (logins, exports, human reviews, purges).
Human review fields are written next to the decision but are NOT part of the sealed payload; they are
recorded in the audit chain instead, so the model output can never be silently rewritten.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

ROOT = Path(__file__).resolve().parent
DEMO_MODE = os.getenv("DEMO_MODE", "1") == "1"
DB_PATH = ROOT / ("equilead-demo.sqlite3" if DEMO_MODE else "equilead.sqlite3")
DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite:") else {})
Session = sessionmaker(bind=engine, expire_on_commit=False)

ROLES = {"analyst": 1, "reviewer": 2, "admin": 3}
GENESIS = "0" * 64
INPUT_FIELDS = ["loan_amount", "mortgage_due", "property_value", "reason", "job", "yoj",
                "derog", "delinq", "clage", "ninq", "clno", "debtinc"]


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def chain_hash(prev_hash, payload):
    return hashlib.sha256(((prev_hash or GENESIS) + "\n" + canonical(payload)).encode()).hexdigest()


def _iso(value):
    """SQLite drops tzinfo, so hashes are computed on naive UTC timestamps at write and read time."""
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return value.isoformat()


def _num(value):
    return None if value is None else float(value)


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


class LoanDecision(Base):
    __tablename__ = "loan_decisions"
    id = Column(Integer, primary_key=True)
    owner_email = Column(String(254), nullable=False, index=True)
    created_at = Column(DateTime, default=utcnow)
    applicant_name = Column(String(200))
    loan_amount = Column(Float)
    mortgage_due = Column(Float)
    property_value = Column(Float)
    reason = Column(String(50))
    job = Column(String(50))
    yoj = Column(Float)
    derog = Column(Float)
    delinq = Column(Float)
    clage = Column(Float)
    ninq = Column(Float)
    clno = Column(Float)
    debtinc = Column(Float)
    # Model output (sealed by record_hash)
    decision = Column(String(20))
    default_probability = Column(Float)      # score used against the threshold (calibrated when available)
    raw_score = Column(Float)                # uncalibrated class-weighted model score
    calibrated_probability = Column(Float)   # None unless a calibrator was applied
    confidence = Column(Float)
    shap_reasons = Column(Text)
    reason_codes = Column(Text)              # adverse-action reason codes (JSON list)
    reason_code_version = Column(String(40))
    email_draft = Column(Text)
    final_email = Column(Text)
    bias_score = Column(Float, nullable=True)
    bias_action = Column(String(20))
    bias_rewrites = Column(Integer, default=0)
    bias_audit_trail = Column(Text)
    model_version = Column(String(100))
    dataset_version = Column(String(100))
    training_date = Column(String(40))
    threshold = Column(Float)
    calibration = Column(String(80))
    is_demo = Column(Integer, default=0)
    prev_hash = Column(String(64))
    record_hash = Column(String(64))
    # Human review (not sealed; every change is an audit event)
    human_decision = Column(String(20))      # pending | approved | declined
    human_note = Column(Text)
    reviewed_by = Column(String(254))
    reviewed_at = Column(DateTime)
    outcome = Column(Integer, nullable=True)  # observed label for monitoring: 1 = defaulted, 0 = repaid

    def set_shap(self, value):
        self.shap_reasons = json.dumps(value)

    def get_shap(self):
        return json.loads(self.shap_reasons or "[]")

    def set_bias_trail(self, value):
        self.bias_audit_trail = json.dumps(value)

    def get_bias_trail(self):
        return json.loads(self.bias_audit_trail or "[]")

    def set_reason_codes(self, value):
        self.reason_codes = json.dumps(value)

    def get_reason_codes(self):
        return json.loads(self.reason_codes or "[]")

    def hash_payload(self):
        inputs = {k: getattr(self, k) for k in INPUT_FIELDS}
        return {
            "owner_email": self.owner_email, "applicant_name": self.applicant_name,
            "inputs": {k: (v if k in ("reason", "job") else _num(v)) for k, v in inputs.items()},
            "decision": self.decision, "default_probability": _num(self.default_probability),
            "raw_score": _num(self.raw_score), "calibrated_probability": _num(self.calibrated_probability),
            "confidence": _num(self.confidence), "shap_reasons": self.shap_reasons,
            "reason_codes": self.reason_codes, "reason_code_version": self.reason_code_version,
            "email_draft": self.email_draft, "final_email": self.final_email,
            "bias_score": _num(self.bias_score), "bias_action": self.bias_action,
            "bias_rewrites": self.bias_rewrites, "bias_audit_trail": self.bias_audit_trail,
            "model_version": self.model_version, "dataset_version": self.dataset_version,
            "training_date": self.training_date, "threshold": _num(self.threshold),
            "calibration": self.calibration, "is_demo": self.is_demo, "created_at": _iso(self.created_at),
        }

    def seal(self, prev_hash):
        self.prev_hash = prev_hash or GENESIS
        self.record_hash = chain_hash(self.prev_hash, self.hash_payload())
        return self.record_hash


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=utcnow)
    actor = Column(String(254))
    action = Column(String(40), index=True)
    record_id = Column(Integer, nullable=True)
    detail = Column(Text)
    prev_hash = Column(String(64))
    hash = Column(String(64))

    def hash_payload(self):
        return {"actor": self.actor, "action": self.action, "record_id": self.record_id,
                "detail": self.detail, "created_at": _iso(self.created_at)}


def latest_decision_hash(db):
    return db.scalar(select(LoanDecision.record_hash).order_by(LoanDecision.id.desc()).limit(1)) or GENESIS


def append_audit(db, actor, action, record_id=None, detail=None):
    """Append one immutable event. Callers commit."""
    last = db.scalar(select(AuditEvent.hash).order_by(AuditEvent.id.desc()).limit(1))
    event = AuditEvent(actor=actor, action=action, record_id=record_id,
                       detail=canonical(detail or {}), created_at=utcnow())
    event.prev_hash = last or GENESIS
    event.hash = chain_hash(event.prev_hash, event.hash_payload())
    db.add(event)
    return event


def verify_decision_chain(db):
    """Recompute every sealed decision hash. Gaps left by a retention purge are allowed only when the
    purge event recorded the hash of the deleted record."""
    purged = set()
    for event in db.scalars(select(AuditEvent).where(AuditEvent.action == "retention_purge")):
        purged.update(json.loads(event.detail or "{}").get("hashes", []))
    prev, checked, unsealed, first_bad = GENESIS, 0, 0, None
    for record in db.scalars(select(LoanDecision).order_by(LoanDecision.id)):
        if not record.record_hash:
            unsealed += 1
            prev = GENESIS
            continue
        linked = record.prev_hash == prev or record.prev_hash in purged
        if not linked or chain_hash(record.prev_hash, record.hash_payload()) != record.record_hash:
            first_bad = first_bad or record.id
        prev = record.record_hash
        checked += 1
    return {"ok": first_bad is None, "checked": checked, "unsealed": unsealed, "first_bad_id": first_bad}


def verify_audit_chain(db):
    prev, checked, first_bad = GENESIS, 0, None
    for event in db.scalars(select(AuditEvent).order_by(AuditEvent.id)):
        if event.prev_hash != prev or chain_hash(event.prev_hash, event.hash_payload()) != event.hash:
            first_bad = first_bad or event.id
        prev = event.hash
        checked += 1
    return {"ok": first_bad is None, "checked": checked, "first_bad_id": first_bad}


def _migrate():
    """Add columns introduced after a database was created. SQLite has no ALTER ... ADD IF NOT EXISTS."""
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
