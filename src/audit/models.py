"""
SQLite audit log — stores every loan decision with full pipeline trace.
"""

import json
import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "audit.db"
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Session = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class LoanDecision(Base):
    __tablename__ = "loan_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    applicant_name = Column(String(200), nullable=True)

    loan_amount = Column(Float)
    mortgage_due = Column(Float, nullable=True)
    property_value = Column(Float, nullable=True)
    reason = Column(String(50), nullable=True)
    job = Column(String(50), nullable=True)
    yoj = Column(Float, nullable=True)
    derog = Column(Float, nullable=True)
    delinq = Column(Float, nullable=True)
    clage = Column(Float, nullable=True)
    ninq = Column(Float, nullable=True)
    clno = Column(Float, nullable=True)
    debtinc = Column(Float, nullable=True)

    decision = Column(String(20))
    default_probability = Column(Float)
    confidence = Column(Float)

    shap_reasons = Column(Text)
    email_draft = Column(Text)
    final_email = Column(Text)

    bias_score = Column(Float)
    bias_action = Column(String(20))
    bias_rewrites = Column(Integer, default=0)
    bias_audit_trail = Column(Text)

    def set_shap(self, reasons: list[dict]):
        self.shap_reasons = json.dumps(reasons)

    def get_shap(self) -> list[dict]:
        return json.loads(self.shap_reasons) if self.shap_reasons else []

    def set_bias_trail(self, trail: list[dict]):
        self.bias_audit_trail = json.dumps(trail)


def init_db():
    Base.metadata.create_all(engine)
