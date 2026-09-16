"""Explicitly synthetic fixtures. These values are not model predictions."""
import datetime as dt
from sqlalchemy import select
from models import Session, LoanDecision, latest_decision_hash, utcnow
from model_card import DEMO_CARD, decision_metadata
from reason_codes import adverse_action_reasons, REASON_CODE_VERSION

DEMO_OWNER = "demo@equilead.local"
DEMO_REVIEWER = "reviewer@equilead.local"
FEATURE_NAMES = {
    "LOAN": "Loan amount", "MORTDUE": "Mortgage balance", "VALUE": "Property value",
    "REASON": "Loan purpose", "JOB": "Occupation", "YOJ": "Employment tenure",
    "DEROG": "Derogatory reports", "DELINQ": "Delinquent accounts", "CLAGE": "Credit history",
    "NINQ": "Recent inquiries", "CLNO": "Credit accounts", "DEBTINC": "Debt-to-income ratio",
}
DEFAULTS = dict(applicant_name="", loan=45000, mortdue=185000, value=420000,
                reason="HomeImp", job="ProfExe", yoj=6, derog=0, delinq=0,
                clage=156, ninq=1, clno=12, debtinc=28.5, demo_outcome="approved")


def fixture(data, outcome="approved"):
    """Chosen preview scenario, deliberately independent of financial inputs."""
    decision = "denied" if outcome == "denied" else "approved"
    probability = 0.684 if decision == "denied" else 0.127
    action = "escalate" if outcome == "review" else "send"
    reasons = [
        {"feature": "DEBTINC", "shap_value": 0.82 if decision == "denied" else -0.74, "value": data["DEBTINC"]},
        {"feature": "CLAGE", "shap_value": -0.52, "value": data["CLAGE"]},
        {"feature": "DELINQ", "shap_value": 0.44 if decision == "denied" else -0.35, "value": data["DELINQ"]},
        {"feature": "LOAN", "shap_value": 0.19, "value": data["LOAN"]},
        {"feature": "YOJ", "shap_value": -0.12, "value": data["YOJ"]},
    ]
    email = ("Dear Applicant,\n\nThis is a sample communication for an interface demonstration. "
             f"The selected example shows an {('approval' if decision == 'approved' else 'adverse')} recommendation "
             "for a home-equity application. No actual application has been assessed.\n\n"
             "A loan officer would verify the application, review the supporting documents, and confirm "
             "the appropriate next steps before communicating a decision.\n\nEquiLead · Loan Services")
    result = dict(decision=decision, default_probability=probability, raw_score=probability, calibrated_probability=None,
                  confidence=max(probability, 1-probability)*100,
                  shap_reasons=reasons, reason_codes=adverse_action_reasons(reasons, decision),
                  reason_code_version=REASON_CODE_VERSION,
                  email_draft=email, final_email=email,
                  bias_score=0.81 if action == "escalate" else 0.08, bias_action=action,
                  bias_rewrites=0, is_demo=1,
                  bias_audit_trail=[{"attempt": 0, "bias_score": 0.81 if action == "escalate" else 0.08,
                                     "summary": "Synthetic language-review example; no API call was made.", "issues": []}])
    result.update(decision_metadata(DEMO_CARD))
    return result


def make_record(owner, name, data, result):
    r = LoanDecision(owner_email=owner, applicant_name=name or "Unnamed applicant",
        loan_amount=data["LOAN"], mortgage_due=data["MORTDUE"], property_value=data["VALUE"],
        reason=data["REASON"], job=data["JOB"], yoj=data["YOJ"], derog=data["DEROG"],
        delinq=data["DELINQ"], clage=data["CLAGE"], ninq=data["NINQ"], clno=data["CLNO"], debtinc=data["DEBTINC"],
        created_at=utcnow(), human_decision="pending")
    for key, value in result.items():
        if key == "shap_reasons": r.set_shap(value)
        elif key == "bias_audit_trail": r.set_bias_trail(value)
        elif key == "reason_codes": r.set_reason_codes(value)
        else: setattr(r, key, value)
    return r


def store_record(db, record):
    """Append and seal a decision. Callers commit."""
    record.seal(latest_decision_hash(db))
    db.add(record)
    db.flush()
    return record


# name, loan, purpose index, occupation, debt-to-income, synthetic preview outcome, synthetic observed outcome
_ROWS = [("Morgan Ellis", 45000, "HomeImp", "ProfExe", 28.5, "approved", 0),
         ("Jamie Rivera", 68000, "DebtCon", "Office", 33.0, "review", None),
         ("Alex Bennett", 32000, "HomeImp", "Mgr", 24.2, "approved", 0),
         ("Taylor Brooks", 95000, "DebtCon", "Other", 41.8, "denied", 1),
         ("Jordan Parker", 55000, "HomeImp", "Sales", 30.1, "approved", 1),
         ("Casey Sullivan", 28000, "DebtCon", "Self", 22.7, "approved", 0),
         ("Avery Collins", 72000, "HomeImp", "Other", 39.4, "denied", 0),
         ("Riley Morgan", 38000, "DebtCon", "Office", 31.6, "review", None)]


def seed_demo():
    with Session() as db:
        if db.scalar(select(LoanDecision.id).where(LoanDecision.owner_email == DEMO_OWNER).limit(1)):
            return
        for i, (name, amount, purpose, job, debtinc, outcome, observed) in enumerate(_ROWS):
            data = {k.upper(): v for k, v in DEFAULTS.items() if k not in ("applicant_name", "demo_outcome")}
            data.update(LOAN=amount, REASON=purpose, JOB=job, DEBTINC=debtinc)
            r = make_record(DEMO_OWNER, name, data, fixture(data, outcome))
            r.created_at = utcnow() - dt.timedelta(hours=i*5+1)
            r.outcome = observed
            store_record(db, r)
        db.commit()
