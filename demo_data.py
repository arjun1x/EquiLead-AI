"""Demo workspace seeding: a handful of synthetic applications in different workflow states.

Everything created here is labelled ``is_demo`` and scored by the real three-model engine on synthetic
inputs from ``synth_data.generate``. No figure is a real lending decision.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select

from calculations import ApplicationInput, PRODUCT_TYPES
from letters import check_language, correct_draft, generate_letter, letter_type_for, LETTER_TEMPLATE_VERSION
from models import Application, Letter, ModelRun, Session, append_audit, next_reference, utcnow
from reason_codes import customer_statements
from synth_data import generate

DEMO_OWNER = "demo@equilead.local"
DEMO_REVIEWER = "reviewer@equilead.local"
DEMO_NAMES = ["Morgan Ellis", "Jamie Rivera", "Alex Bennett", "Taylor Brooks", "Jordan Parker", "Casey Sullivan", "Avery Collins", "Riley Morgan"]


def sample_inputs(n: int = 8, seed: int = 20260917) -> list[ApplicationInput]:
    frame = generate(600, seed)
    # spread the samples across the risk range so the queue shows every outcome
    frame = frame.sort_values("true_probability").reset_index(drop=True)
    picks = [int(len(frame) * q) for q in (0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.88, 0.97)][:n]
    inputs = []
    for i, idx in enumerate(picks):
        row = frame.iloc[idx]
        docs = [d for d in str(row["documents"]).split("|") if d]
        inputs.append(ApplicationInput(applicant_name=DEMO_NAMES[i % len(DEMO_NAMES)], product_type=row["product_type"],
                                       annual_income=float(row["annual_income"]), verified_monthly_income=float(row["verified_monthly_income"]),
                                       employment_status=row["employment_status"], employment_years=float(row["employment_years"]),
                                       monthly_debt=float(row["monthly_debt"]), credit_score=int(row["credit_score"]),
                                       credit_history_years=float(row["credit_history_years"]), delinquencies_24m=int(row["delinquencies_24m"]),
                                       inquiries_6m=int(row["inquiries_6m"]), property_value=float(row["property_value"]),
                                       mortgage_balance=float(row["mortgage_balance"]), requested_amount=float(row["requested_amount"]),
                                       occupancy_type=row["occupancy_type"], property_type=row["property_type"], loan_purpose=row["loan_purpose"],
                                       term_years=int(row["term_years"]), interest_rate=float(row["interest_rate"]),
                                       cash_reserves=float(row["cash_reserves"]), documents=docs))
    return inputs


def store_scoring(db, application: Application, response, actor: str, batch: int) -> None:
    """Persist a ScoreResponse onto an application (shared by seeding and routes)."""
    application.derived = response.derived
    application.policy = response.policy
    application.consensus = response.consensus.model_dump()
    application.reason_codes = response.reason_codes
    application.explanations = {"counterfactuals": response.counterfactuals, "total_ms": response.total_ms}
    application.scoring_hash = response.scoring_hash
    application.scored_at = utcnow()
    for result in response.models:
        db.add(ModelRun(application_id=application.id, batch=batch, model_key=result.model_key, model_name=result.model_name,
                        model_version=result.model_version, model_hash=result.model_hash, preprocessing_version=result.preprocessing_version,
                        probability=result.probability, raw_probability=result.raw_probability, risk_band=result.risk_band,
                        recommendation=result.recommendation, decision_threshold=result.decision_threshold, band_cutoffs=result.band_cutoffs,
                        drivers={"positive": [d.model_dump() for d in result.drivers_positive], "negative": [d.model_dump() for d in result.drivers_negative]},
                        explainer_method=result.explainer_method, processing_ms=result.processing_ms, scoring_ms=result.scoring_ms,
                        in_range=result.in_range, out_of_range=result.out_of_range, error=result.error))
    append_audit(db, actor, "scored", application.id,
                 {"batch": batch, "scoring_hash": response.scoring_hash, "consensus": response.consensus.final_recommendation,
                  "agreement": response.consensus.agreement, "models": {r.model_key: {"version": r.model_version, "hash": r.model_hash, "probability": r.probability, "error": r.error} for r in response.models},
                  "policy": response.policy["result"], "reason_codes": [r["code"] for r in response.reason_codes.get("principal", [])], "total_ms": response.total_ms})


def seed_demo(score_fn) -> int:
    """Create the demo queue if the demo owner has no applications yet. ``score_fn`` is scoring.score_application."""
    with Session() as db:
        if db.scalar(select(Application.id).where(Application.owner_email == DEMO_OWNER).limit(1)):
            return 0
        plan = ["draft", "scored", "scored", "manual_review", "approved", "declined", "scored", "conditionally_approved"]
        created = 0
        for i, app_input in enumerate(sample_inputs()):
            target = plan[i]
            application = Application(reference=next_reference(db), owner_email=DEMO_OWNER, applicant_name=app_input.applicant_name,
                                      product_type=app_input.product_type, inputs=app_input.model_dump(), state="draft", is_demo=1,
                                      created_at=utcnow() - dt.timedelta(hours=6 * (8 - i)))
            db.add(application); db.flush()
            append_audit(db, DEMO_OWNER, "created", application.id, {"reference": application.reference, "demo": True})
            if target != "draft":
                application.state = "submitted"
                append_audit(db, DEMO_OWNER, "submitted", application.id, {"state": "submitted"})
                response = score_fn(app_input)
                store_scoring(db, application, response, DEMO_OWNER, 1)
                application.state = "manual_review" if response.consensus.routed_to_manual_review else "scored"
            if target in ("approved", "declined", "conditionally_approved"):
                application.state = target
                application.human_decision = target
                application.decided_by, application.decided_at = DEMO_REVIEWER, utcnow()
                application.signoff_name, application.signoff_role = "Sam Okafor", "Senior loan officer"
                application.override = target != {"approve": "approved", "conditional_approval": "conditionally_approved", "decline": "declined"}.get(application.consensus.get("final_recommendation"))
                application.override_reason = "Demo: officer judgement differs from the model consensus after reviewing supporting documents." if application.override else None
                application.approved_amount = app_input.requested_amount if target != "declined" else None
                application.conditions = ["Provide the outstanding documents listed on the application."] if target == "conditionally_approved" else []
                append_audit(db, DEMO_REVIEWER, "decision_recorded", application.id, {"human_decision": target, "override": application.override, "signoff": "Sam Okafor"})
                if target in ("approved", "declined"):
                    ltype = letter_type_for(target, application.derived.get("missing_documents"))
                    reasons = customer_statements(application.reason_codes) if target == "declined" else []
                    draft = generate_letter(ltype, applicant_name=application.applicant_name, reference=application.reference,
                                            product_label=PRODUCT_TYPES[application.product_type], approved_amount=application.approved_amount,
                                            term_years=app_input.term_years, reasons=reasons, signoff_name="Sam Okafor", signoff_role="Senior loan officer")
                    issues = check_language(draft, stored_decision=target, stored_reasons=reasons, letter_type=ltype)
                    letter = Letter(application_id=application.id, version=1, template_version=LETTER_TEMPLATE_VERSION, letter_type=ltype,
                                    draft_text=draft, issues=issues, corrected_text=correct_draft(draft, issues), created_by=DEMO_REVIEWER,
                                    decision_snapshot={"human_decision": target, "reasons": reasons})
                    db.add(letter); db.flush()
                    append_audit(db, DEMO_REVIEWER, "letter_generated", application.id, {"letter_id": letter.id, "type": ltype, "issues": len(issues)})
                    if target == "approved":
                        letter.status, letter.approved_by, letter.approved_at = "approved", DEMO_REVIEWER, utcnow()
                        application.letter_approved_at = letter.approved_at
                        application.state = "letter_generated"
                        append_audit(db, DEMO_REVIEWER, "letter_approved", application.id, {"letter_id": letter.id})
            created += 1
        db.commit()
        return created
