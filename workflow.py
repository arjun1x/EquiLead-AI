"""Application workflow states and allowed transitions."""
from __future__ import annotations

STATES = ["draft", "submitted", "scored", "manual_review", "approved", "conditionally_approved", "declined",
          "withdrawn", "letter_generated", "finalized"]
LABELS = {"draft": "Draft", "submitted": "Submitted", "scored": "Scored", "manual_review": "Manual review",
          "approved": "Approved", "conditionally_approved": "Conditionally approved", "declined": "Declined",
          "withdrawn": "Withdrawn", "letter_generated": "Letter generated", "finalized": "Finalized"}
DECISION_STATES = {"approved": "approved", "conditionally_approved": "conditionally_approved",
                   "declined": "declined", "manual_review": "manual_review"}
TERMINAL = {"withdrawn", "finalized"}
TRANSITIONS = {
    "draft": {"submitted", "withdrawn"},
    "submitted": {"scored", "manual_review", "withdrawn"},
    "scored": {"approved", "conditionally_approved", "declined", "manual_review", "submitted", "withdrawn"},
    "manual_review": {"approved", "conditionally_approved", "declined", "submitted", "withdrawn"},
    "approved": {"letter_generated", "scored", "manual_review", "withdrawn"},
    "conditionally_approved": {"letter_generated", "scored", "manual_review", "withdrawn"},
    "declined": {"letter_generated", "scored", "manual_review", "withdrawn"},
    "letter_generated": {"finalized", "approved", "conditionally_approved", "declined", "withdrawn"},
    "withdrawn": set(),
    "finalized": set(),
}


class TransitionError(ValueError):
    pass


def can_transition(current: str, target: str) -> bool:
    return target in TRANSITIONS.get(current, set())


def assert_transition(current: str, target: str):
    if not can_transition(current, target):
        raise TransitionError(f"Cannot move an application from {LABELS.get(current, current)} to {LABELS.get(target, target)}.")


def can_score(state: str) -> bool:
    return state in ("submitted", "scored", "manual_review", "approved", "conditionally_approved", "declined")


def can_decide(state: str) -> bool:
    return state in ("scored", "manual_review", "approved", "conditionally_approved", "declined", "letter_generated")


def can_generate_letter(state: str) -> bool:
    return state in ("approved", "conditionally_approved", "declined", "manual_review", "letter_generated")


def finalize_blockers(application) -> list[str]:
    """Reasons a case cannot be finalized yet. Empty list means it can."""
    blockers = []
    if application.state not in ("letter_generated",):
        blockers.append("A decision letter must be generated and approved first.")
    if not application.human_decision:
        blockers.append("A human decision has not been recorded.")
    if not (application.signoff_name and application.signoff_role):
        blockers.append("Sign-off name and role are missing.")
    if application.override and not application.override_reason:
        blockers.append("The override needs a written explanation.")
    if not application.letter_approved_at:
        blockers.append("The current letter has not been approved by a reviewer.")
    return blockers
