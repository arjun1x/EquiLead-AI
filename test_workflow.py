"""Workflow states and finalization guard. Run: python -m pytest -q -p no:cacheprovider test_workflow.py"""
from types import SimpleNamespace

import pytest

import workflow


def test_every_state_has_a_label_and_transition_entry():
    assert set(workflow.STATES) == set(workflow.LABELS) == set(workflow.TRANSITIONS)
    assert {"draft", "submitted", "scored", "manual_review", "approved", "conditionally_approved", "declined", "withdrawn", "letter_generated", "finalized"} <= set(workflow.STATES)


@pytest.mark.parametrize("current,target", [
    ("draft", "submitted"), ("submitted", "scored"), ("submitted", "manual_review"), ("scored", "approved"), ("manual_review", "declined"),
    ("approved", "letter_generated"), ("letter_generated", "finalized"), ("scored", "withdrawn"), ("letter_generated", "declined"),
])
def test_allowed_transitions(current, target):
    assert workflow.can_transition(current, target)
    workflow.assert_transition(current, target)


@pytest.mark.parametrize("current,target", [
    ("draft", "approved"), ("draft", "finalized"), ("scored", "finalized"), ("approved", "finalized"), ("finalized", "approved"),
    ("finalized", "withdrawn"), ("withdrawn", "submitted"), ("submitted", "approved"), ("nope", "draft"),
])
def test_blocked_transitions_raise(current, target):
    assert not workflow.can_transition(current, target)
    with pytest.raises(workflow.TransitionError):
        workflow.assert_transition(current, target)


def test_capability_helpers():
    assert workflow.can_score("scored") and workflow.can_score("manual_review") and not workflow.can_score("finalized") and not workflow.can_score("draft")
    assert workflow.can_decide("manual_review") and workflow.can_decide("letter_generated") and not workflow.can_decide("draft") and not workflow.can_decide("finalized")
    assert workflow.can_generate_letter("approved") and workflow.can_generate_letter("manual_review") and not workflow.can_generate_letter("scored")


def case(**kw):
    base = dict(state="letter_generated", human_decision="approved", signoff_name="Sam", signoff_role="Loan officer", override=False, override_reason=None, letter_approved_at="2026-09-16")
    return SimpleNamespace(**{**base, **kw})


def test_finalize_requires_decision_signoff_and_approved_letter():
    assert workflow.finalize_blockers(case()) == []
    assert any("decision" in b.lower() for b in workflow.finalize_blockers(case(human_decision=None)))
    assert any("sign-off" in b.lower() for b in workflow.finalize_blockers(case(signoff_role="")))
    assert any("approved" in b.lower() for b in workflow.finalize_blockers(case(letter_approved_at=None)))
    assert any("letter" in b.lower() for b in workflow.finalize_blockers(case(state="approved")))


def test_override_without_explanation_blocks_finalization():
    assert any("override" in b.lower() for b in workflow.finalize_blockers(case(override=True, override_reason="")))
    assert workflow.finalize_blockers(case(override=True, override_reason="Documented salary increase verified with employer.")) == []
