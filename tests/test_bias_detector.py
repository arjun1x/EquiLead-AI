"""Tests for bias detector (unit tests for BiasResult and AgentResult)."""

from src.agents.bias_detector import BiasResult, AgentResult


def test_bias_result_creation():
    result = BiasResult(score=0.1, issues=[], summary="No bias")
    assert result.score == 0.1
    assert result.issues == []
    assert result.summary == "No bias"


def test_agent_result_defaults():
    result = AgentResult(final_email="test", decision="send")
    assert result.bias_scores == []
    assert result.rewrites == 0
    assert result.audit_trail == []


def test_agent_result_send():
    result = AgentResult(
        final_email="Dear Applicant...",
        decision="send",
        bias_scores=[0.05],
        rewrites=0,
    )
    assert result.decision == "send"
    assert result.bias_scores[0] < 0.3


def test_agent_result_rewritten():
    result = AgentResult(
        final_email="Rewritten email...",
        decision="rewritten",
        bias_scores=[0.45, 0.15],
        rewrites=1,
    )
    assert result.decision == "rewritten"
    assert result.rewrites == 1
    assert result.bias_scores[-1] < 0.3


def test_agent_result_escalate():
    result = AgentResult(
        final_email="Problematic email...",
        decision="escalate",
        bias_scores=[0.85],
        rewrites=0,
    )
    assert result.decision == "escalate"
    assert result.bias_scores[0] > 0.7
