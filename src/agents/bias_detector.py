"""
Phase 4: Bias detection agent with rewrite loop.

Scores emails 0.0-1.0 for discriminatory language.
  < 0.3  -> approve and send
  0.3-0.7 -> rewrite agent rewrites (max 3 retries)
  > 0.7  -> escalate to human review
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "llm" / "prompts"
MODEL = "claude-sonnet-4-5-20250929"
MAX_REWRITES = 3


@dataclass
class BiasResult:
    score: float
    issues: list[dict]
    summary: str


@dataclass
class AgentResult:
    final_email: str
    decision: str  # "send", "rewritten", "escalate"
    bias_scores: list[float] = field(default_factory=list)
    rewrites: int = 0
    audit_trail: list[dict] = field(default_factory=list)


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def score_bias(email: str, client: anthropic.Anthropic) -> BiasResult:
    system = _load_prompt("bias_system.txt")
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": f"Review this email for bias:\n\n{email}"}],
    )

    raw = response.content[0].text
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        data = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        data = {"score": 0.5, "issues": [], "summary": "Could not parse bias response"}

    return BiasResult(
        score=float(data.get("score", 0.5)),
        issues=data.get("issues", []),
        summary=data.get("summary", ""),
    )


def rewrite_email(email: str, bias_result: BiasResult, client: anthropic.Anthropic) -> str:
    system = _load_prompt("rewrite_system.txt")

    issues_text = "\n".join(
        f"- [{i['type']}] \"{i['quote']}\" -- {i['concern']}"
        for i in bias_result.issues
    )
    if not issues_text:
        issues_text = f"General concern: {bias_result.summary}"

    user_msg = (
        f"Original email:\n\n{email}\n\n"
        f"Bias concerns identified (score: {bias_result.score}):\n{issues_text}\n\n"
        f"Rewrite the email to address all concerns."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text


def run_bias_agent(email: str, api_key: str | None = None) -> AgentResult:
    """
    Run the full bias detection + rewrite loop.

    Returns AgentResult with final_email, decision, and full audit trail.
    """
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    current_email = email
    result = AgentResult(final_email=email, decision="send")

    for attempt in range(MAX_REWRITES + 1):
        bias = score_bias(current_email, client)
        result.bias_scores.append(bias.score)
        result.audit_trail.append({
            "attempt": attempt,
            "bias_score": bias.score,
            "issues": bias.issues,
            "summary": bias.summary,
            "email_snapshot": current_email,
        })

        print(f"  Bias check #{attempt + 1}: score={bias.score:.2f} - {bias.summary}")

        if bias.score < 0.3:
            result.final_email = current_email
            result.decision = "rewritten" if attempt > 0 else "send"
            return result

        if bias.score > 0.7:
            result.final_email = current_email
            result.decision = "escalate"
            print(f"  ** ESCALATED TO HUMAN REVIEW (score={bias.score:.2f}) **")
            return result

        if attempt < MAX_REWRITES:
            print(f"  Rewriting email (attempt {attempt + 1}/{MAX_REWRITES})...")
            current_email = rewrite_email(current_email, bias, client)
            result.rewrites += 1

    result.final_email = current_email
    result.decision = "escalate"
    print(f"  ** ESCALATED: max rewrites reached, last score={result.bias_scores[-1]:.2f} **")
    return result
