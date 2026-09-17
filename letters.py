"""Decision-letter generator with controlled templates and a language checker.

The decision, amounts and reasons come from the stored record, never from free text. The checker flags
protected-class references, subjective wording, guarantees, vague explanations, internal jargon, unsupported
conclusions and reasons that do not match the stored reason codes, and produces a corrected draft. A human
must approve the corrected draft before the case can be finalized.
"""
from __future__ import annotations

import datetime as dt
import re

LETTER_TEMPLATE_VERSION = "letter-templates-v1"
LETTER_TYPES = {"approval": "Approval", "conditional_approval": "Conditional approval",
                "manual_review": "Manual-review request", "incomplete": "Incomplete application",
                "adverse_action": "Adverse action"}
DISCLAIMER = ("This letter was produced by the EquiLead demonstration workspace. It is not a regulatory notice. "
              "Production use requires legal and compliance review of the templates, reasons and timing.")
HUMAN_REVIEW_LINE = {"reviewed": "This decision was reviewed and recorded by {signoff_name}, {signoff_role}.",
                     "pending": "This letter is a draft awaiting review by a loan officer."}

TEMPLATES = {
    "approval": """{date}

{applicant_name}
Re: Application {reference} — {product}

Dear {applicant_name},

We are pleased to tell you that your application for a {product_lower} has been approved for {approved_amount}{term_clause}.

Your closing team will contact you with the remaining steps, including the final documents to sign and the date your funds or credit line become available. The terms described here are subject to the final closing documents.

{human_review}

Sincerely,
EquiLead Lending Services

{disclaimer}
""",
    "conditional_approval": """{date}

{applicant_name}
Re: Application {reference} — {product}

Dear {applicant_name},

Your application for a {product_lower} has been conditionally approved for {approved_amount}{term_clause}. Final approval depends on the following items:

{conditions}

Please send these within 30 days. If we do not receive them, we will be unable to complete your application. Contact your loan officer with any questions.

{human_review}

Sincerely,
EquiLead Lending Services

{disclaimer}
""",
    "manual_review": """{date}

{applicant_name}
Re: Application {reference} — {product}

Dear {applicant_name},

Thank you for your application for a {product_lower}. Your application has been referred to a loan officer for a closer review, which may take a few additional days.

We may contact you to confirm details or request supporting documents. No decision has been made at this time, and this letter is not a denial of credit.

{human_review}

Sincerely,
EquiLead Lending Services

{disclaimer}
""",
    "incomplete": """{date}

{applicant_name}
Re: Application {reference} — {product}

Dear {applicant_name},

We have received your application for a {product_lower}, but we need the following information or documents to continue:

{conditions}

Please provide these items within 30 days. If we do not receive them by then, we will be unable to give further consideration to your application.

{human_review}

Sincerely,
EquiLead Lending Services

{disclaimer}
""",
    "adverse_action": """{date}

{applicant_name}
Re: Application {reference} — {product}

Dear {applicant_name},

Thank you for your application for a {product_lower}. After careful review, we are unable to approve your application at this time.

The principal reason(s) for our decision:

{reasons}

If you believe any of this information is inaccurate, you may contact us to discuss your application. You have the right to a statement of specific reasons and to obtain a copy of any consumer report we used, from the reporting agency, within 60 days of this notice.

{human_review}

Sincerely,
EquiLead Lending Services

{disclaimer}
""",
}

# ── Language checker ────────────────────────────────────

PROTECTED_TERMS = r"\b(race|racial|color|religion|religious|national origin|nationality|ethnic|ethnicity|sex|gender|female|male|woman|man|women|men|marital|married|divorced|single mother|single father|age|elderly|senior citizen|young|disability|disabled|handicap|pregnan\w*|public assistance|welfare|immigrant|immigration|citizenship)\b"
SUBJECTIVE_TERMS = r"\b(irresponsible|lazy|careless|reckless|unreliable|untrustworthy|bad borrower|poor character|dishonest|unfortunately for you|obviously|clearly failed)\b"
GUARANTEE_TERMS = r"\b(guarantee[ds]?|guaranteed approval|will be approved|promise[sd]?|certainly be approved|assured approval|cannot be declined)\b"
VAGUE_TERMS = r"(failed (our )?internal standards|did not meet (our )?criteria|does not meet our standards|internal (review )?requirements|for internal reasons|our policy does not allow|risk profile)"
JARGON_TERMS = r"\b(SHAP|XGBoost|gradient[- ]boost\w*|random forest|logistic regression|scorecard model|machine learning|algorithm\w*|model score|probability of default|\bPD\b|feature[s]? contribution|risk band|consensus|calibrat\w*|log[- ]odds|classifier|predict\w* default)\b"
UNSUPPORTED_TERMS = r"\b(you will default|you cannot afford|you are likely to default|you will not repay|you are a high[- ]risk borrower|will fail to pay)\b"

CHECKS = [("protected_class", PROTECTED_TERMS, "References a protected characteristic.", "remove"),
          ("subjective", SUBJECTIVE_TERMS, "Subjective or insulting wording.", "neutral"),
          ("guarantee", GUARANTEE_TERMS, "Promises or guarantees an outcome.", "soften"),
          ("vague", VAGUE_TERMS, "Vague explanation that does not state a specific reason.", "specific"),
          ("jargon", JARGON_TERMS, "Internal model or technical jargon.", "plain"),
          ("unsupported", UNSUPPORTED_TERMS, "Conclusion about the applicant that the record does not support.", "neutral")]

REPLACEMENTS = {"neutral": "", "soften": "may be considered", "specific": "the reasons listed above", "plain": "our review"}


def _term_clause(term_years):
    return f" with a {term_years}-year term" if term_years else ""


def _bullets(items):
    return "\n".join(f"  {i}. {text}" for i, text in enumerate(items, 1)) if items else "  (none listed)"


def letter_type_for(human_decision: str, derived_missing_documents: list | None = None) -> str:
    if human_decision == "approved":
        return "approval"
    if human_decision == "conditionally_approved":
        return "incomplete" if derived_missing_documents and len(derived_missing_documents) >= 3 else "conditional_approval"
    if human_decision == "declined":
        return "adverse_action"
    return "manual_review"


def generate_letter(letter_type: str, *, applicant_name: str, reference: str, product_label: str, approved_amount=None,
                    term_years=None, conditions=None, reasons=None, signoff_name=None, signoff_role=None,
                    date: dt.date | None = None) -> str:
    if letter_type not in TEMPLATES:
        raise ValueError(f"Unknown letter type {letter_type}")
    date = date or dt.date.today()
    human_review = (HUMAN_REVIEW_LINE["reviewed"].format(signoff_name=signoff_name, signoff_role=signoff_role)
                    if signoff_name and signoff_role else HUMAN_REVIEW_LINE["pending"])
    return TEMPLATES[letter_type].format(
        date=date.strftime("%B %d, %Y"), applicant_name=applicant_name or "Applicant", reference=reference,
        product=product_label, product_lower=product_label.lower(),
        approved_amount=f"${approved_amount:,.0f}" if approved_amount else "the requested amount",
        term_clause=_term_clause(term_years), conditions=_bullets(conditions or []),
        reasons=_bullets(reasons or []), human_review=human_review, disclaimer=DISCLAIMER)


def check_language(text: str, *, stored_decision: str | None = None, stored_reasons: list[str] | None = None,
                   letter_type: str | None = None) -> list[dict]:
    """Return a list of issues: {category, excerpt, message, line, fix}."""
    issues = []
    lines = text.splitlines()
    for category, pattern, message, fix in CHECKS:
        for line_no, line in enumerate(lines, 1):
            if line.startswith("This letter was produced") or line.startswith("Re: "):
                continue
            for match in re.finditer(pattern, line, flags=re.IGNORECASE):
                issues.append({"category": category, "excerpt": match.group(0), "message": message, "line": line_no, "fix": fix})
    if letter_type == "adverse_action":
        body = "\n".join(lines)
        if stored_reasons is not None:
            reason_block = body.split("The principal reason(s) for our decision:")[-1].split("If you believe")[0] if "principal reason" in body else ""
            listed = [l.strip()[3:].strip() for l in reason_block.splitlines() if re.match(r"\s*\d+\.", l)]
            for reason in listed:
                if reason and reason not in stored_reasons:
                    issues.append({"category": "mismatch", "excerpt": reason, "message": "Reason is not on the stored decision record.", "line": None, "fix": "remove"})
            if not listed:
                issues.append({"category": "vague", "excerpt": "(no reasons listed)", "message": "An adverse-action letter must list specific reasons.", "line": None, "fix": "specific"})
    if stored_decision:
        expected = {"approved": "has been approved", "conditionally_approved": "conditionally approved",
                    "declined": "unable to approve", "manual_review": "referred to a loan officer"}.get(stored_decision)
        if expected and expected not in text:
            issues.append({"category": "mismatch", "excerpt": expected, "message": f"Letter wording does not match the recorded decision ({stored_decision}).", "line": None, "fix": "regenerate"})
    return issues


def correct_draft(text: str, issues: list[dict]) -> str:
    """Apply mechanical corrections: drop protected references' sentences, neutralise or remove flagged terms."""
    lines = text.splitlines()
    # Locate line numbers for record-mismatch issues, which are reported without one.
    for issue in issues:
        if issue.get("line") is None and issue.get("fix") == "remove":
            for i, line in enumerate(lines, 1):
                if issue["excerpt"] and issue["excerpt"].lower() in line.lower():
                    issue["line"] = i
                    break
    ordered = sorted(issues, key=lambda i: 0 if i.get("fix") == "remove" else 1)
    for issue in ordered:
        if issue.get("line") is None:
            continue
        idx = issue["line"] - 1
        if idx >= len(lines):
            continue
        line = lines[idx]
        if issue["fix"] == "remove":
            if re.match(r"\s*\d+\.\s", line):          # a numbered reason: drop the whole entry
                lines[idx] = ""
                continue
            sentences = re.split(r"(?<=[.!?])\s+", line)
            kept = [s for s in sentences if issue["excerpt"].lower() not in s.lower()]
            lines[idx] = " ".join(kept)
        else:
            replacement = REPLACEMENTS.get(issue["fix"], "")
            lines[idx] = re.sub(re.escape(issue["excerpt"]), replacement, line, flags=re.IGNORECASE)
            lines[idx] = re.sub(r"\s{2,}", " ", lines[idx]).replace(" .", ".").replace(" ,", ",")
    corrected = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", corrected)
