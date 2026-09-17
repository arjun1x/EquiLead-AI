"""Letter templates and the language checker. Run: python -m pytest -q -p no:cacheprovider test_letters.py"""
import pytest

from letters import DISCLAIMER, HUMAN_REVIEW_LINE, LETTER_TYPES, check_language, correct_draft, generate_letter, letter_type_for

REASONS = ["Your combined loan-to-value ratio is above the level we can lend at.", "Your credit report shows recent late payments or delinquent accounts."]


def make(letter_type, **kw):
    base = dict(applicant_name="Jordan Lee", reference="EQ-2026-000042", product_label="Home equity line of credit", signoff_name="Sam Okafor", signoff_role="Loan officer")
    return generate_letter(letter_type, **{**base, **kw})


def categories(issues):
    return {i["category"] for i in issues}


def test_letter_type_follows_the_recorded_decision():
    assert letter_type_for("approved") == "approval"
    assert letter_type_for("conditionally_approved") == "conditional_approval"
    assert letter_type_for("conditionally_approved", ["insurance", "identity", "mortgage_statement"]) == "incomplete"
    assert letter_type_for("conditionally_approved", ["insurance"]) == "conditional_approval"
    assert letter_type_for("declined") == "adverse_action"
    assert letter_type_for("manual_review") == "manual_review"


@pytest.mark.parametrize("letter_type", list(LETTER_TYPES))
def test_every_template_carries_reference_disclaimer_and_human_review(letter_type):
    text = make(letter_type, approved_amount=50000, term_years=15, conditions=["Provide two pay stubs"], reasons=REASONS)
    assert "EQ-2026-000042" in text and "Jordan Lee" in text
    assert DISCLAIMER in text and HUMAN_REVIEW_LINE["reviewed"].split("{")[0].strip() in text
    assert "Sam Okafor" in text


def test_generated_templates_pass_the_checker():
    assert check_language(make("approval", approved_amount=50000, term_years=15), stored_decision="approved", letter_type="approval") == []
    assert check_language(make("adverse_action", reasons=REASONS), stored_decision="declined", stored_reasons=REASONS, letter_type="adverse_action") == []
    assert check_language(make("conditional_approval", approved_amount=40000, conditions=["Provide insurance binder"]), stored_decision="conditionally_approved", letter_type="conditional_approval") == []


def test_adverse_letter_without_reasons_is_flagged():
    text = make("adverse_action", reasons=[])
    assert "vague" in categories(check_language(text, stored_decision="declined", stored_reasons=[], letter_type="adverse_action"))


def test_checker_catches_each_prohibited_category():
    text = ("Dear Jordan,\n"
            "As a woman with your background we cannot approve you.\n"
            "You have been an irresponsible borrower.\n"
            "We guarantee approval next time.\n"
            "You failed our internal standards.\n"
            "Our XGBoost model and SHAP values decided this.\n"
            "You will default on any loan.\n")
    found = categories(check_language(text, stored_decision="declined", stored_reasons=REASONS, letter_type="adverse_action"))
    assert {"protected_class", "subjective", "guarantee", "vague", "jargon", "unsupported"} <= found


def test_checker_flags_reasons_that_do_not_match_the_record():
    text = make("adverse_action", reasons=REASONS + ["You live in the wrong neighbourhood."])
    issues = check_language(text, stored_decision="declined", stored_reasons=REASONS, letter_type="adverse_action")
    assert "mismatch" in categories(issues)
    corrected = correct_draft(text, [dict(i) for i in issues])
    assert "wrong neighbourhood" not in corrected and REASONS[0] in corrected


def test_checker_flags_decision_wording_inconsistent_with_record():
    text = make("approval", approved_amount=50000, term_years=15)
    issues = check_language(text, stored_decision="declined", letter_type="approval")
    assert "mismatch" in categories(issues)


def test_correct_draft_removes_or_softens_issues():
    text = make("adverse_action", reasons=REASONS) + "\nAs a woman you should know we guarantee nothing.\n"
    issues = check_language(text, stored_decision="declined", stored_reasons=REASONS, letter_type="adverse_action")
    corrected = correct_draft(text, [dict(i) for i in issues])
    remaining = categories(check_language(corrected, stored_decision="declined", stored_reasons=REASONS, letter_type="adverse_action"))
    assert "protected_class" not in remaining and "guarantee" not in remaining
    assert REASONS[1] in corrected and "EQ-2026-000042" in corrected


def test_letters_never_mention_model_internals():
    for letter_type in LETTER_TYPES:
        text = make(letter_type, approved_amount=1000, term_years=5, reasons=REASONS, conditions=["x"]).lower()
        for word in ("shap", "xgboost", "random forest", "logistic", "probability of default", "model score"):
            assert word not in text
