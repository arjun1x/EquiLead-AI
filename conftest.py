"""Shared test setup. Runs before any test module is imported.

Each pytest session gets its own SQLite file in the system temp directory, so the demo database next to
app.py is never touched and no folder is created inside the project. Cache writing is disabled by
run.py / -p no:cacheprovider; use `python -m pytest -q -p no:cacheprovider` to keep the folder flat."""
import os
import pathlib
import sys
import tempfile

sys.dont_write_bytecode = True
_TMP = pathlib.Path(tempfile.mkdtemp(prefix="equilead-tests-"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(_TMP / 'tests.sqlite3').as_posix()}")
os.environ.setdefault("DEMO_MODE", "1")
os.environ.setdefault("SESSION_SECRET", "test-secret-not-for-production")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from calculations import ApplicationInput  # noqa: E402

BASE_APPLICATION = {
    "applicant_name": "Test Applicant", "product_type": "heloc", "annual_income": 120000, "verified_monthly_income": 9500,
    "employment_status": "employed", "employment_years": 6, "monthly_debt": 2400, "credit_score": 735, "credit_history_years": 12,
    "delinquencies_24m": 0, "inquiries_6m": 1, "property_value": 650000, "mortgage_balance": 310000, "requested_amount": 90000,
    "occupancy_type": "primary", "property_type": "single_family", "loan_purpose": "home_improvement", "term_years": 20,
    "interest_rate": 8.5, "cash_reserves": 40000,
    "documents": ["income_verification", "identity", "property_valuation", "mortgage_statement", "insurance", "credit_authorization"],
}

HIGH_RISK_APPLICATION = {
    **BASE_APPLICATION, "applicant_name": "High Risk", "product_type": "home_equity_loan", "annual_income": 60000, "verified_monthly_income": 4800,
    "employment_years": 0.5, "monthly_debt": 2600, "credit_score": 615, "credit_history_years": 2, "delinquencies_24m": 3, "inquiries_6m": 7,
    "property_value": 300000, "mortgage_balance": 240000, "requested_amount": 40000, "occupancy_type": "investment", "cash_reserves": 500,
    "documents": ["identity"],
}


def application(**overrides) -> ApplicationInput:
    return ApplicationInput(**{**BASE_APPLICATION, **overrides})


def form_data(**overrides) -> dict:
    """The same case as form fields (strings), as a browser would post it."""
    data = {**BASE_APPLICATION, **overrides}
    return {k: (v if isinstance(v, list) else str(v)) for k, v in data.items()}
