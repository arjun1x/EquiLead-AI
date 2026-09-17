"""Home-equity financial calculations and input validation.

This is the single source of truth for every derived number (LTV, CLTV, DTI, payment, reserves,
available equity). Routes, the synthetic data generator, training and scoring all call
``derive_metrics`` so no formula is duplicated anywhere else.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CALCULATION_VERSION = "calc-v1"

PRODUCT_TYPES = {"heloc": "Home equity line of credit", "home_equity_loan": "Home equity loan"}
EMPLOYMENT_STATUS = {"employed": "Employed", "self_employed": "Self-employed", "retired": "Retired",
                     "unemployed": "Unemployed", "other": "Other"}
OCCUPANCY_TYPES = {"primary": "Primary residence", "second_home": "Second home", "investment": "Investment property"}
PROPERTY_TYPES = {"single_family": "Single-family", "condo": "Condominium", "townhouse": "Townhouse",
                  "multi_family": "2–4 unit", "manufactured": "Manufactured home"}
LOAN_PURPOSES = {"home_improvement": "Home improvement", "debt_consolidation": "Debt consolidation",
                 "education": "Education", "major_purchase": "Major purchase", "emergency_reserve": "Emergency reserve",
                 "other": "Other"}
DOCUMENT_ITEMS = {"income_verification": "Income verification (pay stubs, W-2 or tax returns)",
                  "identity": "Government identity", "property_valuation": "Property valuation or AVM",
                  "mortgage_statement": "Current mortgage statement", "insurance": "Homeowner insurance",
                  "credit_authorization": "Credit report authorization"}

# Maximum combined loan-to-value by occupancy (policy assumption for the demonstration, not a legal limit).
MAX_CLTV = {"primary": 0.85, "second_home": 0.80, "investment": 0.75}
RESERVE_MONTHS_CAP = 120.0

# Model feature contract. Protected attributes are never part of it.
NUMERIC_FEATURES = ["credit_score", "credit_history_years", "delinquencies_24m", "inquiries_6m", "employment_years",
                    "monthly_income", "monthly_debt", "dti_after", "cltv_after", "ltv_current", "requested_amount",
                    "property_value", "mortgage_balance", "reserves_months", "term_years", "interest_rate",
                    "doc_completeness"]
CATEGORICAL_FEATURES = ["product_type", "employment_status", "occupancy_type", "property_type", "loan_purpose"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FEATURE_LABELS = {
    "credit_score": "Credit score", "credit_history_years": "Credit history length", "delinquencies_24m": "Delinquencies (24 months)",
    "inquiries_6m": "Credit inquiries (6 months)", "employment_years": "Employment length", "monthly_income": "Monthly income",
    "monthly_debt": "Existing monthly debt", "dti_after": "Debt-to-income after loan", "cltv_after": "Combined LTV after loan",
    "ltv_current": "Current LTV", "requested_amount": "Requested amount", "property_value": "Property value",
    "mortgage_balance": "Existing mortgage balance", "reserves_months": "Cash reserves (months)", "term_years": "Requested term",
    "interest_rate": "Interest-rate assumption", "doc_completeness": "Documentation completeness",
    "product_type": "Product type", "employment_status": "Employment status", "occupancy_type": "Occupancy",
    "property_type": "Property type", "loan_purpose": "Loan purpose",
}
FEATURE_FORMATS = {"dti_after": "pct", "cltv_after": "pct", "ltv_current": "pct", "doc_completeness": "pct",
                   "monthly_income": "money", "monthly_debt": "money", "requested_amount": "money",
                   "property_value": "money", "mortgage_balance": "money", "interest_rate": "rate",
                   "cash_reserves": "money", "annual_income": "money", "verified_monthly_income": "money",
                   "gross_equity": "money", "available_equity": "money", "estimated_payment": "money",
                   "credit_history_years": "years", "employment_years": "years", "reserves_months": "months",
                   "term_years": "years"}


class ApplicationInput(BaseModel):
    """Raw application fields as entered by a loan officer. Ranges reject impossible values;
    plausible-but-unusual values are flagged by ``derive_metrics`` instead."""
    model_config = ConfigDict(allow_inf_nan=False, str_strip_whitespace=True, extra="ignore")

    applicant_name: str = Field(default="", max_length=120, description="Free text; required at intake, blank in synthetic training rows")
    product_type: Literal["heloc", "home_equity_loan"] = "heloc"
    annual_income: float = Field(ge=0, le=10_000_000, description="Stated gross annual income")
    verified_monthly_income: float = Field(default=0, ge=0, le=1_000_000, description="Documented monthly income, 0 if not yet verified")

    @field_validator("verified_monthly_income", mode="before")
    @classmethod
    def _blank_means_unverified(cls, value):
        return 0 if value in (None, "") else value
    employment_status: Literal["employed", "self_employed", "retired", "unemployed", "other"] = "employed"
    employment_years: float = Field(default=0, ge=0, le=60)
    monthly_debt: float = Field(ge=0, le=500_000, description="Existing monthly obligations including the current mortgage payment")
    credit_score: int = Field(ge=300, le=850)
    credit_history_years: float = Field(default=0, ge=0, le=80)
    delinquencies_24m: int = Field(default=0, ge=0, le=99)
    inquiries_6m: int = Field(default=0, ge=0, le=99)
    property_value: float = Field(gt=0, le=100_000_000)
    mortgage_balance: float = Field(ge=0, le=100_000_000)
    requested_amount: float = Field(gt=0, le=10_000_000)
    occupancy_type: Literal["primary", "second_home", "investment"] = "primary"
    property_type: Literal["single_family", "condo", "townhouse", "multi_family", "manufactured"] = "single_family"
    loan_purpose: Literal["home_improvement", "debt_consolidation", "education", "major_purchase", "emergency_reserve", "other"] = "home_improvement"
    term_years: int = Field(default=20, ge=5, le=30)
    interest_rate: float = Field(default=8.5, ge=0.5, le=30, description="Annual rate assumption in percent")
    cash_reserves: float = Field(default=0, ge=0, le=100_000_000)
    documents: list[str] = Field(default_factory=list)

    @field_validator("documents")
    @classmethod
    def _known_documents(cls, value):
        unknown = [d for d in value if d not in DOCUMENT_ITEMS]
        if unknown:
            raise ValueError(f"unknown document item(s): {', '.join(unknown)}")
        return sorted(set(value))

    @model_validator(mode="after")
    def _cross_checks(self):
        if self.employment_years > self.credit_history_years + 60:
            raise ValueError("employment length is not plausible against the credit history length")
        if self.mortgage_balance > self.property_value * 3:
            raise ValueError("mortgage balance is more than three times the property value; check the figures")
        if self.requested_amount > self.property_value * 2:
            raise ValueError("requested amount is more than twice the property value; check the figures")
        return self


class Flag(BaseModel):
    code: str
    severity: Literal["info", "warning", "blocking"]
    message: str


class DerivedMetrics(BaseModel):
    calculation_version: str = CALCULATION_VERSION
    monthly_income: float
    income_basis: Literal["verified", "stated", "none"]
    gross_equity: float
    max_cltv: float
    available_equity: float
    ltv_current: float
    proposed_balance: float
    cltv_after: float
    estimated_payment: float
    payment_basis: str
    dti_before: float | None
    dti_after: float | None
    total_monthly_obligations: float
    reserves_months: float | None
    doc_completeness: float
    missing_documents: list[str]
    requested_exceeds_equity: bool
    flags: list[Flag]

    def as_feature_values(self):
        """Derived numbers that feed the model feature vector."""
        return {"monthly_income": self.monthly_income, "dti_after": self.dti_after if self.dti_after is not None else 9.99,
                "cltv_after": self.cltv_after, "ltv_current": self.ltv_current,
                "reserves_months": self.reserves_months if self.reserves_months is not None else RESERVE_MONTHS_CAP,
                "doc_completeness": self.doc_completeness}


def safe_ratio(numerator, denominator):
    """Return numerator/denominator or None when the denominator is zero or not finite."""
    try:
        if denominator is None or numerator is None or denominator == 0 or not math.isfinite(denominator) or not math.isfinite(numerator):
            return None
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return None


def amortized_payment(principal, annual_rate_percent, years):
    """Standard level-payment formula. A zero rate degrades to straight-line repayment."""
    months = max(1, int(round(years * 12)))
    if principal <= 0:
        return 0.0
    r = annual_rate_percent / 100.0 / 12.0
    if r <= 0:
        return principal / months
    factor = (1 + r) ** months
    return principal * r * factor / (factor - 1)


def interest_only_payment(principal, annual_rate_percent):
    return max(0.0, principal) * annual_rate_percent / 100.0 / 12.0


def derive_metrics(app: ApplicationInput) -> DerivedMetrics:
    flags: list[Flag] = []

    # Income basis: documented income wins; stated income is used with a warning; no income is blocking.
    if app.verified_monthly_income > 0:
        monthly_income, basis = app.verified_monthly_income, "verified"
        stated_monthly = app.annual_income / 12 if app.annual_income else 0
        if stated_monthly and monthly_income < 0.7 * stated_monthly:
            flags.append(Flag(code="INCOME_SHORTFALL", severity="warning",
                              message=f"Verified income (${monthly_income:,.0f}/mo) is well below stated income (${stated_monthly:,.0f}/mo)."))
    elif app.annual_income > 0:
        monthly_income, basis = app.annual_income / 12, "stated"
        flags.append(Flag(code="INCOME_UNVERIFIED", severity="warning", message="Income has not been verified; ratios use stated income."))
    else:
        monthly_income, basis = 0.0, "none"
        flags.append(Flag(code="NO_INCOME", severity="blocking", message="No income was provided, so debt-to-income cannot be calculated."))
    if monthly_income > 100_000:
        flags.append(Flag(code="INCOME_OUTLIER", severity="warning", message="Monthly income is far outside the range the models were trained on."))

    # Equity and loan-to-value.
    gross_equity = app.property_value - app.mortgage_balance
    max_cltv = MAX_CLTV[app.occupancy_type]
    available_equity = max(0.0, app.property_value * max_cltv - app.mortgage_balance)
    ltv_current = safe_ratio(app.mortgage_balance, app.property_value) or 0.0
    proposed_balance = app.mortgage_balance + app.requested_amount
    cltv_after = safe_ratio(proposed_balance, app.property_value) or 0.0
    requested_exceeds_equity = app.requested_amount > available_equity + 1e-9
    if gross_equity < 0:
        flags.append(Flag(code="UNDERWATER", severity="blocking", message="The existing mortgage balance exceeds the property value."))
    if requested_exceeds_equity:
        flags.append(Flag(code="EXCEEDS_EQUITY", severity="blocking",
                          message=f"Requested ${app.requested_amount:,.0f} exceeds the ${available_equity:,.0f} available at {max_cltv:.0%} CLTV."))
    if app.property_value > 20_000_000:
        flags.append(Flag(code="VALUE_OUTLIER", severity="warning", message="Property value is far outside the range the models were trained on."))

    # Qualifying payment: fully amortizing over the requested term even for a HELOC, which is conservative.
    payment = amortized_payment(app.requested_amount, app.interest_rate, app.term_years)
    if app.product_type == "heloc":
        io_payment = interest_only_payment(app.requested_amount, app.interest_rate)
        payment_basis = f"fully amortizing over {app.term_years} years at {app.interest_rate:.2f}% (interest-only draw payment ${io_payment:,.0f})"
    else:
        payment_basis = f"fully amortizing over {app.term_years} years at {app.interest_rate:.2f}%"

    total_obligations = app.monthly_debt + payment
    dti_before = safe_ratio(app.monthly_debt, monthly_income)
    dti_after = safe_ratio(total_obligations, monthly_income)
    if dti_after is not None and dti_after > 1.0:
        flags.append(Flag(code="DTI_OVER_100", severity="warning", message="Obligations after the loan exceed monthly income."))

    reserves = safe_ratio(app.cash_reserves, total_obligations)
    if reserves is not None:
        reserves = min(reserves, RESERVE_MONTHS_CAP)

    provided = [d for d in app.documents if d in DOCUMENT_ITEMS]
    missing = [d for d in DOCUMENT_ITEMS if d not in provided]
    completeness = len(provided) / len(DOCUMENT_ITEMS)
    if missing:
        flags.append(Flag(code="DOCS_INCOMPLETE", severity="warning", message=f"{len(missing)} required document(s) not yet received."))

    if app.credit_score < 620:
        flags.append(Flag(code="SCORE_BELOW_FLOOR", severity="info", message="Credit score is below the program floor of 620."))
    if app.employment_status == "unemployed" and basis != "none":
        flags.append(Flag(code="UNEMPLOYED_WITH_INCOME", severity="warning", message="Applicant is unemployed but reports income; confirm the income source."))

    return DerivedMetrics(monthly_income=monthly_income, income_basis=basis, gross_equity=gross_equity, max_cltv=max_cltv,
                          available_equity=available_equity, ltv_current=ltv_current, proposed_balance=proposed_balance,
                          cltv_after=cltv_after, estimated_payment=payment, payment_basis=payment_basis,
                          dti_before=dti_before, dti_after=dti_after, total_monthly_obligations=total_obligations,
                          reserves_months=reserves, doc_completeness=completeness, missing_documents=missing,
                          requested_exceeds_equity=requested_exceeds_equity, flags=flags)


def feature_row(app: ApplicationInput, derived: DerivedMetrics | None = None) -> dict:
    """The model feature vector for one application. Derived values are always recomputed here,
    so a stored ratio can never disagree with the raw inputs (training-serving skew)."""
    derived = derived or derive_metrics(app)
    row = {"credit_score": app.credit_score, "credit_history_years": app.credit_history_years,
           "delinquencies_24m": app.delinquencies_24m, "inquiries_6m": app.inquiries_6m,
           "employment_years": app.employment_years, "monthly_debt": app.monthly_debt,
           "requested_amount": app.requested_amount, "property_value": app.property_value,
           "mortgage_balance": app.mortgage_balance, "term_years": app.term_years, "interest_rate": app.interest_rate,
           "product_type": app.product_type, "employment_status": app.employment_status,
           "occupancy_type": app.occupancy_type, "property_type": app.property_type, "loan_purpose": app.loan_purpose}
    row.update(derived.as_feature_values())
    return {name: row[name] for name in MODEL_FEATURES}


def format_value(name, value):
    """Human formatting for a feature value, shared by templates, letters and explanations."""
    if value is None:
        return "—"
    kind = FEATURE_FORMATS.get(name)
    if name in CATEGORICAL_FEATURES:
        table = {"product_type": PRODUCT_TYPES, "employment_status": EMPLOYMENT_STATUS, "occupancy_type": OCCUPANCY_TYPES,
                 "property_type": PROPERTY_TYPES, "loan_purpose": LOAN_PURPOSES}[name]
        return table.get(value, str(value))
    if kind == "pct":
        return f"{float(value) * 100:.1f}%"
    if kind == "money":
        return f"${float(value):,.0f}"
    if kind == "rate":
        return f"{float(value):.2f}%"
    if kind == "years":
        return f"{float(value):.1f} years"
    if kind == "months":
        return f"{float(value):.1f} months"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value)}"
    return f"{value}"
