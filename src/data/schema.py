"""
Feature schema for EquiLend AI loan decisioning model.

Aligned with the HMEQ (Home Equity) dataset.
"""

from dataclasses import dataclass
from typing import Literal

FeatureType = Literal["numeric", "categorical"]


@dataclass
class Feature:
    name: str
    dtype: FeatureType
    description: str
    nullable: bool = False


FEATURE_SCHEMA: list[Feature] = [
    Feature("LOAN",    "numeric",      "Loan amount requested"),
    Feature("MORTDUE", "numeric",      "Amount due on existing mortgage",        nullable=True),
    Feature("VALUE",   "numeric",      "Current property value",                 nullable=True),
    Feature("REASON",  "categorical",  "Loan reason: HomeImp or DebtCon",        nullable=True),
    Feature("JOB",     "categorical",  "Job category: Mgr, Office, Other, ProfExe, Sales, Self", nullable=True),
    Feature("YOJ",     "numeric",      "Years at present job",                   nullable=True),
    Feature("DEROG",   "numeric",      "Number of major derogatory reports",     nullable=True),
    Feature("DELINQ",  "numeric",      "Number of delinquent credit lines",      nullable=True),
    Feature("CLAGE",   "numeric",      "Age of oldest credit line (months)",     nullable=True),
    Feature("NINQ",    "numeric",      "Number of recent credit inquiries",      nullable=True),
    Feature("CLNO",    "numeric",      "Number of existing credit lines",        nullable=True),
    Feature("DEBTINC", "numeric",      "Debt-to-income ratio",                   nullable=True),
]

TARGET_COLUMN = "BAD"  # 1 = defaulted, 0 = repaid

MODEL_FEATURES: list[str] = [f.name for f in FEATURE_SCHEMA]

NUMERIC_FEATURES: list[str] = [f.name for f in FEATURE_SCHEMA if f.dtype == "numeric"]

CATEGORICAL_FEATURES: list[str] = [f.name for f in FEATURE_SCHEMA if f.dtype == "categorical"]

NULLABLE_FEATURES: list[str] = [f.name for f in FEATURE_SCHEMA if f.nullable]
