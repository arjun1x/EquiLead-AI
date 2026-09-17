"""Deterministic synthetic HELOC / home-equity dataset generator.

EquiLead ships no real borrower data. This generator produces applicant-like records with realistic
relationships between income, property value, mortgage balance, CLTV, DTI, credit history and simulated
repayment risk. Outcomes are simulated from a documented logistic relationship and must never be read as
real lending performance.

Protected attributes (``sex``, ``age_band``) are generated ONLY for fairness testing. They are stored in
separate columns, are never part of the model feature contract, and the generator gives them modest
correlations with legitimate inputs (age with history length; a small income gap by sex) so that the
fairness dashboard has something honest to monitor.

Dedicated to the public domain (CC0 1.0) together with any file it produces.

Command line:  python synth_data.py [--rows 12000] [--seed 20260916] [--output heloc_synthetic.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from calculations import ApplicationInput, DOCUMENT_ITEMS, derive_metrics, feature_row, MODEL_FEATURES

GENERATOR_VERSION = "synthetic-heloc-v1"
DEFAULT_ROWS = 12_000
DEFAULT_SEED = 20_260_916
TARGET = "defaulted_24m"
PROTECTED_COLUMNS = ["sex", "age_band"]
RAW_COLUMNS = ["product_type", "annual_income", "verified_monthly_income", "employment_status", "employment_years",
               "monthly_debt", "credit_score", "credit_history_years", "delinquencies_24m", "inquiries_6m",
               "property_value", "mortgage_balance", "requested_amount", "occupancy_type", "property_type",
               "loan_purpose", "term_years", "interest_rate", "cash_reserves", "documents"]


def _choice(rng, options, p, size):
    return rng.choice(list(options), size=size, p=list(p))


def generate(rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Return a DataFrame with raw application columns, derived model features, protected columns and the target."""
    if rows < 500:
        raise ValueError("At least 500 rows are needed for a meaningful split.")
    rng = np.random.default_rng(seed)
    n = rows

    # Protected attributes for monitoring only.
    sex = _choice(rng, ["female", "male"], [0.49, 0.51], n)
    age = np.clip(rng.normal(47, 12, n), 22, 85).round()
    age_band = np.select([age < 35, age < 50, age < 65], ["22-34", "35-49", "50-64"], default="65+")

    # Income: lognormal, with a documented small gap by sex so fairness metrics are non-trivial.
    annual_income = np.exp(rng.normal(11.35, 0.45, n)) * np.where(sex == "female", 0.93, 1.0)
    annual_income = np.clip(annual_income, 18_000, 900_000).round(-2)
    verified_share = rng.uniform(0.85, 1.05, n)
    verified_monthly = np.where(rng.random(n) < 0.82, annual_income / 12 * verified_share, 0.0).round(0)

    employment_status = _choice(rng, ["employed", "self_employed", "retired", "unemployed", "other"], [0.68, 0.14, 0.12, 0.03, 0.03], n)
    employment_status = np.where((age >= 66) & (rng.random(n) < 0.7), "retired", employment_status)
    employment_years = np.clip(rng.gamma(2.0, 3.5, n) * (0.6 + age / 100), 0, np.maximum(age - 18, 0)).round(1)
    employment_years = np.where(employment_status == "unemployed", 0.0, employment_years)

    credit_history_years = np.clip(rng.gamma(3.0, 4.0, n) * (0.5 + age / 90), 0.5, np.maximum(age - 18, 1)).round(1)
    credit_score = np.clip(rng.normal(705, 58, n) + 1.2 * (credit_history_years - 12), 300, 850).round().astype(int)
    delinquencies = rng.poisson(np.clip((720 - credit_score) / 90, 0.02, 3.5)).astype(int)
    inquiries = rng.poisson(np.where(credit_score < 660, 2.2, 1.1)).astype(int)

    property_value = np.clip(annual_income * rng.uniform(2.5, 6.0, n) * (1 + 0.15 * rng.standard_normal(n)), 60_000, 6_000_000).round(-3)
    ltv = np.clip(rng.beta(2.6, 2.4, n) * 0.95, 0.0, 0.98)
    ltv = np.where(age >= 60, ltv * 0.7, ltv)
    mortgage_balance = (property_value * ltv).round(-2)
    occupancy = _choice(rng, ["primary", "second_home", "investment"], [0.86, 0.06, 0.08], n)
    property_type = _choice(rng, ["single_family", "condo", "townhouse", "multi_family", "manufactured"], [0.66, 0.15, 0.11, 0.05, 0.03], n)
    purpose = _choice(rng, ["home_improvement", "debt_consolidation", "education", "major_purchase", "emergency_reserve", "other"], [0.38, 0.30, 0.07, 0.11, 0.08, 0.06], n)
    product = _choice(rng, ["heloc", "home_equity_loan"], [0.62, 0.38], n)

    cap = np.select([occupancy == "primary", occupancy == "second_home"], [0.85, 0.80], default=0.75)
    room = np.maximum(property_value * cap - mortgage_balance, 0)
    requested = np.clip(room * rng.uniform(0.15, 1.15, n) + rng.uniform(5_000, 25_000, n), 5_000, 2_500_000).round(-2)
    term = _choice(rng, [10, 15, 20, 25, 30], [0.18, 0.27, 0.30, 0.10, 0.15], n)
    rate = np.clip(6.0 + (760 - credit_score) / 40 + np.where(product == "heloc", 0.6, 0.0) + rng.normal(0, 0.6, n), 4.5, 16.0).round(2)

    monthly_debt = np.clip((mortgage_balance * 0.0065) + annual_income / 12 * rng.uniform(0.02, 0.22, n) + rng.normal(0, 150, n), 0, None).round(0)
    reserves = np.clip(np.exp(rng.normal(9.2, 1.1, n)) * (0.8 + age / 100), 0, 3_000_000).round(-2)

    doc_keys = list(DOCUMENT_ITEMS)
    doc_count = np.clip(rng.binomial(6, 0.82, n), 0, 6)

    # Row-wise derivation through the shared calculator (never a parallel formula).
    records, features = [], []
    for i in range(n):
        docs = doc_keys[: int(doc_count[i])]
        app = ApplicationInput(product_type=product[i], annual_income=float(annual_income[i]), verified_monthly_income=float(verified_monthly[i]),
                               employment_status=employment_status[i], employment_years=float(employment_years[i]),
                               monthly_debt=float(monthly_debt[i]), credit_score=int(credit_score[i]),
                               credit_history_years=float(credit_history_years[i]), delinquencies_24m=int(delinquencies[i]),
                               inquiries_6m=int(inquiries[i]), property_value=float(property_value[i]),
                               mortgage_balance=float(mortgage_balance[i]), requested_amount=float(requested[i]),
                               occupancy_type=occupancy[i], property_type=property_type[i], loan_purpose=purpose[i],
                               term_years=int(term[i]), interest_rate=float(rate[i]), cash_reserves=float(reserves[i]), documents=docs)
        derived = derive_metrics(app)
        raw = app.model_dump()
        raw["documents"] = "|".join(docs)
        records.append(raw)
        features.append(feature_row(app, derived))

    frame = pd.DataFrame(records)
    feats = pd.DataFrame(features)
    for column in MODEL_FEATURES:
        if column not in frame.columns:
            frame[column] = feats[column]

    # Simulated 24-month serious-delinquency outcome from a documented logistic relationship.
    dti = feats["dti_after"].clip(0, 2.0).to_numpy()
    cltv = feats["cltv_after"].clip(0, 1.5).to_numpy()
    reserves_m = feats["reserves_months"].clip(0, 60).to_numpy()
    linear = (-6.4
              + 3.2 * np.maximum(dti - 0.30, 0) * 4
              + 2.6 * np.maximum(cltv - 0.70, 0) * 4
              - 0.012 * (credit_score - 700)
              + 0.55 * np.minimum(delinquencies, 4)
              + 0.18 * np.minimum(inquiries, 6)
              - 0.06 * np.minimum(employment_years, 15)
              - 0.05 * np.minimum(reserves_m, 24)
              + np.where(occupancy == "investment", 0.55, np.where(occupancy == "second_home", 0.25, 0.0))
              + np.where(purpose == "debt_consolidation", 0.30, 0.0)
              + np.where(employment_status == "unemployed", 0.8, 0.0)
              - 0.5 * feats["doc_completeness"].to_numpy()
              + rng.normal(0, 0.55, n))
    probability = 1 / (1 + np.exp(-linear))
    frame[TARGET] = rng.binomial(1, probability)
    frame["true_probability"] = probability.round(4)
    frame["sex"] = sex
    frame["age_band"] = age_band
    frame["generator_version"] = GENERATOR_VERSION
    return frame


def write_csv(path: Path, rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> Path:
    frame = generate(rows, seed)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description="Generate the synthetic HELOC dataset")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default="heloc_synthetic.csv")
    args = parser.parse_args()
    out = write_csv(Path(args.output), args.rows, args.seed)
    frame = pd.read_csv(out)
    print(f"Wrote {len(frame)} synthetic rows to {out} (target rate {frame[TARGET].mean():.1%}).")
