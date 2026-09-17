"""Fairness and responsible-AI monitoring metrics.

Computed on synthetic evaluation data with protected attributes that are stored beside, never inside, the
model feature vector. Every result is a monitoring signal for human and legal review. Passing a metric does
not make a model "fair" in any legal sense, and small groups are flagged as unreliable.

Command line:  python fairness.py   (prints the training-time fairness evaluation from training_report.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MIN_GROUP_N = 30
CAUTION_GROUP_N = 100
FOUR_FIFTHS = 0.80
PARITY_TOLERANCE = 0.10
ERROR_GAP_TOLERANCE = 0.10

PROTECTED_CLASSES = ["race", "color", "religion", "national origin", "sex", "marital status", "age",
                     "receipt of public assistance", "good-faith exercise of consumer credit rights"]
GOVERNANCE = {
    "model_inputs": "Protected attributes and obvious proxies are excluded from the model feature contract (calculations.MODEL_FEATURES).",
    "test_data": "Group membership for testing comes from a separate column set produced by the synthetic generator or, in production, "
                 "from a governed monitoring dataset. It is never captured on the application form.",
    "tests": "Selection rate, demographic parity difference, disparate-impact ratio, false-positive and false-negative rates and "
             "equal-opportunity difference per model, with sample-size warnings, at every training run.",
    "review": "Signals outside tolerance require fair-lending, model-risk and legal review before a release; they are not a verdict.",
}


def group_metrics(y_true, y_prob, groups, selected_cutoff: float) -> dict:
    """Per-group selection (approval) rate and error rates.
    A 'selection' is a favourable recommendation: probability below ``selected_cutoff``.
    Positive class for error rates is a default (label 1); a non-selection is the positive prediction."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    groups = np.asarray(groups).astype(str)
    selected = y_prob < selected_cutoff
    predicted_default = ~selected
    out = {}
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        n = int(mask.sum())
        yt, pdft, sel = y_true[mask], predicted_default[mask], selected[mask]
        positives, negatives = int((yt == 1).sum()), int((yt == 0).sum())
        tp = int(((yt == 1) & pdft).sum()); fn = int(((yt == 1) & ~pdft).sum())
        fp = int(((yt == 0) & pdft).sum()); tn = int(((yt == 0) & ~pdft).sum())
        out[name] = {"n": n, "selection_rate": float(sel.mean()) if n else None, "actual_default_rate": float(yt.mean()) if n else None,
                     "fpr": fp / negatives if negatives else None, "fnr": fn / positives if positives else None,
                     "tpr": tp / positives if positives else None, "tnr": tn / negatives if negatives else None,
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "warning": "too small for a reliable conclusion" if n < MIN_GROUP_N else ("small sample; interpret with caution" if n < CAUTION_GROUP_N else None)}
    return out


def summarise(groups: dict) -> dict:
    """Demographic parity difference, disparate-impact ratio, equal-opportunity difference, error gaps."""
    eligible = {k: v for k, v in groups.items() if v["n"] >= MIN_GROUP_N and v["selection_rate"] is not None}
    if len(eligible) < 2:
        return {"status": "INSUFFICIENT", "reference": None, "demographic_parity_difference": None, "disparate_impact_ratio": None,
                "equal_opportunity_difference": None, "fpr_gap": None, "fnr_gap": None, "notes": ["Fewer than two groups have enough records."]}
    rates = {k: v["selection_rate"] for k, v in eligible.items()}
    reference = max(rates, key=rates.get)
    dpd = max(rates.values()) - min(rates.values())
    dir_ratio = (min(rates.values()) / rates[reference]) if rates[reference] else None
    tprs = {k: v["tpr"] for k, v in eligible.items() if v["tpr"] is not None}
    fprs = {k: v["fpr"] for k, v in eligible.items() if v["fpr"] is not None}
    fnrs = {k: v["fnr"] for k, v in eligible.items() if v["fnr"] is not None}
    eod = (max(tprs.values()) - min(tprs.values())) if len(tprs) >= 2 else None
    fpr_gap = (max(fprs.values()) - min(fprs.values())) if len(fprs) >= 2 else None
    fnr_gap = (max(fnrs.values()) - min(fnrs.values())) if len(fnrs) >= 2 else None
    notes = []
    status = "WITHIN_TOLERANCE"
    if dir_ratio is not None and dir_ratio < FOUR_FIFTHS:
        status = "REVIEW"; notes.append(f"Disparate-impact ratio {dir_ratio:.2f} is below the four-fifths guideline.")
    if dpd > PARITY_TOLERANCE:
        status = "REVIEW"; notes.append(f"Selection-rate difference {dpd:.2f} exceeds {PARITY_TOLERANCE:.2f}.")
    if eod is not None and eod > ERROR_GAP_TOLERANCE:
        status = "REVIEW"; notes.append(f"Equal-opportunity difference {eod:.2f} exceeds {ERROR_GAP_TOLERANCE:.2f}.")
    if fpr_gap is not None and fpr_gap > ERROR_GAP_TOLERANCE:
        status = "REVIEW"; notes.append(f"False-positive-rate gap {fpr_gap:.2f} exceeds {ERROR_GAP_TOLERANCE:.2f}.")
    small = [k for k, v in groups.items() if v["n"] < MIN_GROUP_N]
    if small:
        notes.append(f"Excluded small groups: {', '.join(small)}.")
    return {"status": status, "reference": reference, "demographic_parity_difference": dpd, "disparate_impact_ratio": dir_ratio,
            "equal_opportunity_difference": eod, "fpr_gap": fpr_gap, "fnr_gap": fnr_gap, "notes": notes}


def evaluate(y_true, y_prob, protected: dict, selected_cutoff: float) -> dict:
    """protected: {"sex": array, "age_band": array}. Returns {attribute: {"groups":..., "summary":...}}."""
    return {attribute: {"groups": (g := group_metrics(y_true, y_prob, values, selected_cutoff)), "summary": summarise(g)}
            for attribute, values in protected.items()}


def compare_models(per_model: dict) -> list[dict]:
    """Flatten {model_key: evaluate()} into rows for a comparison table."""
    rows = []
    for model_key, attributes in per_model.items():
        for attribute, block in attributes.items():
            s = block["summary"]
            rows.append({"model": model_key, "attribute": attribute, "status": s["status"], "dpd": s["demographic_parity_difference"],
                         "dir": s["disparate_impact_ratio"], "eod": s["equal_opportunity_difference"], "fpr_gap": s["fpr_gap"], "fnr_gap": s["fnr_gap"]})
    return rows


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    report_path = ROOT / "training_report.json"
    if not report_path.exists():
        sys.exit("training_report.json not found. Run: python train.py")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print("Fairness evaluation on the synthetic held-out test split (monitoring signals only)\n")
    for row in compare_models(report["fairness"]):
        print(f"{row['model']:>10} {row['attribute']:>9}  status={row['status']:<16} DPD={row['dpd']!s:>8} DIR={row['dir']!s:>8} EOD={row['eod']!s:>8}")
