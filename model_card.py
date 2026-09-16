"""Model card: the version, data, training date, threshold and calibration facts shown with every decision.

Demo mode ships a synthetic card. Research mode fails closed unless MODEL_CARD_PATH points to a JSON file
with the required fields (see model_card.example.json)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = ["model_version", "dataset_version", "training_date", "threshold", "calibration_method"]
FEATURES = ["LOAN", "MORTDUE", "VALUE", "REASON", "JOB", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]

# Synthetic reference distributions for the demo drift monitor. Shaped like HMEQ, but not derived from it.
_DEMO_REFERENCE = {
    "numeric": {
        "LOAN": {"edges": [0, 10000, 20000, 30000, 45000, 70000, 10000000], "proportions": [0.18, 0.32, 0.22, 0.16, 0.08, 0.04]},
        "VALUE": {"edges": [0, 50000, 100000, 150000, 250000, 400000, 100000000], "proportions": [0.10, 0.32, 0.30, 0.18, 0.07, 0.03]},
        "MORTDUE": {"edges": [0, 25000, 50000, 90000, 150000, 250000, 100000000], "proportions": [0.12, 0.28, 0.31, 0.18, 0.08, 0.03]},
        "DEBTINC": {"edges": [0, 20, 30, 35, 40, 45, 1000], "proportions": [0.12, 0.30, 0.28, 0.16, 0.09, 0.05]},
        "CLAGE": {"edges": [0, 60, 120, 180, 240, 360, 1200], "proportions": [0.12, 0.24, 0.30, 0.18, 0.12, 0.04]},
        "YOJ": {"edges": [0, 2, 5, 10, 15, 25, 80], "proportions": [0.24, 0.26, 0.24, 0.14, 0.09, 0.03]},
        "DELINQ": {"edges": [0, 1, 2, 4, 1000], "proportions": [0.78, 0.12, 0.07, 0.03]},
        "DEROG": {"edges": [0, 1, 2, 1000], "proportions": [0.86, 0.09, 0.05]},
        "NINQ": {"edges": [0, 1, 2, 4, 1000], "proportions": [0.50, 0.24, 0.16, 0.10]},
        "CLNO": {"edges": [0, 10, 20, 30, 1000], "proportions": [0.22, 0.44, 0.26, 0.08]},
    },
    "categorical": {
        "REASON": {"HomeImp": 0.31, "DebtCon": 0.69},
        "JOB": {"Mgr": 0.13, "Office": 0.16, "Other": 0.42, "ProfExe": 0.22, "Sales": 0.02, "Self": 0.03, "Unknown": 0.02},
    },
    "score": {"edges": [0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0], "proportions": [0.38, 0.22, 0.14, 0.08, 0.10, 0.08]},
}

DEMO_CARD = {
    "is_demo": True,
    "model_version": "synthetic-fixture-v1",
    "model_type": "none (fixed demo fixtures, no model is evaluated)",
    "dataset_name": "none",
    "dataset_version": "synthetic-demo",
    "training_date": None,
    "threshold": 0.5,
    "score_type": "fixed synthetic score",
    "class_weighting": "not applicable",
    "calibration_method": "none",
    "calibration_status": "Synthetic fixture. The demo score is not a probability of any kind.",
    "features": FEATURES,
    "protected_attributes_in_model": [],
    "validation_metrics": {},
    "reason_code_version": "aa-codes-2026.09",
    "reason_codes_approved": False,
    "reason_codes_approved_on": None,
    "fair_lending_review": {"status": "not performed", "reviewer": None, "date": None,
                            "notes": "Demo fixtures are synthetic; no fair-lending validation applies to them."},
    "reference_stats": _DEMO_REFERENCE,
}


def load_model_card(demo=None, path=None):
    if demo is None:
        demo = os.getenv("DEMO_MODE", "1") == "1"
    if demo:
        return json.loads(json.dumps(DEMO_CARD))
    path = Path(path or os.getenv("MODEL_CARD_PATH", "model_card.json"))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise RuntimeError(f"Model card missing: add {path.name} (see model_card.example.json) before running with DEMO_MODE=0.")
    card = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED if key not in card]
    if missing:
        raise RuntimeError(f"Model card {path.name} is missing required fields: {', '.join(missing)}")
    if not 0 < float(card["threshold"]) < 1:
        raise RuntimeError("Model card threshold must be between 0 and 1.")
    card.setdefault("is_demo", False)
    card.setdefault("score_type", "class-weighted model score")
    card.setdefault("features", FEATURES)
    card.setdefault("reason_code_version", "aa-codes-2026.09")
    card.setdefault("reason_codes_approved", False)
    card.setdefault("reason_codes_approved_on", None)
    card.setdefault("fair_lending_review", {"status": "not performed", "reviewer": None, "date": None, "notes": ""})
    card.setdefault("reference_stats", {})
    card.setdefault("protected_attributes_in_model", [])
    if card["protected_attributes_in_model"]:
        raise RuntimeError("Model card lists protected attributes as model inputs. That model may not be used here.")
    return card


def calibration_label(card, calibrated_probability):
    """Never call the raw class-weighted score a probability."""
    if card.get("is_demo"):
        return "synthetic fixture (not a probability)"
    if calibrated_probability is None:
        return f"uncalibrated {card.get('score_type', 'model score')}"
    return f"{card.get('calibration_method', 'unknown')} calibrated probability"


def decision_metadata(card, calibrated_probability=None):
    return {
        "model_version": card["model_version"],
        "dataset_version": card["dataset_version"],
        "training_date": card.get("training_date"),
        "threshold": float(card["threshold"]),
        "calibration": calibration_label(card, calibrated_probability),
        "reason_code_version": card.get("reason_code_version"),
    }
