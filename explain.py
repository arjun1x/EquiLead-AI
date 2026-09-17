"""Local explanations for each model.

* Tree models (random forest, gradient boosting): SHAP TreeExplainer when the shap package works.
* Scorecard (logistic regression): exact coefficient × standardized-value contributions.
* Fallback for either: reference-perturbation sensitivity (feature set to its training reference value).

One-hot columns are folded back into the original application field, so a driver always names something a
loan officer can see on the form. Explanations are computed once per scoring run and stored with the run.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from calculations import CATEGORICAL_FEATURES, FEATURE_LABELS, MODEL_FEATURES, NUMERIC_FEATURES, format_value

try:  # shap is optional at runtime; scoring must not fail without it
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    shap = None
    SHAP_AVAILABLE = False

TOP_N = 5


def _base_feature(name: str) -> str:
    name = name.split("__", 1)[-1]
    for cat in CATEGORICAL_FEATURES:
        if name == cat or name.startswith(cat + "_"):
            return cat
    return name


def _fold(names, contributions):
    totals = {}
    for name, value in zip(names, contributions):
        base = _base_feature(name)
        totals[base] = totals.get(base, 0.0) + float(value)
    return totals


def _tree_shap(classifier, transformed):
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:          # (n, features, classes) in newer shap versions
        values = values[:, :, -1]
    return values[0]


def _linear_contributions(classifier, transformed):
    return classifier.coef_[0] * transformed[0]


def _perturbation(pipeline, frame, reference_values):
    """Fallback: probability change when a feature is replaced by its training reference value."""
    base = float(pipeline.predict_proba(frame)[0][1])
    contributions = {}
    for feature in MODEL_FEATURES:
        if feature not in reference_values:
            continue
        altered = frame.copy()
        altered.loc[:, feature] = reference_values[feature]
        contributions[feature] = base - float(pipeline.predict_proba(altered)[0][1])
    return contributions


def explain(entry: dict, frame: pd.DataFrame, feature_values: dict) -> dict:
    """Return {"method", "drivers_positive", "drivers_negative", "all", "note"} for one model entry
    ({"key", "pipeline", "meta"}) and one-row feature frame."""
    pipeline = entry["pipeline"]
    kind = entry["meta"].get("algorithm_family", "tree")
    method, note = None, ""
    try:
        prep = pipeline.named_steps["prep"]
        clf = pipeline.named_steps["clf"]
        transformed = np.asarray(prep.transform(frame))
        names = list(prep.get_feature_names_out())
        if kind == "linear":
            folded = _fold(names, _linear_contributions(clf, transformed))
            method = "coefficient contributions (log-odds)"
        elif SHAP_AVAILABLE:
            folded = _fold(names, _tree_shap(clf, transformed))
            units = "probability" if "RandomForest" in type(clf).__name__ else "log-odds"
            method = f"SHAP TreeExplainer ({units})"
        else:
            raise RuntimeError("shap unavailable")
    except Exception as exc:  # any explainer failure degrades to the perturbation fallback
        folded = _perturbation(pipeline, frame, entry["meta"].get("reference_values", {}))
        method = "reference-perturbation sensitivity (probability change)"
        note = f"Primary explainer unavailable ({type(exc).__name__}); fallback used."
    rows = [{"feature": f, "label": FEATURE_LABELS.get(f, f), "contribution": round(v, 5),
             "observed": feature_values.get(f), "observed_text": format_value(f, feature_values.get(f))}
            for f, v in folded.items() if math.isfinite(v)]
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    positive = [r for r in rows if r["contribution"] > 0][:TOP_N]
    negative = [r for r in rows if r["contribution"] < 0][:TOP_N]
    return {"method": method, "note": note, "drivers_positive": positive, "drivers_negative": negative, "all": rows}


# ── Counterfactual guidance ─────────────────────────────

COUNTERFACTUAL_LEVERS = [
    # raw input, direction, step, floor/ceiling, label
    ("requested_amount", -1, 0.05, 0.10, "Reduce the requested amount"),
    ("monthly_debt", -1, 0.10, 0.0, "Reduce existing monthly debt"),
    ("credit_score", +1, 10, 850, "Raise the credit score"),
    ("cash_reserves", +1, 0.25, 12.0, "Increase cash reserves"),
    ("delinquencies_24m", -1, 1, 0, "Resolve reported delinquencies"),
    ("inquiries_6m", -1, 1, 0, "Reduce recent credit inquiries"),
]


def counterfactuals(app_input, current_probability: float, probability_fn, band_for, max_steps: int = 20) -> list[dict]:
    """Estimate the single-input change that would move the consensus probability into the next better band.
    ``probability_fn(list_of_ApplicationInput) -> list[float]``. Returns estimates, clearly labelled."""
    from calculations import ApplicationInput  # local import keeps this module import-light
    current_band = band_for(current_probability)["band"]
    if current_band == "A":
        return []
    target_upper = next(b["upper"] for b in _bands_iter(band_for) if b["band"] == current_band)
    previous = [b for b in _bands_iter(band_for) if b["upper"] < target_upper]
    target = previous[-1]["upper"] if previous else 0
    out = []
    base = app_input.model_dump()
    for field, direction, step, limit, label in COUNTERFACTUAL_LEVERS:
        start = base[field]
        candidates, values = [], []
        for i in range(1, max_steps + 1):
            if field in ("requested_amount", "monthly_debt"):
                value = start * (1 - step * i)
                if value < (start * limit if field == "requested_amount" else limit):
                    break
            elif field == "cash_reserves":
                value = start + step * i * max(base["monthly_debt"], 500) * 4
                if value > limit * 50_000 + start * 3:
                    break
            elif field == "credit_score":
                value = start + step * i
                if value > limit:
                    break
            else:
                value = start - step * i
                if value < limit:
                    break
            data = dict(base)
            data[field] = int(round(value)) if field in ("credit_score", "delinquencies_24m", "inquiries_6m") else round(value, 2)
            try:
                candidates.append(ApplicationInput(**data))
                values.append(data[field])
            except Exception:
                break
        if not candidates or values[0] == start:
            continue
        probabilities = probability_fn(candidates)
        for value, probability in zip(values, probabilities):
            if probability < target:
                out.append({"lever": label, "field": field, "from": start, "to": value,
                            "from_text": format_value(field, start), "to_text": format_value(field, value),
                            "estimated_probability": round(float(probability), 4),
                            "estimated_band": band_for(probability)["band"], "current_band": current_band})
                break
    return out


def _bands_iter(band_for):
    from policy import BANDS
    return BANDS
