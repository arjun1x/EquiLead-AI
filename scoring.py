"""Three-model scoring engine with transparent consensus.

Models are loaded once (``load_models``) and reused. Each model returns a typed ``ModelResult``; the
consensus never hides disagreement and routes material disagreement to manual review. The underwriting
policy result is combined last, so every layer stays visible.
"""
from __future__ import annotations

import hashlib
import json
import logging
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from calculations import (ApplicationInput, CATEGORICAL_FEATURES, FEATURE_LABELS, MODEL_FEATURES, NUMERIC_FEATURES,
                          derive_metrics, feature_row, format_value)
from explain import counterfactuals, explain
from policy import (BANDS, BANDS_VERSION, BAND_CUTOFFS, DECISION_THRESHOLD, RECOMMENDATION_LABELS, band_for,
                    evaluate_policy, most_conservative, severity)
from reason_codes import assign_reason_codes

log = logging.getLogger("equilead.scoring")
ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "model_registry.json"
MODEL_KEYS = ["scorecard", "forest", "gbm"]
MATERIAL_SPREAD = 0.10     # absolute probability spread that counts as a material disagreement
PROBABILITY_FLOOR = 0.002  # calibrated probabilities are clipped to [floor, 1 - floor]
_STATE = {"models": {}, "registry": None, "loaded_at": None, "errors": {}}


class Driver(BaseModel):
    feature: str
    label: str
    contribution: float
    observed: float | str | None = None
    observed_text: str = ""


class ModelResult(BaseModel):
    model_key: str
    model_name: str
    model_version: str
    model_hash: str = ""
    preprocessing_version: str = ""
    probability: float | None = None
    raw_probability: float | None = None
    risk_band: str | None = None
    risk_label: str | None = None
    recommendation: str | None = None
    recommendation_label: str | None = None
    decision_threshold: float = DECISION_THRESHOLD
    band_cutoffs: dict = Field(default_factory=lambda: dict(BAND_CUTOFFS))
    drivers_positive: list[Driver] = Field(default_factory=list)
    drivers_negative: list[Driver] = Field(default_factory=list)
    explainer_method: str = ""
    explainer_note: str = ""
    scoring_ms: float = 0.0
    processing_ms: float = 0.0
    in_range: bool = True
    out_of_range: list[str] = Field(default_factory=list)
    unusual: list[str] = Field(default_factory=list)
    error: str | None = None


class ConsensusResult(BaseModel):
    models_used: list[str]
    models_failed: list[str]
    probabilities: dict[str, float]
    median_probability: float | None
    spread: float | None
    agreement: str                       # unanimous | minor | material | insufficient
    model_recommendation: str
    policy_result: str
    final_recommendation: str
    final_label: str
    routed_to_manual_review: bool
    explanation: list[str]
    bands_version: str = BANDS_VERSION


class ScoreRequest(BaseModel):
    """Typed API request: the raw application fields."""
    application: ApplicationInput


class ScoreResponse(BaseModel):
    derived: dict
    policy: dict
    models: list[ModelResult]
    consensus: ConsensusResult
    reason_codes: dict
    counterfactuals: list[dict]
    scoring_hash: str
    total_ms: float


# ── Loading ──────────────────────────────────────────────

def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_models(auto_train: bool = True) -> dict:
    """Load every registered model once. A missing registry triggers training when ``auto_train`` is set.
    A model whose file hash differs from the registry is refused (governance: no silent artifact swaps)."""
    import joblib
    if not REGISTRY_PATH.exists() and auto_train:
        log.warning("No model registry found; training the synthetic demonstration models now.")
        from train import train_all
        train_all()
    _STATE["models"], _STATE["errors"] = {}, {}
    if not REGISTRY_PATH.exists():
        _STATE["registry"] = None
        _STATE["errors"]["registry"] = "model_registry.json not found; run python train.py"
        return status()
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    _STATE["registry"] = registry
    for key, meta in registry.get("models", {}).items():
        path = ROOT / meta["file"]
        try:
            if not path.exists():
                raise FileNotFoundError(f"{meta['file']} is missing")
            digest = file_sha256(path)
            if digest != meta.get("sha256"):
                raise ValueError(f"{meta['file']} hash {digest[:12]} does not match the registry ({meta.get('sha256', '')[:12]})")
            bundle = joblib.load(path)
            _STATE["models"][key] = {"key": key, "pipeline": bundle["pipeline"], "calibrator": bundle["calibrator"], "meta": meta}
        except Exception as exc:  # one broken model must not take the others down
            _STATE["errors"][key] = str(exc)
            log.error("Model %s unavailable: %s", key, exc)
    _STATE["loaded_at"] = time.time()
    return status()


def status() -> dict:
    return {"loaded": sorted(_STATE["models"]), "errors": dict(_STATE["errors"]),
            "registry_version": (_STATE["registry"] or {}).get("registry_version"), "loaded_at": _STATE["loaded_at"]}


def registry() -> dict:
    return _STATE["registry"] or {}


def models_available() -> dict:
    return _STATE["models"]


# ── Scoring ──────────────────────────────────────────────

def _frame(feature_values: dict) -> pd.DataFrame:
    return pd.DataFrame([{name: feature_values[name] for name in MODEL_FEATURES}])


def _frames(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{name: row[name] for name in MODEL_FEATURES} for row in rows])


def _range_check(meta: dict, feature_values: dict):
    ranges = meta.get("ranges", {})
    out, unusual = [], []
    for name in NUMERIC_FEATURES:
        spec = ranges.get(name)
        value = feature_values.get(name)
        if not spec or value is None:
            continue
        if value < spec["min"] or value > spec["max"]:
            out.append(name)
        elif value < spec["p01"] or value > spec["p99"]:
            unusual.append(name)
    for name in CATEGORICAL_FEATURES:
        seen = meta.get("categories", {}).get(name)
        if seen and feature_values.get(name) not in seen:
            out.append(name)
    return not out, out, unusual


def score_model(key: str, feature_values: dict, with_explanation: bool = True) -> ModelResult:
    entry = _STATE["models"].get(key)
    meta = (registry().get("models", {}).get(key) or {})
    result = ModelResult(model_key=key, model_name=meta.get("name", key), model_version=meta.get("version", "unavailable"),
                         model_hash=meta.get("sha256", ""), preprocessing_version=registry().get("preprocessing_version", ""))
    if entry is None:
        result.error = _STATE["errors"].get(key, "model not loaded")
        return result
    started = time.perf_counter()
    try:
        frame = _frame(feature_values)
        raw = float(entry["pipeline"].predict_proba(frame)[0][1])
        # Isotonic calibration can return exactly 0 or 1; clip so no estimate is presented as certain.
        probability = float(np.clip(entry["calibrator"].predict_proba(frame)[0][1], PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR))
        result.scoring_ms = round((time.perf_counter() - started) * 1000, 2)
        band = band_for(probability)
        result.probability, result.raw_probability = round(probability, 5), round(raw, 5)
        result.risk_band, result.risk_label = band["band"], band["label"]
        result.recommendation, result.recommendation_label = band["recommendation"], RECOMMENDATION_LABELS[band["recommendation"]]
        result.in_range, result.out_of_range, result.unusual = _range_check(entry["meta"], feature_values)
        if with_explanation:
            explanation = explain(entry, frame, feature_values)
            result.explainer_method, result.explainer_note = explanation["method"], explanation["note"]
            result.drivers_positive = [Driver(**d) for d in explanation["drivers_positive"]]
            result.drivers_negative = [Driver(**d) for d in explanation["drivers_negative"]]
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        log.exception("Scoring failed for %s", key)
    result.processing_ms = round((time.perf_counter() - started) * 1000, 2)
    return result


def consensus(results: list[ModelResult], policy_result: str) -> ConsensusResult:
    ok = [r for r in results if r.error is None and r.probability is not None]
    failed = [r.model_key for r in results if r not in ok]
    probabilities = {r.model_key: r.probability for r in ok}
    explanation = []
    if len(ok) < 2:
        model_rec, agreement = "manual_review", "insufficient"
        explanation.append(f"Only {len(ok)} model result(s) available; at least two are required for an automated recommendation.")
        median = ok[0].probability if ok else None
        spread = 0.0 if ok else None
    else:
        values = [r.probability for r in ok]
        median, spread = statistics.median(values), max(values) - min(values)
        recs = {r.recommendation for r in ok}
        sev = [severity(r.recommendation) for r in ok]
        detail = "; ".join(f"{r.model_name}: {r.probability:.1%} → {r.recommendation_label}" for r in ok)
        if len(recs) == 1:
            model_rec, agreement = ok[0].recommendation, "unanimous"
            explanation.append(f"All {len(ok)} models agree on {RECOMMENDATION_LABELS[model_rec].lower()} ({detail}).")
        elif max(sev) - min(sev) <= 1 and spread <= MATERIAL_SPREAD:
            model_rec, agreement = most_conservative(*recs), "minor"
            explanation.append(f"Models differ by one step with a {spread:.1%} probability spread; the more conservative recommendation is used ({detail}).")
        else:
            model_rec, agreement = "manual_review", "material"
            explanation.append(f"Material disagreement: probability spread {spread:.1%} and recommendations {', '.join(sorted(recs))} ({detail}). Routed to manual review.")
        if spread > MATERIAL_SPREAD and agreement != "material":
            explanation.append(f"Probability spread {spread:.1%} exceeds {MATERIAL_SPREAD:.0%}; treat the estimate with caution.")
    final = most_conservative(model_rec, policy_result)
    if severity(policy_result) > severity(model_rec):
        explanation.append(f"Underwriting policy result ({RECOMMENDATION_LABELS[policy_result].lower()}) is more conservative than the models and takes precedence.")
    elif policy_result != "approve":
        explanation.append(f"Underwriting policy result: {RECOMMENDATION_LABELS[policy_result].lower()}.")
    return ConsensusResult(models_used=[r.model_key for r in ok], models_failed=failed, probabilities=probabilities,
                           median_probability=round(median, 5) if median is not None else None,
                           spread=round(spread, 5) if spread is not None else None, agreement=agreement,
                           model_recommendation=model_rec, policy_result=policy_result, final_recommendation=final,
                           final_label=RECOMMENDATION_LABELS[final], routed_to_manual_review=final == "manual_review", explanation=explanation)


def consensus_drivers(results: list[ModelResult]) -> list[dict]:
    """Average of normalized contributions across models (positive = raises risk)."""
    totals, counts = {}, 0
    for r in results:
        rows = list(r.drivers_positive) + list(r.drivers_negative)
        scale = sum(abs(d.contribution) for d in rows) or 1.0
        if not rows:
            continue
        counts += 1
        for d in rows:
            totals.setdefault(d.feature, {"feature": d.feature, "label": d.label, "contribution": 0.0, "observed": d.observed, "observed_text": d.observed_text})
            totals[d.feature]["contribution"] += d.contribution / scale
    out = [dict(v, contribution=round(v["contribution"] / max(counts, 1), 5)) for v in totals.values()]
    return sorted(out, key=lambda d: d["contribution"], reverse=True)


def consensus_probabilities(inputs: list[ApplicationInput]) -> list[float]:
    """Median calibrated probability across available models for a batch (used by counterfactuals)."""
    rows = [feature_row(app) for app in inputs]
    frame = _frames(rows)
    columns = []
    for entry in _STATE["models"].values():
        try:
            columns.append(np.clip(entry["calibrator"].predict_proba(frame)[:, 1], PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR))
        except Exception:
            continue
    if not columns:
        return [1.0] * len(inputs)
    return np.median(np.vstack(columns), axis=0).tolist()


def score_application(app: ApplicationInput, with_counterfactuals: bool = True) -> ScoreResponse:
    started = time.perf_counter()
    derived = derive_metrics(app)
    feature_values = feature_row(app, derived)
    policy = evaluate_policy(app, derived)
    results = [score_model(key, feature_values) for key in MODEL_KEYS]
    agreed = consensus(results, policy["result"])
    drivers = [d for d in consensus_drivers([r for r in results if r.error is None]) if d["contribution"] > 0]
    versions = ", ".join(f"{r.model_key}@{r.model_version}" for r in results if r.error is None)
    codes = assign_reason_codes(policy, drivers, agreed.final_recommendation, feature_values, versions)
    cf = []
    if with_counterfactuals and agreed.median_probability is not None and agreed.final_recommendation != "approve":
        try:
            cf = counterfactuals(app, agreed.median_probability, consensus_probabilities, band_for)
        except Exception as exc:  # guidance is optional
            log.warning("Counterfactual estimation failed: %s", exc)
    payload = {"inputs": app.model_dump(), "derived": derived.model_dump(), "policy": policy,
               "models": [r.model_dump(exclude={"scoring_ms", "processing_ms"}) for r in results], "consensus": agreed.model_dump()}
    # the hash covers inputs, derived figures, policy, model outputs and consensus - not timings - so a re-score of
    # identical inputs with the same artifacts reproduces it
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return ScoreResponse(derived=derived.model_dump(), policy=policy, models=results, consensus=agreed, reason_codes=codes,
                         counterfactuals=cf, scoring_hash=digest, total_ms=round((time.perf_counter() - started) * 1000, 2))
