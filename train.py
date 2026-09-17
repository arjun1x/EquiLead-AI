"""Train, calibrate, evaluate and register the three EquiLead models.

    python train.py [--rows 12000] [--seed 20260916] [--quick]

Pipeline: synthetic HELOC data -> stratified 60/20/20 split (train / calibration+validation / final test)
-> 5-fold CV on the training split -> fit -> isotonic calibration on the validation split -> evaluation on the
untouched test split (discrimination, calibration, threshold analysis, risk-band validation, fairness by
protected attribute) -> artifacts with SHA-256 hashes, model_registry.json, training_report.json and MODEL_CARDS.md.

Everything is deterministic for a given seed. The data are synthetic, so every metric describes the
generator's simulated relationship and nothing about real borrowers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from calculations import CATEGORICAL_FEATURES, FEATURE_LABELS, MODEL_FEATURES, NUMERIC_FEATURES
from fairness import evaluate as fairness_evaluate
from policy import BANDS, BANDS_VERSION, BAND_CUTOFFS, DECISION_THRESHOLD, POLICY_VERSION, band_for
from synth_data import DEFAULT_ROWS, DEFAULT_SEED, GENERATOR_VERSION, PROTECTED_COLUMNS, TARGET, generate

ROOT = Path(__file__).resolve().parent
REGISTRY_VERSION = "registry-v1"
PREPROCESSING_VERSION = "prep-v1"
MODEL_FILES = {"scorecard": "model_scorecard.pkl", "forest": "model_forest.pkl", "gbm": "model_gbm.pkl"}
MODEL_DOCS = {
    "scorecard": {"description": "Regularised logistic regression on standardised numeric features and one-hot categories; the most transparent of the three and the reference for the scorecard points table.",
                  "explainer": "Coefficient contributions (log-odds)", "intended_use": "Decision support and reason-code grounding for home-equity applications in this workspace",
                  "not_for": "Automated decisions, pricing, or any use with real applicants without validation on real outcomes",
                  "limitations": "Linear in the (standardised) inputs; trained on synthetic data whose default mechanism is documented in synth_data.py."},
    "forest": {"description": "Random forest with class weighting; captures interactions the scorecard misses and provides a second, uncorrelated opinion.",
               "explainer": "SHAP TreeExplainer (probability)", "intended_use": "Cross-check of the scorecard within the consensus rule",
               "not_for": "Sole basis for a recommendation or customer explanations", "limitations": "Probability-space SHAP values are approximate for ensembles; slower than the other two."},
    "gbm": {"description": "Gradient-boosted trees (XGBoost when installed, otherwise scikit-learn HistGradientBoosting) with early depth limits.",
            "explainer": "SHAP TreeExplainer (log-odds)", "intended_use": "Third independent opinion; sensitive to non-linear thresholds such as CLTV caps",
            "not_for": "Direct customer-facing explanations", "limitations": "Can extrapolate poorly outside the training ranges; out-of-range inputs are flagged at scoring time."},
}
MODEL_NAMES = {"scorecard": "Policy scorecard (logistic regression)", "forest": "Random forest", "gbm": "Gradient boosting"}
FALSE_NEGATIVE_COST, FALSE_POSITIVE_COST = 5.0, 1.0   # documented assumption for the threshold analysis only
SELECTION_CUTOFF = BAND_CUTOFFS["conditional_approval"]   # favourable recommendation = probability below 0.20


def _preprocessor(scale: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)], sparse_threshold=0)


def build_pipeline(key: str, seed: int, quick: bool = False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    if key == "scorecard":
        clf, family, algorithm = LogisticRegression(C=0.5, max_iter=3000), "linear", "sklearn LogisticRegression (L2, C=0.5)"
    elif key == "forest":
        clf = RandomForestClassifier(n_estimators=80 if quick else 300, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=seed)
        family, algorithm = "tree", f"sklearn RandomForestClassifier ({clf.n_estimators} trees, depth 10, min leaf 20)"
    else:
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(n_estimators=150 if quick else 400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                                random_state=seed, eval_metric="logloss", n_jobs=4, verbosity=0)
            family, algorithm = "tree", f"XGBoost XGBClassifier ({clf.n_estimators} rounds, depth 4, eta 0.05)"
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(max_iter=150 if quick else 400, max_depth=4, learning_rate=0.05, random_state=seed)
            family, algorithm = "tree", "sklearn HistGradientBoostingClassifier (400 iterations, depth 4)"
    return Pipeline([("prep", _preprocessor(scale=key == "scorecard")), ("clf", clf)]), family, algorithm


def _metrics(y, p, threshold):
    from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    y_pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred, labels=[0, 1]).tolist()
    return {"threshold": threshold, "roc_auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4),
            "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4), "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4), "brier": round(float(brier_score_loss(y, p)), 5),
            "confusion_matrix": {"tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1]}, "n": int(len(y)), "positives": int(y.sum())}


def _threshold_table(y, p):
    rows = []
    for threshold in np.round(np.arange(0.05, 0.61, 0.05), 2):
        y_pred = (p >= threshold).astype(int)
        tp = int(((y == 1) & (y_pred == 1)).sum()); fp = int(((y == 0) & (y_pred == 1)).sum())
        fn = int(((y == 1) & (y_pred == 0)).sum()); tn = int(((y == 0) & (y_pred == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"threshold": float(threshold), "precision": round(precision, 4), "recall": round(recall, 4),
                     "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
                     "flag_rate": round((tp + fp) / len(y), 4), "cost": FALSE_NEGATIVE_COST * fn + FALSE_POSITIVE_COST * fp})
    return rows


def _calibration_curve(y, p, bins=10):
    from sklearn.calibration import calibration_curve
    frac, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return [{"mean_predicted": round(float(m), 4), "observed_rate": round(float(f), 4)} for m, f in zip(mean_pred, frac)]


def _band_validation(y, p):
    out = []
    bands = np.array([band_for(v)["band"] for v in p])
    for band in BANDS:
        mask = bands == band["band"]
        n = int(mask.sum())
        out.append({"band": band["band"], "label": band["label"], "recommendation": band["recommendation"], "count": n,
                    "share": round(n / len(y), 4), "observed_default_rate": round(float(y[mask].mean()), 4) if n else None})
    return out


def _ranges(X):
    ranges = {name: {"min": float(X[name].min()), "max": float(X[name].max()), "p01": float(X[name].quantile(0.01)),
                     "p99": float(X[name].quantile(0.99)), "median": float(X[name].median())} for name in NUMERIC_FEATURES}
    categories = {name: sorted(X[name].astype(str).unique().tolist()) for name in CATEGORICAL_FEATURES}
    reference = {name: ranges[name]["median"] for name in NUMERIC_FEATURES}
    reference.update({name: X[name].mode().iloc[0] for name in CATEGORICAL_FEATURES})
    return ranges, categories, reference


def _reference_stats(X, scores):
    numeric = {}
    for name in NUMERIC_FEATURES:
        edges = np.unique(np.quantile(X[name], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]))
        if len(edges) < 3:
            continue
        counts, _ = np.histogram(X[name], bins=edges)
        numeric[name] = {"edges": [float(e) for e in edges], "proportions": [round(float(c / len(X)), 5) for c in counts]}
    categorical = {name: {str(k): round(float(v), 5) for k, v in X[name].astype(str).value_counts(normalize=True).items()} for name in CATEGORICAL_FEATURES}
    edges = [0, 0.05, 0.10, 0.20, 0.35, 0.5, 1.0]
    counts, _ = np.histogram(np.clip(scores, 0, 0.999999), bins=edges)
    return {"numeric": numeric, "categorical": categorical, "score": {"edges": edges, "proportions": [round(float(c / len(scores)), 5) for c in counts]}}


def _global_importance(key, pipeline, X_sample):
    """Mean |SHAP| for tree models, |coefficient| for the scorecard; permutation importance as fallback."""
    from explain import _fold, SHAP_AVAILABLE, _tree_shap
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    try:
        if key == "scorecard":
            folded = _fold(names, np.abs(clf.coef_[0]))
            method = "absolute standardized coefficient"
        elif SHAP_AVAILABLE:
            import shap
            transformed = np.asarray(prep.transform(X_sample))
            values = shap.TreeExplainer(clf).shap_values(transformed)
            if isinstance(values, list):
                values = values[-1]
            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:, :, -1]
            folded = _fold(names, np.abs(values).mean(axis=0))
            method = "mean |SHAP| on a test sample"
        else:
            raise RuntimeError("shap unavailable")
    except Exception:
        from sklearn.inspection import permutation_importance
        y_dummy = pipeline.predict(X_sample)
        result = permutation_importance(pipeline, X_sample, y_dummy, n_repeats=3, random_state=0, scoring="roc_auc")
        folded = dict(zip(MODEL_FEATURES, result.importances_mean))
        method = "permutation importance (fallback)"
    total = sum(abs(v) for v in folded.values()) or 1.0
    rows = sorted(({"feature": f, "label": FEATURE_LABELS.get(f, f), "importance": round(abs(v) / total, 4)} for f, v in folded.items()),
                  key=lambda r: r["importance"], reverse=True)
    return {"method": method, "features": rows}


def _scorecard_points(pipeline):
    """Points-style view of the logistic scorecard: 20 points per doubling of odds, base 600 at the intercept."""
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    factor = 20 / np.log(2)
    rows = [{"feature": n.split("__", 1)[-1], "coefficient": round(float(c), 4), "points_per_unit": round(float(-c * factor), 2)}
            for n, c in zip(names, clf.coef_[0])]
    return {"base_points": round(600 - float(clf.intercept_[0]) * factor, 1), "points_per_doubling_of_odds": 20, "rows": rows,
            "note": "Points are on standardized inputs; negative points raise risk. Illustrative scorecard view of the logistic model."}


def train_all(rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED, quick: bool = False, out_dir: Path = ROOT) -> dict:
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

    started = time.perf_counter()
    data = generate(rows, seed)
    X, y = data[MODEL_FEATURES].copy(), data[TARGET].astype(int)
    protected = data[PROTECTED_COLUMNS]
    X_train, X_rest, y_train, y_rest, p_train, p_rest = train_test_split(X, y, protected, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test, p_val, p_test = train_test_split(X_rest, y_rest, p_rest, test_size=0.5, stratify=y_rest, random_state=seed)
    ranges, categories, reference = _ranges(X_train)
    folds = StratifiedKFold(n_splits=3 if quick else 5, shuffle=True, random_state=seed)
    trained_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    registry = {"registry_version": REGISTRY_VERSION, "preprocessing_version": PREPROCESSING_VERSION, "policy_version": POLICY_VERSION,
                "bands_version": BANDS_VERSION, "trained_at": trained_at, "seed": seed, "quick": quick,
                "dataset": {"generator_version": GENERATOR_VERSION, "rows": int(len(data)), "target": TARGET,
                            "target_rate": round(float(y.mean()), 4), "splits": {"train": int(len(X_train)), "validation": int(len(X_val)), "test": int(len(X_test))},
                            "protected_attributes": {col: {str(k): int(v) for k, v in data[col].value_counts().items()} for col in PROTECTED_COLUMNS},
                            "license": "CC0 1.0 (synthetic, generated by synth_data.py)"},
                "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
                "bands": BANDS, "decision_threshold": DECISION_THRESHOLD, "selection_cutoff": SELECTION_CUTOFF,
                "cost_assumption": {"false_negative": FALSE_NEGATIVE_COST, "false_positive": FALSE_POSITIVE_COST},
                "reference_stats": None, "models": {}}
    report = {"trained_at": trained_at, "seed": seed, "dataset": registry["dataset"], "models": {}, "fairness": {}, "comparison": []}
    scores_for_reference = None

    for key in ("scorecard", "forest", "gbm"):
        t0 = time.perf_counter()
        pipeline, family, algorithm = build_pipeline(key, seed, quick)
        cv = cross_validate(pipeline, X_train, y_train, cv=folds, scoring=["roc_auc", "average_precision"], n_jobs=1)
        pipeline.fit(X_train, y_train)
        calibrator = CalibratedClassifierCV(FrozenEstimator(pipeline), method="isotonic")
        calibrator.fit(X_val, y_val)
        p_raw = pipeline.predict_proba(X_test)[:, 1]
        p_cal = calibrator.predict_proba(X_test)[:, 1]
        p_val_cal = calibrator.predict_proba(X_val)[:, 1]
        metrics = {"cross_validation": {"folds": folds.get_n_splits(), "roc_auc_mean": round(float(cv["test_roc_auc"].mean()), 4),
                                        "roc_auc_std": round(float(cv["test_roc_auc"].std()), 4),
                                        "pr_auc_mean": round(float(cv["test_average_precision"].mean()), 4),
                                        "pr_auc_std": round(float(cv["test_average_precision"].std()), 4)},
                   "validation": _metrics(y_val.to_numpy(), p_val_cal, DECISION_THRESHOLD),
                   "test": _metrics(y_test.to_numpy(), p_cal, DECISION_THRESHOLD),
                   "test_uncalibrated_brier": round(float(np.mean((p_raw - y_test.to_numpy()) ** 2)), 5),
                   "calibration_curve": _calibration_curve(y_test.to_numpy(), p_cal),
                   "threshold_table": _threshold_table(y_test.to_numpy(), p_cal),
                   "band_validation": _band_validation(y_test.to_numpy(), p_cal)}
        best = min(metrics["threshold_table"], key=lambda r: (r["cost"], -r["f1"]))
        metrics["cost_optimal_threshold"] = best["threshold"]
        fairness = fairness_evaluate(y_test.to_numpy(), p_cal, {col: p_test[col].to_numpy() for col in PROTECTED_COLUMNS}, SELECTION_CUTOFF)
        importance = _global_importance(key, pipeline, X_test.sample(min(300, len(X_test)), random_state=seed))
        version = f"{key}-1.0.0-{seed}"
        file_name = MODEL_FILES[key]
        joblib.dump({"pipeline": pipeline, "calibrator": calibrator, "key": key, "version": version}, out_dir / file_name)
        digest = hashlib.sha256((out_dir / file_name).read_bytes()).hexdigest()
        meta = {"key": key, "name": MODEL_NAMES[key], "version": version, "file": file_name, "sha256": digest, "algorithm": algorithm,
                "algorithm_family": family, "calibration": "isotonic on the validation split", "trained_at": trained_at,
                "training_seconds": round(time.perf_counter() - t0, 1), "ranges": ranges, "categories": categories,
                "reference_values": reference, "metrics": metrics, "importance": importance}
        if key == "scorecard":
            meta["scorecard_points"] = _scorecard_points(pipeline)
        registry["models"][key] = meta
        report["models"][key] = {"name": MODEL_NAMES[key], "version": version, "algorithm": algorithm, "metrics": metrics, "importance": importance}
        report["fairness"][key] = fairness
        report["comparison"].append({"model": key, "name": MODEL_NAMES[key], **{k: metrics["test"][k] for k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "brier")},
                                     "cv_roc_auc": metrics["cross_validation"]["roc_auc_mean"], "cost_optimal_threshold": best["threshold"]})
        if key == "gbm":
            scores_for_reference = calibrator.predict_proba(X_train)[:, 1]
        print(f"{MODEL_NAMES[key]:<44} CV AUC {metrics['cross_validation']['roc_auc_mean']:.3f}  test AUC {metrics['test']['roc_auc']:.3f}  "
              f"PR-AUC {metrics['test']['pr_auc']:.3f}  Brier {metrics['test']['brier']:.4f}  ({meta['training_seconds']}s)")

    registry["reference_stats"] = _reference_stats(X_train, scores_for_reference)
    registry["training_seconds"] = round(time.perf_counter() - started, 1)
    (out_dir / "model_registry.json").write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "MODEL_CARDS.md").write_text(model_cards_markdown(registry, report), encoding="utf-8")
    print(f"Registry, report and MODEL_CARDS.md written to {out_dir} in {registry['training_seconds']}s.")
    return registry


def model_cards_markdown(registry: dict, report: dict) -> str:
    ds = registry["dataset"]
    lines = ["# EquiLead model cards", "",
             f"Generated {registry['trained_at']} from `train.py` (seed {registry['seed']}). All three models are trained on the same synthetic",
             f"dataset (`{ds['generator_version']}`, {ds['rows']} rows, simulated 24-month default rate {ds['target_rate']:.1%}, CC0). Metrics describe the",
             "generator's simulated relationship only. None of these models is validated for real U.S. home-equity underwriting.", "",
             "| Model | Version | Test ROC-AUC | Test PR-AUC | Brier | CV ROC-AUC | Cost-optimal threshold (info) |", "|---|---|---|---|---|---|---|"]
    for row in report["comparison"]:
        lines.append(f"| {row['name']} | {registry['models'][row['model']]['version']} | {row['roc_auc']} | {row['pr_auc']} | {row['brier']} | {row['cv_roc_auc']} | {row['cost_optimal_threshold']} |")
    for key, meta in registry["models"].items():
        m, t = meta["metrics"], meta["metrics"]["test"]
        lines += ["", f"## {meta['name']}", "",
                  f"* **Version:** `{meta['version']}` · file `{meta['file']}` · SHA-256 `{meta['sha256'][:16]}…` · trained {meta['trained_at']}",
                  f"* **Algorithm:** {meta['algorithm']} · preprocessing `{registry['preprocessing_version']}` · {meta['calibration']}",
                  "* **Intended use:** decision support inside the EquiLead demonstration; one of three models feeding a transparent consensus that a loan officer must review.",
                  "* **Prohibited use:** autonomous credit decisions, real applicants, generating regulatory adverse-action notices, any use without independent validation on approved data.",
                  f"* **Training data:** synthetic HELOC generator `{ds['generator_version']}`, {ds['splits']['train']} training / {ds['splits']['validation']} calibration / {ds['splits']['test']} test rows.",
                  f"* **Features:** {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical application fields (see calculations.MODEL_FEATURES). No protected attributes.",
                  f"* **Target:** `{ds['target']}` (simulated serious delinquency within 24 months).",
                  f"* **Metrics (test, threshold {t['threshold']}):** ROC-AUC {t['roc_auc']}, PR-AUC {t['pr_auc']}, precision {t['precision']}, recall {t['recall']}, F1 {t['f1']}, Brier {t['brier']} (uncalibrated {m['test_uncalibrated_brier']}); CV ROC-AUC {m['cross_validation']['roc_auc_mean']} ± {m['cross_validation']['roc_auc_std']}.",
                  f"* **Thresholds:** risk bands `{registry['bands_version']}` (approve < {BAND_CUTOFFS['approve']:.0%}, conditional < {BAND_CUTOFFS['conditional_approval']:.0%}, manual review < {BAND_CUTOFFS['manual_review']:.0%}, decline otherwise). Cost-optimal threshold {m['cost_optimal_threshold']} is informational only.",
                  "* **Band validation (test):** " + "; ".join(f"{b['band']} {b['observed_default_rate'] if b['observed_default_rate'] is not None else '—'} (n={b['count']})" for b in m["band_validation"]),
                  "* **Known limitations:** synthetic labels; single-family U.S.-style assumptions baked into the generator; no macro-economic or property-market features; isotonic calibration fitted on 2,400 rows; explanations describe the uncalibrated base model.",
                  "* **Fairness considerations:** evaluated on synthetic `sex` and `age_band` columns that are never model inputs; see training_report.json and the Fairness page. Signals require human and legal review and are not a fairness certification."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description="Train the EquiLead demonstration models")
    parser.add_argument("--rows", type=int, default=int(os.getenv("TRAIN_ROWS", DEFAULT_ROWS)))
    parser.add_argument("--seed", type=int, default=int(os.getenv("TRAIN_SEED", DEFAULT_SEED)))
    parser.add_argument("--quick", action="store_true", help="fewer trees and folds (for tests)")
    args = parser.parse_args()
    train_all(args.rows, args.seed, args.quick)
