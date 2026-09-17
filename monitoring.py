"""Input and score drift monitoring against the training reference (population stability index).

Command line:  python monitoring.py [--days 90] [--json monitoring_report.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import numpy as np
import json
import math
import sys
from pathlib import Path

from calculations import CATEGORICAL_FEATURES, FEATURE_LABELS, NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parent
PSI_STABLE, PSI_MODERATE = 0.10, 0.25
MIN_ROWS_FOR_DRIFT = 20


def row_from_application(app) -> dict:
    """Feature values, consensus probability and recommendation for one stored application."""
    inputs, derived = app.inputs or {}, app.derived or {}
    row = {name: inputs.get(name) for name in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    for name in ("monthly_income", "dti_after", "cltv_after", "ltv_current", "reserves_months", "doc_completeness"):
        row[name] = derived.get(name)
    consensus = app.consensus or {}
    row.update(id=app.id, score=consensus.get("median_probability"), recommendation=consensus.get("final_recommendation"),
               agreement=consensus.get("agreement"), probabilities=consensus.get("probabilities") or {},
               state=app.state, product_type=inputs.get("product_type"), occupancy_type=inputs.get("occupancy_type"),
               human_decision=app.human_decision, override=bool(app.override), created_at=app.created_at)
    return row


def psi(expected, actual, eps=1e-4):
    total = 0.0
    for e, a in zip(expected, actual):
        e, a = max(float(e), eps), max(float(a), eps)
        total += (a - e) * math.log(a / e)
    return total


def bin_proportions(values, edges):
    counts, kept = [0] * (len(edges) - 1), 0
    for value in values:
        if value is None:
            continue
        value = float(value)
        for i in range(len(edges) - 1):
            last = i == len(edges) - 2
            if edges[i] <= value < edges[i + 1] or (last and value >= edges[i]):
                counts[i] += 1; kept += 1
                break
    return [c / kept for c in counts] if kept else None


def category_proportions(values, categories):
    counts, kept = {name: 0 for name in categories}, 0
    for value in values:
        if value in counts:
            counts[value] += 1; kept += 1
    return [counts[name] / kept for name in categories] if kept else None


def psi_status(value):
    if value is None:
        return "muted"
    return "stable" if value < PSI_STABLE else ("review" if value < PSI_MODERATE else "alert")


def drift_report(reference: dict, rows: list[dict]) -> dict:
    reference = reference or {}
    enough = len(rows) >= MIN_ROWS_FOR_DRIFT
    items = []
    for feature, spec in (reference.get("numeric") or {}).items():
        actual = bin_proportions([r.get(feature) for r in rows], spec["edges"]) if enough else None
        value = psi(spec["proportions"], actual) if actual else None
        items.append({"feature": feature, "label": FEATURE_LABELS.get(feature, feature), "kind": "numeric", "psi": value, "status": psi_status(value)})
    for feature, spec in (reference.get("categorical") or {}).items():
        categories = list(spec.keys())
        actual = category_proportions([r.get(feature) for r in rows], categories) if enough else None
        value = psi([spec[c] for c in categories], actual) if actual else None
        items.append({"feature": feature, "label": FEATURE_LABELS.get(feature, feature), "kind": "categorical", "psi": value, "status": psi_status(value)})
    score = None
    if reference.get("score"):
        actual = bin_proportions([r.get("score") for r in rows if r.get("score") is not None], reference["score"]["edges"]) if enough else None
        value = psi(reference["score"]["proportions"], actual) if actual else None
        score = {"psi": value, "status": psi_status(value)}
    worst = max((i["psi"] for i in items if i["psi"] is not None), default=None)
    return {"items": items, "score": score, "worst_psi": worst, "status": psi_status(worst), "enough_rows": enough,
            "min_rows": MIN_ROWS_FOR_DRIFT, "has_reference": bool(items or score)}


def recommendation_mix(rows: list[dict], key: str) -> dict:
    out = {}
    for row in rows:
        if not row.get("recommendation"):
            continue
        group = out.setdefault(str(row.get(key) or "unknown"), {"n": 0, "favourable": 0})
        group["n"] += 1
        group["favourable"] += row["recommendation"] in ("approve", "conditional_approval")
    for group in out.values():
        group["favourable_rate"] = group["favourable"] / group["n"] if group["n"] else None
    return out


def share_of(rows: list[dict], key: str) -> dict:
    """Share of each value of `key` among rows that have one (e.g. recommendation mix)."""
    counts = {}
    for row in rows:
        if row.get(key):
            counts[row[key]] = counts.get(row[key], 0) + 1
    total = sum(counts.values())
    return {k: v / total for k, v in sorted(counts.items())} if total else {}


def median_by_model(rows: list[dict]) -> dict:
    values = {}
    for row in rows:
        for key, p in (row.get("probabilities") or {}).items():
            if p is not None:
                values.setdefault(key, []).append(p)
    return {k: float(np.median(v)) for k, v in values.items()}


def build_report(rows: list[dict], registry: dict, days=None) -> dict:
    scored = [r for r in rows if r.get("score") is not None]
    decided = [r for r in rows if r.get("human_decision")]
    return {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "window_days": days,
            "n": len(rows), "scored": len(scored), "decided": len(decided), "registry_version": registry.get("registry_version"),
            "drift": drift_report(registry.get("reference_stats"), scored),
            "mix": {"product_type": recommendation_mix(scored, "product_type"), "occupancy_type": recommendation_mix(scored, "occupancy_type")},
            "recommendation_mix": share_of(scored, "recommendation"), "decision_mix": share_of(decided, "human_decision"),
            "agreement_mix": share_of(scored, "agreement"), "median_probability": median_by_model(scored),
            "override_rate": (sum(1 for r in decided if r.get("override")) / len(decided)) if decided else None,
            "note": "Drift compares recent applications with the training reference. Outcome-based performance monitoring "
                    "needs repayment data, which this workspace does not have."}


def rows_from_db(days=None):
    from sqlalchemy import select
    from models import Application, Session, init_db, utcnow
    init_db()
    with Session() as db:
        query = select(Application).where(Application.state != "draft")
        if days:
            query = query.where(Application.created_at >= utcnow().replace(tzinfo=None) - dt.timedelta(days=days))
        return [row_from_application(a) for a in db.scalars(query.order_by(Application.id))]


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description="Drift monitoring report")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    registry_path = ROOT / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    report = build_report(rows_from_db(args.days), registry, args.days)
    print(f"applications={report['n']} scored={report['scored']} drift={report['drift']['status'].upper()} worst PSI={report['drift']['worst_psi']}")
    for item in report["drift"]["items"]:
        print(f"  {item['label']:<32} {'n/a' if item['psi'] is None else f'{item['psi']:.3f}':>6} {item['status']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
