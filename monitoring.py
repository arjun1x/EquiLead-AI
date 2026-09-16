"""Model drift and subgroup performance monitoring.

Drift: population stability index (PSI) of each input and of the model score against the reference
distributions stored in the model card at training time. Subgroup performance: approval rates and, where a
reviewer has recorded the observed outcome, error rates per group (see fairness.py).

Command line:
  python monitoring.py [--days 90] [--json monitoring_report.json]
"""
import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

from fairness import group_rates, four_fifths, error_rate_gaps, DEFAULT_GROUPS, MIN_GROUP_N

ROOT = Path(__file__).resolve().parent
FEATURE_ATTR = {"LOAN": "loan_amount", "MORTDUE": "mortgage_due", "VALUE": "property_value", "REASON": "reason",
                "JOB": "job", "YOJ": "yoj", "DEROG": "derog", "DELINQ": "delinq", "CLAGE": "clage",
                "NINQ": "ninq", "CLNO": "clno", "DEBTINC": "debtinc"}
PSI_STABLE, PSI_MODERATE = 0.10, 0.25
MIN_ROWS_FOR_DRIFT = 20


def row_from_record(record):
    row = {attr: getattr(record, attr) for attr in FEATURE_ATTR.values()}
    row.update(id=record.id, decision=record.decision, score=record.default_probability,
               outcome=record.outcome, is_demo=bool(record.is_demo), created_at=record.created_at,
               human_decision=record.human_decision, model_version=record.model_version)
    return row


def psi(expected, actual, eps=1e-4):
    """Population stability index over two proportion vectors of equal length."""
    total = 0.0
    for e, a in zip(expected, actual):
        e, a = max(float(e), eps), max(float(a), eps)
        total += (a - e) * math.log(a / e)
    return total


def bin_proportions(values, edges):
    counts = [0] * (len(edges) - 1)
    kept = 0
    for value in values:
        if value is None:
            continue
        value = float(value)
        for i in range(len(edges) - 1):
            last = i == len(edges) - 2
            if edges[i] <= value < edges[i + 1] or (last and value >= edges[i]):
                counts[i] += 1
                kept += 1
                break
    return [c / kept for c in counts] if kept else None


def category_proportions(values, categories):
    counts = {name: 0 for name in categories}
    kept = 0
    for value in values:
        name = value if value in counts else "Unknown"
        if name not in counts:
            continue
        counts[name] += 1
        kept += 1
    return [counts[name] / kept for name in categories] if kept else None


def psi_status(value):
    if value is None:
        return "muted"
    return "stable" if value < PSI_STABLE else ("review" if value < PSI_MODERATE else "alert")


def drift_report(card, rows):
    reference = card.get("reference_stats") or {}
    items = []
    enough = len(rows) >= MIN_ROWS_FOR_DRIFT
    for feature, spec in (reference.get("numeric") or {}).items():
        values = [r.get(FEATURE_ATTR.get(feature, feature)) for r in rows]
        actual = bin_proportions(values, spec["edges"]) if enough else None
        value = psi(spec["proportions"], actual) if actual else None
        items.append({"feature": feature, "kind": "numeric", "psi": value, "status": psi_status(value)})
    for feature, spec in (reference.get("categorical") or {}).items():
        categories = list(spec.keys())
        values = [r.get(FEATURE_ATTR.get(feature, feature)) for r in rows]
        actual = category_proportions(values, categories) if enough else None
        value = psi([spec[c] for c in categories], actual) if actual else None
        items.append({"feature": feature, "kind": "categorical", "psi": value, "status": psi_status(value)})
    score_spec = reference.get("score")
    score = None
    if score_spec:
        actual = bin_proportions([r.get("score") for r in rows], score_spec["edges"]) if enough else None
        value = psi(score_spec["proportions"], actual) if actual else None
        score = {"psi": value, "status": psi_status(value)}
    worst = max((i["psi"] for i in items if i["psi"] is not None), default=None)
    return {"items": items, "score": score, "worst_psi": worst, "status": psi_status(worst),
            "enough_rows": enough, "min_rows": MIN_ROWS_FOR_DRIFT, "has_reference": bool(items or score_spec)}


def subgroup_report(rows, group_keys=DEFAULT_GROUPS, min_n=MIN_GROUP_N):
    out = {}
    for key in group_keys:
        rates = group_rates(rows, key)
        out[key] = {"rates": rates, "four_fifths": four_fifths(rates, min_n), "error_rates": error_rate_gaps(rates, min_n)}
    return out


def build_report(rows, card, days=None, group_keys=DEFAULT_GROUPS):
    versions = sorted({r.get("model_version") for r in rows if r.get("model_version")})
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "window_days": days, "n": len(rows),
        "labeled": sum(1 for r in rows if r.get("outcome") in (0, 1)),
        "synthetic": bool(rows) and all(r.get("is_demo") for r in rows),
        "model_versions": versions, "card_model_version": card.get("model_version"),
        "drift": drift_report(card, rows), "subgroups": subgroup_report(rows, group_keys),
        "min_group_n": MIN_GROUP_N,
    }


def rows_from_db(days=None):
    from sqlalchemy import select
    from models import Session, LoanDecision, init_db, utcnow
    init_db()
    with Session() as db:
        query = select(LoanDecision)
        if days:
            query = query.where(LoanDecision.created_at >= utcnow().replace(tzinfo=None) - dt.timedelta(days=days))
        return [row_from_record(r) for r in db.scalars(query.order_by(LoanDecision.id))]


def _main(argv):
    parser = argparse.ArgumentParser(description="Drift and subgroup monitoring report")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    from model_card import load_model_card
    card = load_model_card()
    report = build_report(rows_from_db(args.days), card, args.days)
    if report["synthetic"]:
        print("NOTE: every record is a synthetic demo fixture; drift figures are illustrative only.\n")
    print(f"records={report['n']} labeled={report['labeled']} window={args.days}d model_versions={report['model_versions']}")
    print(f"drift: {report['drift']['status'].upper()} (worst PSI {report['drift']['worst_psi']})")
    for item in report["drift"]["items"]:
        value = "n/a" if item["psi"] is None else f"{item['psi']:.3f}"
        print(f"  {item['feature']:>8} {value:>6} {item['status']}")
    for key, block in report["subgroups"].items():
        ff = block["four_fifths"]
        print(f"subgroup {key}: four-fifths {ff['status']}" + (f" min ratio {ff['min_ratio']:.2f}" if ff["min_ratio"] else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    _main(sys.argv[1:])
