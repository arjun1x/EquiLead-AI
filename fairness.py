"""Fair-lending validation helpers.

The model never sees protected attributes (model_card.py refuses a card that lists any). Testing for
disparate impact still needs group membership, which must come from a governed source, never from the
application form. This module computes the standard subgroup tests on whatever grouping the governance
process supplies. Nothing here replaces legal and compliance review; it produces the evidence for it.

Command line:
  python fairness.py [--days N] [--group job reason] [--json report.json]
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PROTECTED_CLASSES = ["race", "color", "religion", "national origin", "sex", "marital status", "age",
                     "receipt of public assistance", "good-faith exercise of consumer credit rights"]

GOVERNANCE = {
    "model_inputs": "Protected attributes and obvious proxies are excluded from model inputs. The model card must list "
                    "protected_attributes_in_model as empty or the app refuses to load it.",
    "test_data": "Group membership for testing comes from a governed monitoring dataset (self-reported under HMDA "
                 "rules where applicable, or a documented proxy method such as BISG approved by compliance). "
                 "It is never captured on the application form or stored beside the decision.",
    "tests": "Approval-rate disparate impact (four-fifths rule), false-positive and false-negative rate gaps "
             "(equal opportunity), and score-distribution comparison per group, at each model release and "
             "at least quarterly.",
    "review": "Results, sample sizes, and any remediation go to the model risk and fair-lending reviewers named "
              "in the model card before a release is approved.",
    "reason_codes": "Adverse-action statements come only from the compliance-approved code table (reason_codes.py).",
}

DEFAULT_GROUPS = ("job", "reason")
MIN_GROUP_N = 30


def group_rates(rows, group_key, decision_key="decision", label_key="outcome"):
    """Per-group approval/denial rates plus error rates where an observed outcome exists.
    Positive class for the error rates is a default (label 1); a 'denied' recommendation is the positive prediction."""
    groups = {}
    for row in rows:
        group = str(row.get(group_key) if row.get(group_key) not in (None, "") else "Unknown")
        g = groups.setdefault(group, {"n": 0, "approved": 0, "denied": 0, "labeled": 0,
                                      "tp": 0, "fp": 0, "fn": 0, "tn": 0})
        g["n"] += 1
        denied = row.get(decision_key) == "denied"
        g["denied" if denied else "approved"] += 1
        label = row.get(label_key)
        if label in (0, 1):
            g["labeled"] += 1
            if denied and label == 1: g["tp"] += 1
            elif denied and label == 0: g["fp"] += 1
            elif not denied and label == 1: g["fn"] += 1
            else: g["tn"] += 1
    for g in groups.values():
        g["approval_rate"] = g["approved"] / g["n"] if g["n"] else None
        g["denial_rate"] = g["denied"] / g["n"] if g["n"] else None
        positives, negatives = g["tp"] + g["fn"], g["fp"] + g["tn"]
        g["tpr"] = g["tp"] / positives if positives else None
        g["fnr"] = g["fn"] / positives if positives else None
        g["fpr"] = g["fp"] / negatives if negatives else None
    return groups


def four_fifths(rates, min_n=MIN_GROUP_N):
    """Approval-rate ratio of every group to the most-approved group of adequate size."""
    eligible = {name: g for name, g in rates.items() if g["n"] >= min_n and g["approval_rate"] is not None}
    if len(eligible) < 2:
        return {"status": "INSUFFICIENT", "reference": None, "ratios": {}, "min_ratio": None, "min_n": min_n}
    reference = max(eligible, key=lambda name: eligible[name]["approval_rate"])
    ref_rate = eligible[reference]["approval_rate"]
    ratios = {name: (g["approval_rate"] / ref_rate if ref_rate else None) for name, g in eligible.items()}
    min_ratio = min(r for r in ratios.values() if r is not None)
    return {"status": "PASS" if min_ratio >= 0.8 else "REVIEW", "reference": reference,
            "ratios": ratios, "min_ratio": min_ratio, "min_n": min_n}


def error_rate_gaps(rates, min_n=MIN_GROUP_N, tolerance=0.10):
    """Largest difference in false-positive (good applicant declined) and false-negative rates between groups."""
    eligible = {name: g for name, g in rates.items() if g["labeled"] >= min_n}
    out = {}
    for metric in ("fpr", "fnr"):
        values = {name: g[metric] for name, g in eligible.items() if g[metric] is not None}
        if len(values) < 2:
            out[metric] = {"status": "INSUFFICIENT", "gap": None, "values": values}
            continue
        gap = max(values.values()) - min(values.values())
        out[metric] = {"status": "PASS" if gap <= tolerance else "REVIEW", "gap": gap, "values": values}
    return out


def fair_lending_report(rows, group_keys=DEFAULT_GROUPS, min_n=MIN_GROUP_N):
    report = {"n": len(rows), "labeled": sum(1 for r in rows if r.get("outcome") in (0, 1)),
              "min_group_n": min_n, "groups": {}, "protected_classes": PROTECTED_CLASSES,
              "governance": GOVERNANCE, "synthetic": all(r.get("is_demo") for r in rows) if rows else None}
    for key in group_keys:
        rates = group_rates(rows, key)
        report["groups"][key] = {"rates": rates, "four_fifths": four_fifths(rates, min_n), "error_rates": error_rate_gaps(rates, min_n)}
    return report


def rows_from_db(days=None):
    from sqlalchemy import select
    from models import Session, LoanDecision, init_db, utcnow
    from monitoring import row_from_record
    init_db()
    with Session() as db:
        query = select(LoanDecision)
        if days:
            query = query.where(LoanDecision.created_at >= utcnow().replace(tzinfo=None) - dt.timedelta(days=days))
        return [row_from_record(r) for r in db.scalars(query.order_by(LoanDecision.id))]


def _main(argv):
    parser = argparse.ArgumentParser(description="Fair-lending subgroup report")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--group", nargs="+", default=list(DEFAULT_GROUPS))
    parser.add_argument("--json", default=None, help="write the report to this file")
    args = parser.parse_args(argv)
    report = fair_lending_report(rows_from_db(args.days), tuple(args.group))
    if report["synthetic"]:
        print("NOTE: every record is a synthetic demo fixture. These numbers say nothing about a real model.\n")
    for key, block in report["groups"].items():
        print(f"== {key} ==")
        for name, g in sorted(block["rates"].items()):
            print(f"  {name:>10}: n={g['n']:>4} approval={g['approval_rate']:.1%} labeled={g['labeled']}"
                  + (f" fpr={g['fpr']:.2f}" if g["fpr"] is not None else ""))
        ff = block["four_fifths"]
        print(f"  four-fifths: {ff['status']}" + (f" (min ratio {ff['min_ratio']:.2f} vs {ff['reference']})" if ff["min_ratio"] else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    _main(sys.argv[1:])
