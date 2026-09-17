# Project notes

## Structure rule

Every file sits directly in this folder. No `src`, `templates`, `static`, `models`, `data`, `tests` or other
subfolders, no React/Vite/Node. Jinja loads templates from the folder root; `app.py` serves `styles.css`, `app.js`,
`scene.js`, `three.module.min.js` and the two SVGs from an explicit allow-list. Tests write their databases to the
system temp directory, and `run.py` / `conftest.py` set `sys.dont_write_bytecode`. Before handing over:

```bash
python -m pytest -q -p no:cacheprovider
find . -mindepth 1 -type d -not -name .git -print     # must print nothing
```

## Where things live

* Financial formulas exist once, in `calculations.py`. The form's live figures call `POST /api/calculate`; the JS
  only formats what the server returns.
* Thresholds and bands exist once, in `policy.py`; both `train.py` (band validation) and `scoring.py` import them.
* The three models are trained by `train.py` and loaded once at startup by `scoring.load_models()`. The registry
  stores each artifact's SHA-256, training ranges (p01/p99 fitted on the training split) and reference values.
  A hash mismatch marks that model unavailable; the consensus continues with the remaining models and falls back to
  manual review when fewer than two are available.
* Reason codes are the only customer-facing wording. Model drivers (SHAP / coefficients) stay on the staff screens.
* `models.record_change` writes previous and updated values into the audit chain before mutating the row.

## Regenerating artifacts

```bash
python train.py                 # ~15 s; writes model_*.pkl, model_registry.json, training_report.json, MODEL_CARDS.md
python train.py --quick         # smaller forest/boosting for fast local iteration (metrics differ)
python synth_data.py            # writes heloc_synthetic.csv (CC0)
python fairness.py              # prints the fairness summary from training_report.json
python monitoring.py            # prints the drift report from the demo database
```

Artifacts are ignored by git and regenerated automatically at first start when `model_registry.json` is missing.

## Conventions

* Every recommendation goes through `scoring.consensus`; never read a single model's recommendation as the answer.
* Anything shown to a customer must come from `reason_codes.REASON_CODES` or `letters.TEMPLATES`.
* New state transitions go into `workflow.TRANSITIONS`; routes call `assert_transition` rather than setting `state`.
* Log messages never include application records; log ids and actions only.
* Keep synthetic-data and demonstration labelling on every screen (`base.html` top bar and footer).
