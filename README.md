# EquiLead

Decision-support workspace for home-equity loans and HELOCs, built as a **single flat folder** of Python, Jinja
templates, one stylesheet and one script. Three genuinely different models score each application, a transparent
consensus rule combines them with deterministic lending policy, controlled reason codes and letter templates keep
customer communication consistent, and a hash-chained audit timeline records every step. **Models recommend. A loan
officer decides, signs and approves the letter before anything is finalized.**

> **Demonstration software with synthetic data.** Every application, model, metric and letter in this repository is
> produced from a reproducible synthetic generator. Nothing here is a validated model for U.S. home-equity lending,
> and nothing here claims production, legal or regulatory readiness. See *Limitations* and *DATASETS.md*.

## Contents

1. [Quick start](#quick-start)
2. [What the workspace does](#what-the-workspace-does)
3. [Architecture and files](#architecture-and-files)
4. [Models](#models)
5. [Consensus and policy](#consensus-and-policy)
6. [Explainability and reason codes](#explainability-and-reason-codes)
7. [Workflow, human decision and letters](#workflow-human-decision-and-letters)
8. [Fairness dashboard](#fairness-dashboard)
9. [Audit timeline and monitoring](#audit-timeline-and-monitoring)
10. [Security](#security)
11. [API](#api)
12. [Tests](#tests)
13. [Datasets, licences and acknowledgements](#datasets-licences-and-acknowledgements)
14. [Limitations](#limitations)
15. [What real production use would require](#what-real-production-use-would-require)

## Quick start

```bash
python -m pip install -r requirements.txt   # Python 3.11+
python train.py                             # optional: ~15 s; run.py trains automatically if artifacts are missing
python run.py                               # http://127.0.0.1:8000  (API_PORT=8010 python run.py to change)
python -m pytest -q -p no:cacheprovider     # all suites; browser tests skip if Playwright/Chromium are absent
```

Open the local URL, choose a demo role on the sign-in page and click **Open demo workspace**:

| Demo role | Can do |
|---|---|
| Loan processor (`analyst`) | create, edit, score and withdraw own applications; read model, fairness and audit pages |
| Loan officer (`reviewer`) | everything above for all applications, plus decisions, overrides, letters, finalization, monitoring |

Registered accounts start as processors; `python security.py set-role user@example.com reviewer` promotes one.
Copy `.env.example` to `.env` to change ports, secrets or limits. Never commit `.env`.

## What the workspace does

| Screen | Purpose |
|---|---|
| Dashboard | queue totals, cases waiting for a decision, letters to review, recent audit events, model status |
| Applications | filterable, searchable, paginated queue; CSV export (formula-safe) |
| New application | five-step intake form; partial drafts are saved; live figures come from the server, not duplicated JS formulas |
| Application | key figures and flags, three-model comparison with drivers and latency, consensus, policy rules, reason codes, counterfactual estimates, human decision form, finalize checklist, full inputs |
| Letter | controlled template generation, original vs corrected draft, language-check findings, human approval |
| History | append-only timeline with previous/updated values, every model run, every letter version |
| Models | registry, test metrics, CV, calibration, threshold analysis, band validation, importance, scorecard points, reason-code table |
| Fairness | group metrics per model and attribute on the synthetic test split |
| Monitoring | population stability (PSI) against the training reference, recommendation/decision mix, override rate |
| Audit history | the whole chain with verification status |

## Architecture and files

Everything sits directly in this folder. There is no `src/`, `templates/`, `static/`, `models/` or `data/`, no
Node.js, no bundler. Templates are loaded by Jinja from the folder root; `styles.css`, `app.js`, `scene.js` and the
SVGs are served from an explicit allow-list (`/assets/<name>`), so model files and the database are never exposed.

| File | Role |
|---|---|
| `app.py` | FastAPI routes, sessions, CSRF, roles, error pages, typed JSON API |
| `run.py` | starts uvicorn without writing bytecode |
| `calculations.py` | typed `ApplicationInput`, validation, every financial formula (LTV, CLTV, DTI, payment, equity, reserves, documentation) |
| `policy.py` | risk bands, decision threshold, deterministic underwriting rules with versions |
| `train.py` | synthetic data → three models, calibration, metrics, fairness report, registry, `MODEL_CARDS.md` |
| `scoring.py` | loads artifacts once (SHA-256 verified), scores, consensus, counterfactuals, `ScoreRequest`/`ScoreResponse` |
| `explain.py` | SHAP for trees, coefficients for the scorecard, perturbation fallback, counterfactual search |
| `reason_codes.py` | controlled reason-code table and assignment rules |
| `letters.py` | letter templates, language checker, draft correction |
| `workflow.py` | states, transitions, finalize guard |
| `fairness.py` | group metrics, thresholds, comparison |
| `monitoring.py` | PSI drift and mix reports |
| `models.py` | SQLAlchemy schema (applications, model runs, letters, users, hash-chained audit), indexes |
| `demo_data.py` | synthetic demo queue seeding |
| `synth_data.py` | reproducible HELOC generator (CC0) |
| `security.py` | rate limiter, roles, encrypted secrets, retention purge, CLI |
| `*.html` | Jinja templates (`ui.html` holds the macros) |
| `styles.css`, `app.js`, `scene.js`, `three.module.min.js` | stylesheet, progressive-enhancement script, local three.js scene on the sign-in page |
| `test_*.py`, `conftest.py` | test suites (temporary databases only) |
| `MODEL_CARDS.md`, `DATASETS.md`, `PROJECT_NOTES.md` | generated model cards, dataset comparison, engineering notes |

Generated at training time and ignored by git: `model_scorecard.pkl`, `model_forest.pkl`, `model_gbm.pkl`,
`model_registry.json`, `training_report.json`. The demo database `equilead-demo.sqlite3` is also ignored.

## Models

`python train.py` generates 12,000 synthetic rows (seed 20260916), splits them 60/20/20 stratified, runs 5-fold
cross-validation on the training split, fits each model, calibrates it with isotonic regression on the validation
split, and reports everything on the untouched test split. Each artifact is stored with its SHA-256; `scoring.py`
refuses an artifact whose hash no longer matches the registry.

| Key | Model | Explainer | Why it is here |
|---|---|---|---|
| `scorecard` | Regularised logistic regression on standardised inputs (points table derived) | coefficient contributions (log-odds) | interpretable baseline that anchors reason codes |
| `forest` | Random forest with class weighting | SHAP TreeExplainer (probability) | captures interactions, uncorrelated second opinion |
| `gbm` | XGBoost (falls back to scikit-learn HistGradientBoosting if XGBoost is not installed) | SHAP TreeExplainer (log-odds) | sensitive to non-linear thresholds such as CLTV caps |

Every model returns: name, version, artifact hash, calibrated probability of default, raw (uncalibrated) score, risk
band A–E, recommendation, decision threshold and band cut-offs, top positive and negative drivers with observed
values, explainer method, processing time, and whether the inputs were inside the training ranges. The raw
class-weighted score is never presented as a probability; only the calibrated value is.

Risk bands (`policy.py`, `bands-v1`): A < 5% approve · B < 10% approve · C < 20% conditional approval ·
D < 35% manual review · E ≥ 35% decline. The models page shows, per model, that observed default rates rise
across the bands on the test split.

## Consensus and policy

1. Each available model produces a recommendation from its calibrated probability.
2. **Unanimous** → that recommendation. **Minor** disagreement (one step apart and probability spread ≤ 10 points) →
   the more conservative recommendation. **Material** disagreement → manual review, with the reason spelled out.
   Fewer than two working models → manual review.
3. Deterministic policy rules (`policy-v1`: CLTV caps by occupancy, underwater property, DTI limits, credit-score
   floors, delinquencies, inquiries, income verification, documentation, reserves, employment, investment
   occupancy) are evaluated on the calculated figures. The policy result can only make the outcome **more**
   conservative, and when it does the explanation says so.

The consensus panel lists the explanation lines, the median probability, the spread, the agreement type and the
versions of the bands and policy that were applied.

## Explainability and reason codes

Three layers are kept separate on purpose:

* **Model explanation** (SHAP / coefficients / perturbation): shown to staff, per model, labelled with the method.
* **Policy result**: rule id, observed value, threshold and outcome.
* **Customer reasons**: entries from the controlled table in `reason_codes.py` (`reason-codes-v2`). Each entry
  carries a stable code, fixed customer wording, internal wording, the source feature or rule, the observed value,
  the comparison, the rule/model version and an approved-for-customer-use flag. Policy rules produce reasons first;
  a model driver becomes a reason only if it is a risk-raising driver on an eligible feature whose observed value
  crosses the documented comparison. Features such as loan purpose or interest rate can never become customer reasons.

Counterfactual rows (“what would change the estimate”) change one input, re-run all models and are labelled as
estimates, never advice or commitments.

## Workflow, human decision and letters

States: Draft → Submitted → Scored / Manual review → Approved / Conditionally approved / Declined → Letter generated →
Finalized, with Withdrawn reachable from any open state (`workflow.py`). Only a loan officer can record a decision.
A decision that differs from the combined recommendation, including resolving a manual-review routing, requires a
written explanation of at least 20 characters; sign-off name and role are required. Finalizing requires a recorded
decision, sign-off, an override explanation where relevant and an approved letter.

Letters come from five controlled templates (approval, conditional approval, manual review, incomplete file, adverse
action). The checker flags protected-class references, subjective language, unsupported conclusions, guarantees,
vague denials, jargon (including model names and SHAP terms), and reasons that are not in the stored record or a
decision statement that contradicts it. The page shows the original draft, the findings and the corrected draft; a
human must approve the corrected text, and approval is refused while blocking findings remain. Letters are never
sent by the system.

## Fairness dashboard

`train.py` records `sex` and `age_band` for every synthetic row **outside** the feature set and computes, per model
and attribute: selection rate, demographic parity difference, disparate impact ratio (four-fifths reference), true
positive / false positive / false negative rates, equal-opportunity difference, sample sizes, small-group warnings
(n < 30 not interpreted, n < 100 low confidence) and a review status. The page explicitly does not claim that any
model is fair; it surfaces what a person should look at. Because the synthetic attributes are independent of the
target by construction, the numbers demonstrate the metrics rather than any real disparity.

## Audit timeline and monitoring

Every action (creation, submission, scoring, edits with previous and updated values, decisions, overrides, letter
versions and approvals, finalization, withdrawal, exports, sign-ins) is appended to a hash-chained table. Each event
stores the previous event's hash; `verify_audit_chain` (shown on the audit page) detects any edited or deleted row.
Model runs are never overwritten: re-scoring adds a new batch. The monitoring page compares recent applications with
the training reference using PSI and shows recommendation, decision, agreement and override mixes.

## Security

* Session cookies (`SameSite=Lax`, `Secure` when `COOKIE_SECURE=1`), PBKDF2 password hashing, CSRF tokens on every
  form and JSON POST, in-process rate limiting on POST routes, security headers, HTML auto-escaping.
* Roles (analyst / reviewer / admin) enforced server-side; applications are scoped to their owner for analysts.
* Assets served only from an allow-list; no path traversal; no directory listing; API docs disabled.
* Secrets come from environment variables (`.env.example` documents them, never real values); optional Fernet-encrypted
  secrets file via `security.py`; optional retention purge.
* Logs never contain complete application records; only the applicant's name is stored as free text.
* CSV export neutralises spreadsheet formulas.

## API

All endpoints use the workspace session and the CSRF token (`<body data-csrf>` / `X-CSRF-Token` header).

| Route | Purpose |
|---|---|
| `POST /api/calculate` | derived figures for a (possibly incomplete) application; returns field errors instead of numbers |
| `POST /api/score` | `{"application": {...}}` → typed `ScoreResponse` (derived, policy, models, consensus, reason codes, counterfactuals, hash, latency); does not persist |
| `GET /api/applications/{id}` | stored summary, derived figures and latest model runs |
| `GET /api/models` | registry summary and model status |
| `GET /healthz` | status, demo flag, loaded models and errors |

## Tests

```bash
python -m pytest -q -p no:cacheprovider                     # everything
python -m pytest -q -p no:cacheprovider test_smoke.py       # routes, persistence, roles, CSRF, audit, export, API
python -m pytest -q -p no:cacheprovider test_browser.py     # Playwright (skips without Chromium)
```

| Suite | Covers |
|---|---|
| `test_calculations.py` | formulas, division by zero, outliers, documents, validation of impossible values |
| `test_scoring.py` | output schema, determinism, bands, consensus (unanimous / minor / material / policy precedence / single model), missing-artifact fallback, SHAP fallback, out-of-range flags, counterfactual labelling |
| `test_reason_codes.py` | table integrity, policy-first assignment, approved wording, forbidden features |
| `test_letters.py` | template contents, every prohibited-language category, record mismatches, correction |
| `test_workflow.py` | allowed and blocked transitions, finalize guard, override requirement |
| `test_smoke.py` | routes, CSRF, allow-listed assets, owner scoping, role checks, override explanation, letter approval, finalization, re-scoring history, edit audit with previous values, chain tamper detection, CSV injection, JSON API |
| `test_browser.py` | sign-in, live server-side figures, step validation, override reveal, mobile overflow, no-WebGL fallback |

Tests always use a temporary SQLite file; the demo database is never touched, and nothing creates a folder here.

## Datasets, licences and acknowledgements

EquiLead's data is synthetic and generated by `synth_data.py` (CC0). Three public repositories were studied for
ideas; no code was copied, and no external dataset is used at runtime. Full comparison in `DATASETS.md`.

* **Aegis-Credit** (MIT code, CC0 synthetic data): feature contract, calibration split, threshold economics,
  band validation, governance workflow.
* **Explainable-Credit-Risk-Scoring** (MIT; uses the UCI *Default of credit card clients* dataset, CC BY 4.0):
  interpretable scorecard alongside tree models, selection-rate metrics.
* **CreditIQ** (no licence file, therefore ideas only; its CSV has unclear provenance): XGBoost + SHAP, 5-fold CV,
  train-only clipping thresholds, disparate-impact check.

Third-party code shipped in this folder: three.js (MIT, see `THREE-LICENSE.txt`). Python dependencies are listed in
`requirements.txt` under their own licences.

## Limitations

* Synthetic data only; the default mechanism is an assumption and the models are not validated for real lending.
* Fairness metrics on independent synthetic attributes cannot reveal real disparate impact.
* Reason codes and letter templates have not been reviewed by counsel against ECOA/Regulation B, FCRA or state law.
* The rate limiter and session store are in-process; SQLite is single-file. Fine for a workstation, not for a fleet.
* The language checker is rule-based; it catches listed patterns, not every problematic sentence.
* No document ingestion, identity verification, valuation feed or core-banking integration.

## What real production use would require

Real, licensed outcome data and a model built and validated on it; model-risk-management review (documentation,
challenger models, ongoing performance monitoring with actual outcomes); fair-lending testing on real populations
with legal review; compliance review of reason codes, letters and timing requirements; a hardened deployment
(managed database, shared rate limiting, key management, TLS, SSO, backups, retention policy); penetration testing;
and an operating model in which loan officers, compliance and model owners each sign off.
