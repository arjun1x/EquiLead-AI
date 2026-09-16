# EquiLead

A flat, single-folder FastAPI + Jinja workspace for home-equity loan analysis: a model recommendation, an
engineering explanation, compliance reason codes, a reviewed communication draft, and a separate human decision,
with the records, roles, and monitoring a lender would need around it.

Everything lives directly in this folder. There is no build step and no subfolder.

## Run

```bash
python -m pip install -r requirements.txt
python run.py
```

Open http://127.0.0.1:8000 and choose **Explore demo workspace** (as an analyst or a reviewer).
`API_PORT` changes the port; copy `.env.example` to `.env` for any other setting.

## Modes

| | Demo mode (default, `DEMO_MODE=1`) | Research mode (`DEMO_MODE=0`) |
|---|---|---|
| Scores, explanations, reason codes | Fixed synthetic fixtures, labelled as such on every screen and in the CSV | XGBoost + SHAP through `pipeline.py` |
| Requirements | none beyond `requirements.txt` | `xgb_model.json`, `label_encoders.pkl`, `model_card.json`, `ANTHROPIC_API_KEY`, `SESSION_SECRET`, ML packages |
| Missing prerequisite | n/a | The server refuses to start (secret, model card) or the analysis fails closed before any computation or API call |

Research mode never sends email. Drafts are shown for a loan officer to review.

## What happens to an application

1. **Recommendation.** The model score is compared with the threshold on the model card. Without a calibrator the
   score is a class-weighted model output and is labelled *uncalibrated*, never a probability.
2. **Explanation.** SHAP contributions, shown as an engineering view only.
3. **Adverse-action reasons.** For a decline, risk-raising features are mapped to the reason-code table in
   `reason_codes.py` (at most four, deduplicated, occupation excluded). The table is marked *pending compliance
   approval* until `reason_codes_approved` is true on the model card.
4. **Communication draft.** Written from the coded reasons, then scored by a separate language reviewer.
   Anything above the escalation threshold, or any reason without a code, requires human review.
5. **Human decision.** A reviewer records approve/decline, a note, and later the observed outcome. It is stored
   apart from the model output and every change is an audit event.
6. **Records.** Each decision is sealed with a hash chained to the previous one; workspace actions form a second
   chain. `python security.py verify` checks both.

## Roles

`analyst` (default on registration) sees only their own applications. `reviewer` sees the whole workspace,
records human decisions, and opens **Monitoring**. `admin` is reserved for operators.
Promote a user with `python security.py set-role EMAIL reviewer`.

## Governance tooling

```bash
python security.py verify              # hash chains
python security.py purge --days 365    # retention (keeps hashes of purged rows in the audit chain)
python security.py keygen              # SECRETS_KEY for secrets.enc; then: python security.py encrypt-secrets ANTHROPIC_API_KEY=...
python monitoring.py --days 90         # PSI drift and subgroup performance, also at /monitoring for reviewers
python fairness.py --group job reason  # four-fifths and error-rate gaps for any governed grouping
```

Rate limiting for POST routes is on by default (`RATE_LIMIT_PER_MINUTE`, in-process).

## Tests

```bash
python -m pytest -q test_smoke.py      # routes, CSRF, validation, scoping, filters, CSV, roles, chains, governance
python -m pytest -q test_browser.py    # desktop, tablet, mobile, reduced motion, no-WebGL (needs playwright + chromium)
```

Both suites use a temporary database and never touch `equilead-demo.sqlite3`.

## Files

`app.py` routes · `models.py` schema and hash chains · `demo_data.py` synthetic fixtures · `pipeline.py` research bridge ·
`model_card.py` metadata · `reason_codes.py` adverse-action table · `fairness.py` subgroup tests · `monitoring.py` drift ·
`security.py` roles, rate limit, secrets, retention · `*.html` templates · `styles.css` · `app.js` · `scene.js` +
`three.module.min.js` (local, MIT) · `run.py`.

See `PROJECT_NOTES.md` for decisions and open risks, and `CLAUDE_CODE_PROMPT.md` for the working brief.
