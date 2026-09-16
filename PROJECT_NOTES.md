# Project notes

## Structure rule

Every file sits directly in this folder. No `src`, `templates`, `static`, `tests` or other subfolders, and no
migration to React or Vite. Jinja templates, CSS, JS and the local three.js build are served by `app.py` from
an explicit allow-list (`ASSETS`). Tests write their databases to the system temp directory, and
`run.py` sets `sys.dont_write_bytecode`, so nothing generates a folder here. Before handing over:

```bash
python -m pytest -q -p no:cacheprovider test_smoke.py
find . -mindepth 1 -type d -print      # must print nothing
```

## Visual direction (keep)

Editorial finance workspace: soft paper background, forest-green type and controls, muted brass accents,
architectural and geometric motifs, calm human copy, generous spacing. No neon, glows, blobs, stock imagery,
robot graphics, heavy gradients, fake analytics or unexplained AI labels. The interactive Equity House in
`scene.js` is a local three.js model with drag-to-rotate, arrow-key rotation, pause and reset controls,
`prefers-reduced-motion` support and an SVG fallback when WebGL is unavailable.

## Decisions made in the handoff pass

* **Fixed** raw JSON 422 responses for bad path/query values (now the styled error page), a mobile horizontal
  scroll caused by the visually hidden table header label (`.table-scroll{position:relative}`), and a
  malformed `close` icon path.
* **Form errors return 422** (was 200) so tooling can tell a rejected submission from a rendered form.
  The analysis-failure path returns 503.
* **Score wording.** The raw class-weighted score is labelled *uncalibrated* everywhere. When
  `calibrator.pkl` is present the calibrated probability drives the decision and the raw score is shown beside it.
* **Reason codes replace SHAP for customers.** `reason_codes.py` holds the table and its approval state; the draft
  email prompt receives only coded statements. Occupation (`JOB`) can never become a reason.
* **Model card is mandatory in research mode.** Version, dataset, training date, threshold and calibration method
  are copied onto every decision record and into the CSV. A card that lists protected attributes as inputs is refused.
* **Roles** are stored on the user row (`analyst`, `reviewer`, `admin`) and in the session. Demo mode offers both
  analyst and reviewer sign-in. There is no self-service promotion.
* **Immutable records.** Decision rows carry `prev_hash`/`record_hash` over the model output; human review fields
  are outside the sealed payload and every change is an `audit_events` row in its own chain. Retention purges
  store the hashes they remove so verification still passes.
* **Rate limiting** is an in-process fixed window on POST routes. Behind a load balancer set `TRUST_PROXY=1`
  and replace `RateLimiter` with a shared store.
* **Secrets at rest.** Optional Fernet-encrypted `secrets.enc` unlocked by `SECRETS_KEY`; values fill env vars that
  are empty. The key itself must come from the host's secret manager.

## Open risks

1. **Compliance sign-off is not code.** Reason-code statements, the fair-lending review, and the model card fields
   are placeholders until legal, compliance and model-risk reviewers approve them. The UI says so on every decision.
2. **Protected-class testing needs data this app must not collect.** `fairness.py` computes the tests for whatever
   governed grouping is supplied; the demo only groups by occupation and loan purpose, which are model inputs.
3. **The rate limiter and session store are per process.** Multi-worker deployments need shared storage.
4. **SQLite** is fine for a workstation. Set `DATABASE_URL` to PostgreSQL before any shared use; the hash chains do
   not depend on the engine.
5. **Language review is an LLM judgement**, not a fairness guarantee. Escalation thresholds (0.3 / 0.7) are unvalidated.
6. **Calibration is opt-in.** Without `calibrator.pkl` the threshold is applied to an uncalibrated score.
7. **Browser tests need Chromium** installed for Playwright; they skip otherwise, so CI must install it to get coverage.

## Roadmap

* Compliance review of `reason_codes.py`, then set `reason_codes_approved` on the model card.
* Governed monitoring dataset for protected-class fairness tests; record the review on the model card.
* Shared rate limiter and session store; PostgreSQL; admin screen for roles and retention.
* Calibrator training script and validation of escalation thresholds against labelled outcomes.
* Scheduled `monitoring.py` runs with alerting on PSI above 0.25 or a four-fifths failure.
