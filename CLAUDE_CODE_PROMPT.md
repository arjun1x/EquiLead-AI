# Working brief for EquiLead

You are a senior full-stack engineer taking over the EquiLead project in this folder.

## Read first

`README.md`, `PROJECT_NOTES.md`, `app.py`, `models.py`, `demo_data.py`, `pipeline.py`, `security.py`,
`model_card.py`, `reason_codes.py`, `fairness.py`, `monitoring.py`, the `*.html` templates, `styles.css`,
`app.js`, `scene.js`.

## Structure rule

Keep every file directly inside this folder. Do not create `frontend`, `backend`, `src`, `templates`, `static`,
`components`, `public`, `node_modules`, or any other subfolder. Do not migrate to React or Vite unless absolutely
necessary. Remove generated `__pycache__` and `.pytest_cache` folders before finishing.

## Run and test

```bash
python -m pip install -r requirements.txt
python run.py
python -m pytest -q -p no:cacheprovider test_smoke.py
python -m pytest -q -p no:cacheprovider test_browser.py
find . -mindepth 1 -type d -print      # must return nothing
```

Test these workflows after any change: login page, demo login (analyst and reviewer), overview dashboard,
the three-step application with live equity and LTV figures, approval / decline / human-review demo outcomes,
decision detail (recommendation, explanation, reason codes, draft, language review, human decision, model card),
archive search, filters and pagination, CSV export, monitoring, guide, mobile layout, keyboard navigation,
reduced-motion mode, and the SVG fallback when WebGL is unavailable.

## Preserve

* The visual direction: editorial finance workspace, soft paper background, forest-green typography and
  controls, muted brass accents, architectural and geometric design language, calm professional copy,
  generous spacing and clear hierarchy.
* The local interactive three.js Equity House in `scene.js`: local build, keyboard accessible, drag-to-rotate,
  pause and reset controls, `prefers-reduced-motion`, SVG fallback. Never a remote model or external embed.
* Demo safety: demo mode clearly labelled; demo outcomes, scores and reason codes remain synthetic fixtures;
  demo data is never presented as a real lending decision.
* Research mode fails closed when model artifacts, the model card, or the Anthropic API key are missing.
* No email is ever sent automatically.
* Model recommendation, model explanation, adverse-action reasons, communication draft, language review and
  human decision stay visibly separate.
* The raw class-weighted score is never called a calibrated probability.

## Do not add

Neon colours, glowing blobs, generic AI robot graphics, stock images, excessive gradients, fake analytics,
unexplained AI labels, or futuristic dashboard styling.

## Do not commit

`.env`, `secrets.enc`, any `*.sqlite3`, screenshots, customer data, or production model artifacts.

## When finishing

Report: a concise summary of changes, exact run commands, tests completed, remaining risks, and confirmation
that the folder still contains no subfolders.
