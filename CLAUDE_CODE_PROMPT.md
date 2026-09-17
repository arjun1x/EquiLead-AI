# Working brief for EquiLead

You are a senior full-stack engineer with lending-risk and responsible-AI experience taking over the EquiLead
project in this folder.

## Read first

`README.md`, `PROJECT_NOTES.md`, `DATASETS.md`, `MODEL_CARDS.md`, then `app.py`, `calculations.py`, `policy.py`,
`scoring.py`, `explain.py`, `reason_codes.py`, `letters.py`, `workflow.py`, `train.py`, `models.py`, `security.py`,
`fairness.py`, `monitoring.py`, `demo_data.py`, `synth_data.py`, the `*.html` templates, `styles.css`, `app.js`.

## Non-negotiables

* Every file stays directly in this folder. No subfolders, no build tools, no Node.
* Synthetic data only. Never add real applicant data, credentials or production artifacts to the repository.
* Models recommend; a human decides. Never add a path that finalizes, approves, declines or sends anything without a
  recorded human decision, sign-off and approved letter.
* Keep model explanation, policy result and customer reasons separate. Customer wording comes only from the
  controlled reason-code table and the letter templates.
* Protected attributes are never model features.
* Never present a raw score as a probability, a demo result as a real decision, or the software as production-ready.

## Before finishing any change

```bash
python -m pytest -q -p no:cacheprovider
find . -mindepth 1 -type d -not -name .git -print     # must be empty
```

Report: what changed, which files, how you ran it, what you tested, remaining limitations.
