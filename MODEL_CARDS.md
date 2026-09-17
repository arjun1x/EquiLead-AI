# EquiLead model cards

Generated 2026-09-16T16:53:43+00:00 from `train.py` (seed 20260916). All three models are trained on the same synthetic
dataset (`synthetic-heloc-v1`, 12000 rows, simulated 24-month default rate 8.7%, CC0). Metrics describe the
generator's simulated relationship only. None of these models is validated for real U.S. home-equity underwriting.

| Model | Version | Test ROC-AUC | Test PR-AUC | Brier | CV ROC-AUC | Cost-optimal threshold (info) |
|---|---|---|---|---|---|---|
| Policy scorecard (logistic regression) | scorecard-1.0.0-20260916 | 0.9281 | 0.6286 | 0.04702 | 0.9255 | 0.15 |
| Random forest | forest-1.0.0-20260916 | 0.9191 | 0.592 | 0.05021 | 0.9201 | 0.2 |
| Gradient boosting | gbm-1.0.0-20260916 | 0.9233 | 0.5706 | 0.05046 | 0.9203 | 0.2 |

## Policy scorecard (logistic regression)

* **Version:** `scorecard-1.0.0-20260916` · file `model_scorecard.pkl` · SHA-256 `26ad8ab36a8b6c20…` · trained 2026-09-16T16:53:43+00:00
* **Algorithm:** sklearn LogisticRegression (L2, C=0.5) · preprocessing `prep-v1` · isotonic on the validation split
* **Intended use:** decision support inside the EquiLead demonstration; one of three models feeding a transparent consensus that a loan officer must review.
* **Prohibited use:** autonomous credit decisions, real applicants, generating regulatory adverse-action notices, any use without independent validation on approved data.
* **Training data:** synthetic HELOC generator `synthetic-heloc-v1`, 7200 training / 2400 calibration / 2400 test rows.
* **Features:** 17 numeric + 5 categorical application fields (see calculations.MODEL_FEATURES). No protected attributes.
* **Target:** `defaulted_24m` (simulated serious delinquency within 24 months).
* **Metrics (test, threshold 0.35):** ROC-AUC 0.9281, PR-AUC 0.6286, precision 0.6416, recall 0.5337, F1 0.5827, Brier 0.04702 (uncalibrated 0.04619); CV ROC-AUC 0.9255 ± 0.0092.
* **Thresholds:** risk bands `bands-v1` (approve < 10%, conditional < 20%, manual review < 35%, decline otherwise). Cost-optimal threshold 0.15 is informational only.
* **Band validation (test):** A 0.0084 (n=1549); B 0.042 (n=238); C 0.0987 (n=304); D 0.3235 (n=136); E 0.6416 (n=173)
* **Known limitations:** synthetic labels; single-family U.S.-style assumptions baked into the generator; no macro-economic or property-market features; isotonic calibration fitted on 2,400 rows; explanations describe the uncalibrated base model.
* **Fairness considerations:** evaluated on synthetic `sex` and `age_band` columns that are never model inputs; see training_report.json and the Fairness page. Signals require human and legal review and are not a fairness certification.

## Random forest

* **Version:** `forest-1.0.0-20260916` · file `model_forest.pkl` · SHA-256 `b24c7367625d6e8d…` · trained 2026-09-16T16:53:43+00:00
* **Algorithm:** sklearn RandomForestClassifier (300 trees, depth 10, min leaf 20) · preprocessing `prep-v1` · isotonic on the validation split
* **Intended use:** decision support inside the EquiLead demonstration; one of three models feeding a transparent consensus that a loan officer must review.
* **Prohibited use:** autonomous credit decisions, real applicants, generating regulatory adverse-action notices, any use without independent validation on approved data.
* **Training data:** synthetic HELOC generator `synthetic-heloc-v1`, 7200 training / 2400 calibration / 2400 test rows.
* **Features:** 17 numeric + 5 categorical application fields (see calculations.MODEL_FEATURES). No protected attributes.
* **Target:** `defaulted_24m` (simulated serious delinquency within 24 months).
* **Metrics (test, threshold 0.35):** ROC-AUC 0.9191, PR-AUC 0.592, precision 0.6456, recall 0.4904, F1 0.5574, Brier 0.05021 (uncalibrated 0.05298); CV ROC-AUC 0.9201 ± 0.0105.
* **Thresholds:** risk bands `bands-v1` (approve < 10%, conditional < 20%, manual review < 35%, decline otherwise). Cost-optimal threshold 0.2 is informational only.
* **Band validation (test):** A 0.0044 (n=1353); B 0.0518 (n=541); C 0.1583 (n=120); D 0.2325 (n=228); E 0.6456 (n=158)
* **Known limitations:** synthetic labels; single-family U.S.-style assumptions baked into the generator; no macro-economic or property-market features; isotonic calibration fitted on 2,400 rows; explanations describe the uncalibrated base model.
* **Fairness considerations:** evaluated on synthetic `sex` and `age_band` columns that are never model inputs; see training_report.json and the Fairness page. Signals require human and legal review and are not a fairness certification.

## Gradient boosting

* **Version:** `gbm-1.0.0-20260916` · file `model_gbm.pkl` · SHA-256 `73e8962932875cdd…` · trained 2026-09-16T16:53:43+00:00
* **Algorithm:** XGBoost XGBClassifier (400 rounds, depth 4, eta 0.05) · preprocessing `prep-v1` · isotonic on the validation split
* **Intended use:** decision support inside the EquiLead demonstration; one of three models feeding a transparent consensus that a loan officer must review.
* **Prohibited use:** autonomous credit decisions, real applicants, generating regulatory adverse-action notices, any use without independent validation on approved data.
* **Training data:** synthetic HELOC generator `synthetic-heloc-v1`, 7200 training / 2400 calibration / 2400 test rows.
* **Features:** 17 numeric + 5 categorical application fields (see calculations.MODEL_FEATURES). No protected attributes.
* **Target:** `defaulted_24m` (simulated serious delinquency within 24 months).
* **Metrics (test, threshold 0.35):** ROC-AUC 0.9233, PR-AUC 0.5706, precision 0.6842, recall 0.4375, F1 0.5337, Brier 0.05046 (uncalibrated 0.05034); CV ROC-AUC 0.9203 ± 0.0067.
* **Thresholds:** risk bands `bands-v1` (approve < 10%, conditional < 20%, manual review < 35%, decline otherwise). Cost-optimal threshold 0.2 is informational only.
* **Band validation (test):** A 0.008 (n=1625); B 0.0667 (n=225); C 0.1327 (n=211); D 0.2961 (n=206); E 0.6842 (n=133)
* **Known limitations:** synthetic labels; single-family U.S.-style assumptions baked into the generator; no macro-economic or property-market features; isotonic calibration fitted on 2,400 rows; explanations describe the uncalibrated base model.
* **Fairness considerations:** evaluated on synthetic `sex` and `age_band` columns that are never model inputs; see training_report.json and the Fairness page. Signals require human and legal review and are not a fairness certification.
