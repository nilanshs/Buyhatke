# Demographic Inference from Transactional Data — README

## Project overview

This repository implements a **rule-based demographic inference** pipeline that predicts **age groups** (`<25`, `25–40`, `40+`) and **gender** (`male`, `female`) for users using **transaction-only data** (no explicit demographics). The approach is intentionally interpretable for assessment use: feature engineering -> rule-based scoring with synergy terms -> probabilities via softmax.

**Key idea:** transform transaction-level data into user-level behavioral signals (category/brand spend share, spending power, frequency, time-of-day), apply weighted rules and interaction (synergy) terms, and produce probabilistic demographic labels per user.

---

## Files included / produced

* `analyzed_audio_data.xlsx` — raw transactional dataset (input).
* `user_features.csv` — user-level engineered features (produced by feature-engineering step).
* `user_features_scored.csv` — final scored output with probabilities and explainability columns (produced by scoring step).
* `score_users.py` / `feature_engineering.ipynb` — (optional) scripts / notebook with the full pipeline.
* `Business Analyst Intern Assignment.pdf` — assignment brief (reference).

---

## Quick start (Google Colab) — recommended (copy/paste cells)

1. Open Google Colab: [https://colab.research.google.com](https://colab.research.google.com)
2. Upload `analyzed_audio_data.xlsx` via the left Files panel or use `files.upload()`.
3. Run the **Feature engineering** cell (creates `user_features.csv`).
4. Run the **Scoring** cell (creates `user_features_scored.csv`).
5. Download outputs: `user_features.csv`, `user_features_scored.csv`.

> Both cells are copy-paste blocks in the notebook. They are defensive (handle missing columns) and print progress messages.

---

## Feature engineering (summary)

For each user we compute:

* **Aggregate metrics:** `total_spend`, `num_txns`, `avg_price`, `median_price`, `min_price`, `max_price`, `first_txn`, `last_txn`, `active_days`, `span_days`, `txns_per_day`.
* **Category % spend:** top-N `level1_name` categories → `pct_cat_<name>` = spend in category / total spend.
* **Brand % spend:** top-N brands → `pct_brand_<brand>`.
* **Time features:** `pct_night` (20:00–02:00), `pct_weekend`, `avg_hour`.
* **High-value transaction features:** `pct_high_value`, `count_high_value` (threshold configurable; default ₹2000).
* **Hypothesis flags:** `pct_audio_video`, `pct_beauty`, `pct_kids` (derived from level names).
* **Synergy (interaction) features:** e.g., `synergy_beauty_apparel = pct_beauty * pct_apparel` (used to amplify combined signals).

All features are saved to `user_features.csv`.

---

## Scoring model (rule-based) — high-level

1. **Raw scores** for gender (`male_score`, `female_score`) and age (`score_<25`, `score_25_40`, `score_40+`) are computed as linear combinations of features using pre-defined multipliers (weights).
2. **Synergy**: interaction terms multiply when two signals co-occur (e.g., Beauty × Apparel strongly increases female score).
3. **Normalization**: numeric features are normalized (min-max) so weights are stable.
4. **Convert to probabilities**: raw scores → probabilities via softmax (gender uses 2-class softmax; age uses 3-class softmax).
5. **Explainability**: top contributing components per user are saved for debugging (`top_3_components`).

### Example weight highlights (tunable)

* `female_score` gains from: `pct_beauty * 2.8`, `pct_apparel * 1.6`, `pct_footwear * 1.1`, `synergy_beauty_apparel * 4.0`, `synergy_beauty_footwear * 3.0`, female-lean brand shares ×1.3.
* `male_score` gains from: `pct_audio_video * 2.0`, `pct_high_value * 1.0`, male-lean brand shares ×1.2, `pct_night * 0.4`.
* `score_<25` gains from: `pct_night * 1.9`, high `num_txns_norm * 1.2`, `audio affinity * 1.6`, `1 - avg_price_norm * 1.0` (younger => lower avg price).
* `score_25_40` gains from: `pct_kids * 1.8`, `apparel * 1.0`, `pct_weekend * 0.9`, `txns_per_day_norm * 0.8`.
* `score_40+` gains from: `pct_high_value * 2.2`, `avg_price_norm * 1.6`, `home category * 1.3`, `1 - num_txns_norm * 0.6`.

> These weights were chosen for interpretability and to reflect domain intuition. They can be tuned or replaced with ML once labeled data are available.

---

## Scoring formula (concise)

For each user:

```
raw_gender_scores = sum(w_i * feature_i)
male_prob, female_prob = softmax(raw_male, raw_female)

raw_age_scores = [s_<25, s_25_40, s_40+]
age_probs = softmax(raw_age_scores)
```

`synergy` appears as additional `w * (feature_a * feature_b)` terms.

---

## Validation plan (when labels are available)

* **Confusion matrix** for age & gender predictions.
* **Precision / Recall / F1-score** per class (important if classes are imbalanced).
* **Lift over random baseline** (e.g., model accuracy - random accuracy).
* **A/B test or holdout**: if labels are available for a subset, hold them out for evaluation and calibrate weights.
* **Error analysis**: inspect `top_3_components` for misclassified users to refine weights or add features.

---

## Assumptions & caveats

* Transactional signals are **proxies** for demographic attributes — they are noisy and not deterministic.
* Brand-to-gender mappings and category heuristics are **market-dependent** and may require localization.
* The rule-based approach is interpretable but may underperform a supervised ML model trained on labeled demographics.
* Ethical/privacy: inferred demographics should be used responsibly and in compliance with privacy regulations.

---

## How to tune & move to ML

1. If you obtain labeled demographics for a subset, you can:

   * Use the engineered features to train classifiers (Logistic Regression, RandomForest, XGBoost).
   * Compare ML performance with the rule-based baseline (use lift, precision/recall).
2. Use feature importance (from tree-based models) to update rule weights.
3. Consider calibration (Platt scaling, isotonic) to make probabilities well-calibrated.

---

## Deliverables to submit

* `user_features.csv` (engineered features)
* `user_features_scored.csv` (predictions + explainability)
* A short methodology document (200–700 words) describing assumptions, weights, synergy, and validation plan — ready for assessment submission.

---

## Contact / Next steps

If you want, I can:

* Generate the **methodology write-up** (formatted, \~500 words) ready to attach to your submission.
* Tune the weights and re-run scoring, or train a supervised model if you upload labeled users.
* Package a single Colab notebook with the exact runnable cells (already prepared) and export it as `.ipynb`.

---

*Keyword note:* this README explicitly uses the term **synergy** to describe interaction effects (e.g., `Beauty × Apparel`) as requested by the assignment.
