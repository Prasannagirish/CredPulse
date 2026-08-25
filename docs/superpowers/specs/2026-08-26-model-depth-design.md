# CreditPulse Model Depth — Design (Phases 6-7)

Date: 2026-08-26
Status: Draft for review

## 1. Purpose

Phases 1-5 built a complete MLOps lifecycle (data → model → drift → retrain/promote →
MCP server → quantified backtest) but the modeling substance itself is thin: one fixed
XGBoost hyperparameter config reused everywhere, no calibration check, no baseline
comparison against the industry-standard interpretable model, and — given the project
already frames `explain_prediction` in adverse-action/ECOA language — no fairness analysis
at all. This work closes that gap with two new phases: **Phase 6 (model selection)** and
**Phase 7 (fairness and sensitivity)**.

Both phases produce a real report run against real data and a README section, following the
exact pattern established in Phases 1-5.

## 2. Phase 6 — Model Selection

### 2.1 Hyperparameter tuning

`model/tune.py`, using **Optuna** (new dependency — added to `pyproject.toml`).

- Objective: maximize AUC on the same temporal holdout `model.train.train_baseline` already
  uses (the chronological last 10% of the pre-2019 reference window). Reusing this exact
  split means tuning introduces no new leak surface — it's the same boundary every other
  phase already respects.
- Search space: `n_estimators` (100-500), `max_depth` (3-8), `learning_rate` (0.01-0.3, log
  scale), `subsample` (0.6-1.0), `colsample_bytree` (0.6-1.0), `min_child_weight` (1-10).
- Budget: 30 trials — a deliberately bounded, real (not exhaustive) search given local
  compute; documented as such in the report, not presented as a full grid search.
- Decision rule, applied automatically and reported either way: if the best trial's AUC
  beats `train_baseline`'s current default config by more than 0.005, `model/train.py`'s
  `build_model_pipeline()` is updated to the tuned hyperparameters — a real, committed model
  change. If not, the default config is kept and the report states plainly that tuning did
  not find a meaningfully better configuration. Both outcomes are legitimate; neither is
  assumed ahead of running it.

### 2.2 Baseline comparison — Logistic Regression + WOE

`model/baseline_woe.py`, hand-rolled (no new dependency — matches the project's existing
"only add what's needed" pattern, and demonstrates understanding of the technique rather
than delegating to a library).

- `fit_woe_bins(reference_df, numeric_features, categorical_features, n_bins=10) ->
  WOEBins` — quantile bin edges per numeric feature (frozen, same discipline as
  `drift/statistical.py`'s reference bins), natural categories for categoricals. WOE per
  bin/category = `ln(%good / %bad)` where good = non-default, computed only on the
  reference window.
- `transform_woe(bins: WOEBins, df: pd.DataFrame) -> pd.DataFrame` — replaces each raw
  feature value with its bin's WOE value.
- `train_logistic_baseline(reference_df) -> LogisticRegression` fit on WOE-transformed
  reference features.
- Comparison: AUC and Brier score for the logistic+WOE baseline vs. the (possibly newly
  tuned) XGBoost champion, on the same held-out temporal slice used throughout. A written
  paragraph on the interpretability tradeoff — each WOE coefficient is directly explainable
  as scorecard points added/subtracted, which is why logistic+WOE scorecards remain an
  industry norm in regulated lending, versus XGBoost's higher AUC but SHAP-dependent
  explainability.

### 2.3 Calibration analysis

`model/calibration.py`.

- `reliability_diagram_data(pipeline, df, n_bins=10) -> pd.DataFrame` — buckets predicted
  probabilities into deciles, computes mean predicted probability vs. observed default rate
  per bucket.
- `expected_calibration_error(pipeline, df, n_bins=10) -> float` — standard ECE: the
  bucket-size-weighted average of `|mean predicted prob - observed default rate|`.
- Run against the final champion (post-tuning) on the same held-out slice. Produces a
  reliability diagram chart and an honest, ECE-driven discussion of whether raw
  `predict_proba` output is usable directly for pricing, or would need Platt
  scaling/isotonic regression — written after seeing the real number, not assumed.

### 2.4 Report

`reports/generate_phase6_report.py` — runs tuning, then baseline comparison, then
calibration (in that order, since baseline/calibration analyze whichever config tuning
lands on), against real data. Saves `reports/phase6_tuning_comparison.png` and
`reports/phase6_calibration.png`. Updates `README.md` with a new Phase 6 section.

## 3. Phase 7 — Fairness and Sensitivity

### 3.1 Fairness / disparate-impact check

`backtest/fairness.py`.

**Stated limitation, up front, in both the code's module docstring and the report output:**
Lending Club's public dataset contains no protected-class labels (race, sex, age, national
origin, marital status) — genuine fair-lending analysis is not possible on this data. This
analysis instead uses two literature-grounded **proxies**: `home_ownership`
(RENT/OWN/MORTGAGE, which correlates with age and accumulated wealth) and income quartiles
of `annual_inc` (which correlate historically with race and ethnicity per fair-lending
research). Results describe disparate impact along these proxies, not genuine protected-class
outcomes, and are a screening heuristic, not a legal or regulatory determination.

- `group_approval_rates(pipeline, df, group_col, approval_rate) -> pd.DataFrame` — same
  "accept the lowest-risk N%" mechanic as `backtest.impact.dollar_impact`, broken out per
  group: each group's approval rate and default rate among its approved loans, at the
  project's existing fixed 80% approval threshold.
- `disparate_impact_ratio(group_rates: pd.DataFrame) -> dict` — minimum group approval rate
  divided by maximum group approval rate (the standard "4/5ths rule" screen used as a first
  pass in US fair-lending analysis), flagged if below 0.8.
- Run against the final champion, on the same reliable evaluation window Phase 5 used.
  Saves `reports/phase7_fairness.png` (grouped bar chart of approval/default rates).

### 3.2 Approval-rate sensitivity

Extends `backtest/impact.py` usage rather than its interface — no new function signature
needed, `dollar_impact` already accepts `approval_rate`.

`backtest/sensitivity.py`: sweeps `approval_rate` across `[0.5, 0.6, 0.7, 0.8, 0.9, 0.95]`,
calling `dollar_impact` for both the static and the final/monitored model at each rate on
Phase 5's final reliable month. Produces a curve (`reports/phase7_approval_sensitivity.png`)
of loss-per-10,000-loans vs. approval rate for both, replacing Phase 5's single 80%-only
data point with the full tradeoff curve.

### 3.3 Report

`reports/generate_phase7_report.py` — runs the fairness check then the sensitivity sweep
against real data. Updates `README.md` with a new Phase 7 section.

## 4. Out of scope

- No mitigation strategies for any disparate impact found (reweighting, adversarial
  debiasing, threshold adjustment per group) — Phase 7 measures and reports, it does not
  attempt to fix. Explicitly noted as a follow-up, not silently omitted.
- No hyperparameter search beyond the 30-trial Optuna budget — not a claim of finding the
  global optimum, a claim of a real, bounded, honest search.
- No deployment of the WOE/logistic baseline as an alternative champion — it exists as a
  comparison point in the report, not a second model the registry tracks.
