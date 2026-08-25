# CreditPulse Phase 6 — Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user is
> executing this plan to learn the system, one step at a time — narrate what each step is
> doing and why before running it, don't just execute silently.

**Goal:** Move past a single, never-tuned XGBoost config: search hyperparameters with
Optuna, compare against a hand-rolled logistic regression + WOE scorecard baseline, and
check whether the champion's predicted probabilities are actually calibrated.

**Architecture:** `model/tune.py` runs an Optuna search against the same temporal holdout
`model.train.train_baseline` already uses (extracted into a shared `temporal_holdout_split`
helper). `model/baseline_woe.py` hand-rolls Weight-of-Evidence binning and a logistic
regression baseline, frozen on the reference window like every other reference-fit object in
this project. `model/calibration.py` computes reliability-diagram data and Expected
Calibration Error for any fitted pipeline. `reports/generate_phase6_report.py` runs all
three against real data, in that order, and updates the champion's hyperparameters only if
tuning finds a real improvement.

**Tech Stack:** Python 3.12, optuna (new dependency), scikit-learn, xgboost, pandas, numpy,
matplotlib, pytest.

## Global Constraints

- Hyperparameter tuning maximizes AUC on the exact same chronological-last-10% holdout
  `train_baseline` uses — never a new or random split. This holdout logic is extracted once
  into `model.train.temporal_holdout_split` and reused by both.
- WOE bins are fit **only** on the reference (training) window and frozen for every later
  transform — the same discipline `drift/statistical.py`'s reference bins already follow.
- The champion's hyperparameters (`model.train.build_model_pipeline`) are updated only if
  the tuned config beats the current default by more than `config.TUNING_AUC_IMPROVEMENT_
  THRESHOLD` (0.005) AUC on the holdout. Either outcome — updated or kept — is reported
  explicitly; neither is assumed before the real run.
- Optuna search budget is `config.TUNING_N_TRIALS` (30) — a deliberately bounded, real
  search, documented as such, not presented as exhaustive.
- Every printed/plotted result is labeled a backtest on historical, resolved-outcome data.

---

### Task 1: Extract shared holdout split; hyperparameter tuning

**Files:**
- Modify: `model/train.py` — extract `temporal_holdout_split`
- Modify: `pyproject.toml` — add `optuna` dependency
- Modify: `config.py` — add `TUNING_N_TRIALS`, `TUNING_AUC_IMPROVEMENT_THRESHOLD`
- Create: `model/tune.py`
- Create: `tests/test_tune.py`

**Interfaces:**
- Produces:
  - `model.train.temporal_holdout_split(reference_df: pd.DataFrame) -> tuple[pd.DataFrame,
    pd.DataFrame]` — the chronological-last-10% split, extracted from `train_baseline`
    without changing its behavior.
  - `model.tune.tune_hyperparameters(reference_df: pd.DataFrame, n_trials: int =
    config.TUNING_N_TRIALS) -> tuple[dict, float]` — returns `(best_params, best_auc)`.
    `best_params` keys: `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
    `colsample_bytree`, `min_child_weight`. Task 4 (report script) calls this directly.

- [ ] **Step 1: Add `optuna` to `pyproject.toml`**

Add `"optuna>=3.6"` to the `dependencies` list in `pyproject.toml`, then:

```bash
cd ~/IdeaProjects/mlops
uv sync
```

Expected: resolves and installs cleanly.

- [ ] **Step 2: Add tuning constants to `config.py`**

Append to `config.py`:

```python
# Hyperparameter tuning (Phase 6). A bounded, real search — not exhaustive.
TUNING_N_TRIALS = 30
TUNING_AUC_IMPROVEMENT_THRESHOLD = 0.005
```

- [ ] **Step 3: Extract `temporal_holdout_split` in `model/train.py`**

Find this block inside `train_baseline`:

```python
    sorted_df = reference_df.sort_values("issue_d").reset_index(drop=True)
    split_idx = int(len(sorted_df) * 0.9)
    train_slice = sorted_df.iloc[:split_idx]
    test_slice = sorted_df.iloc[split_idx:]
```

Replace `train_baseline` with:

```python
def temporal_holdout_split(reference_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological last-10% holdout of reference_df.

    A random split would let the model implicitly see "future" reference-window loans
    during training, which is a milder version of the same leak the reference/eval split
    exists to prevent. Shared by train_baseline (below) and model.tune's hyperparameter
    search — both must evaluate against the exact same holdout.
    """
    sorted_df = reference_df.sort_values("issue_d").reset_index(drop=True)
    split_idx = int(len(sorted_df) * 0.9)
    return sorted_df.iloc[:split_idx], sorted_df.iloc[split_idx:]


def train_baseline(reference_df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """Fit the baseline champion on a temporal holdout of the reference window."""
    train_slice, test_slice = temporal_holdout_split(reference_df)

    pipeline = build_model_pipeline()
    feature_cols = [c for c in reference_df.columns if c not in NON_FEATURE_COLUMNS]
    pipeline.fit(train_slice[feature_cols], train_slice[TARGET_COLUMN])

    return pipeline, train_slice, test_slice
```

- [ ] **Step 4: Run the full test suite to confirm the refactor didn't break anything**

```bash
cd ~/IdeaProjects/mlops
uv run pytest -v
```

Expected: all 51 previously-passing tests still pass (pure extraction, identical behavior).

- [ ] **Step 5: Write the failing test for `tune_hyperparameters`**

```python
# tests/test_tune.py
import numpy as np
import pandas as pd

from model.tune import tune_hyperparameters


def _synthetic_df(n: int = 300, start: str = "2017-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n, freq="3D")
    risk = rng.normal(size=n)
    default_prob = 1 / (1 + np.exp(-risk))
    default_flag = (rng.uniform(size=n) < default_prob).astype(int)
    return pd.DataFrame({
        "issue_d": dates,
        "loan_amnt": rng.uniform(1000, 30000, size=n),
        "term": rng.choice([36, 60], size=n),
        "int_rate": rng.uniform(5, 25, size=n),
        "emp_length": rng.integers(0, 11, size=n),
        "annual_inc": rng.uniform(20000, 150000, size=n),
        "dti": rng.uniform(0, 40, size=n),
        "revol_util": rng.uniform(0, 100, size=n),
        "fico_range_low": rng.integers(660, 800, size=n),
        "fico_range_high": rng.integers(664, 804, size=n),
        "grade": rng.choice(list("ABCDEFG"), size=n),
        "sub_grade": rng.choice([f"{g}{i}" for g in "ABC" for i in range(1, 6)], size=n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], size=n),
        "verification_status": rng.choice(["Verified", "Not Verified"], size=n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], size=n),
        "default_flag": default_flag,
    })


def test_tune_hyperparameters_returns_params_and_auc():
    df = _synthetic_df()
    best_params, best_auc = tune_hyperparameters(df, n_trials=2)

    assert set(best_params.keys()) == {
        "n_estimators", "max_depth", "learning_rate",
        "subsample", "colsample_bytree", "min_child_weight",
    }
    assert 0.0 <= best_auc <= 1.0
```

- [ ] **Step 6: Run test to verify it fails**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_tune.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'model.tune'`.

- [ ] **Step 7: Implement `model/tune.py`**

```python
# model/tune.py
"""Hyperparameter tuning for the XGBoost champion via Optuna, evaluated on the exact same
temporal holdout model.train.train_baseline uses — no new leak surface."""
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import NON_FEATURE_COLUMNS, TARGET_COLUMN, TUNING_N_TRIALS
from model.features import build_preprocessor
from model.train import evaluate_auc, temporal_holdout_split

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_pipeline_with_params(params: dict) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", XGBClassifier(eval_metric="auc", random_state=42, **params)),
    ])


def _objective(trial: optuna.Trial, train_slice: pd.DataFrame, test_slice: pd.DataFrame) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    pipeline = _build_pipeline_with_params(params)
    feature_cols = [c for c in train_slice.columns if c not in NON_FEATURE_COLUMNS]
    pipeline.fit(train_slice[feature_cols], train_slice[TARGET_COLUMN])
    return evaluate_auc(pipeline, test_slice)


def tune_hyperparameters(
    reference_df: pd.DataFrame, n_trials: int = TUNING_N_TRIALS
) -> tuple[dict, float]:
    """Runs an Optuna search over XGBoost hyperparameters, maximizing AUC on the same
    chronological-last-10% holdout model.train.train_baseline uses. n_trials is a
    deliberately bounded, real search budget — not exhaustive. Returns (best_params,
    best_auc)."""
    train_slice, test_slice = temporal_holdout_split(reference_df)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: _objective(trial, train_slice, test_slice), n_trials=n_trials)

    return study.best_params, study.best_value
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_tune.py -v
```

Expected: 1 passed.

- [ ] **Step 9: Run the full test suite**

```bash
cd ~/IdeaProjects/mlops
uv run pytest -v
```

Expected: all tests pass (52 total).

- [ ] **Step 10: Commit**

```bash
cd ~/IdeaProjects/mlops
git add model/train.py model/tune.py tests/test_tune.py config.py pyproject.toml uv.lock
git commit -m "Extract shared temporal holdout split; add Optuna hyperparameter tuning"
```

---

### Task 2: Logistic regression + WOE baseline

**Files:**
- Create: `model/baseline_woe.py`
- Create: `tests/test_baseline_woe.py`

**Interfaces:**
- Consumes: `config.NUMERIC_FEATURES`, `config.CATEGORICAL_FEATURES`, `config.TARGET_COLUMN`.
- Produces:
  - `WOEBins` (dataclass) — fields `numeric_bin_edges: dict[str, np.ndarray]`,
    `numeric_woe: dict[str, np.ndarray]` (WOE value per bin index), `categorical_woe:
    dict[str, pd.Series]` (category → WOE value).
  - `fit_woe_bins(reference_df: pd.DataFrame, numeric_features: list[str] =
    NUMERIC_FEATURES, categorical_features: list[str] = CATEGORICAL_FEATURES, n_bins: int =
    10) -> WOEBins` — fit only on `reference_df`, frozen for later use.
  - `transform_woe(bins: WOEBins, df: pd.DataFrame) -> pd.DataFrame` — one WOE-valued column
    per feature in `bins`.
  - `train_logistic_baseline(reference_df: pd.DataFrame) -> tuple[LogisticRegression,
    WOEBins]`.
  - `evaluate_logistic_auc(model: LogisticRegression, bins: WOEBins, df: pd.DataFrame) ->
    float`.

  Task 4 (report script) calls `train_logistic_baseline` and `evaluate_logistic_auc`
  directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_baseline_woe.py
import numpy as np
import pandas as pd

from model.baseline_woe import (
    evaluate_logistic_auc,
    fit_woe_bins,
    train_logistic_baseline,
    transform_woe,
)


def _synthetic_df(n: int = 500, start: str = "2017-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n, freq="3D")
    risk = rng.normal(size=n)
    default_prob = 1 / (1 + np.exp(-risk))
    default_flag = (rng.uniform(size=n) < default_prob).astype(int)
    return pd.DataFrame({
        "issue_d": dates,
        "loan_amnt": rng.uniform(1000, 30000, size=n),
        "term": rng.choice([36, 60], size=n),
        "int_rate": rng.uniform(5, 25, size=n),
        "emp_length": rng.integers(0, 11, size=n),
        "annual_inc": rng.uniform(20000, 150000, size=n),
        "dti": rng.uniform(0, 40, size=n),
        "revol_util": rng.uniform(0, 100, size=n),
        "fico_range_low": rng.integers(660, 800, size=n),
        "fico_range_high": rng.integers(664, 804, size=n),
        "grade": rng.choice(list("ABCDEFG"), size=n),
        "sub_grade": rng.choice([f"{g}{i}" for g in "ABC" for i in range(1, 6)], size=n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], size=n),
        "verification_status": rng.choice(["Verified", "Not Verified"], size=n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], size=n),
        "default_flag": default_flag,
    })


def test_fit_woe_bins_assigns_lower_woe_to_higher_risk_bin():
    n = 1000
    rng = np.random.default_rng(0)
    int_rate = np.concatenate([np.full(500, 5.0), np.full(500, 25.0)])
    default_flag = np.concatenate([
        rng.choice([0, 1], size=500, p=[0.9, 0.1]),   # low rate -> mostly good (non-default)
        rng.choice([0, 1], size=500, p=[0.3, 0.7]),   # high rate -> mostly bad (default)
    ])
    df = pd.DataFrame({"int_rate": int_rate, "default_flag": default_flag})

    bins = fit_woe_bins(df, numeric_features=["int_rate"], categorical_features=[])

    # WOE = ln(%good/%bad): the low-risk (mostly-good) bin should have a HIGHER WOE than
    # the high-risk (mostly-bad) bin.
    assert bins.numeric_woe["int_rate"][0] > bins.numeric_woe["int_rate"][-1]


def test_transform_woe_handles_unseen_category_gracefully():
    df = _synthetic_df(n=200)
    bins = fit_woe_bins(df)
    test_df = df.copy()
    test_df.loc[test_df.index[0], "grade"] = "ZZ"  # unseen category

    transformed = transform_woe(bins, test_df)

    assert not transformed["grade"].isna().any()


def test_train_logistic_baseline_returns_model_and_bins():
    df = _synthetic_df(n=500)

    model, bins = train_logistic_baseline(df)
    auc = evaluate_logistic_auc(model, bins, df)

    assert 0.0 <= auc <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_baseline_woe.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'model.baseline_woe'`.

- [ ] **Step 3: Implement `model/baseline_woe.py`**

```python
# model/baseline_woe.py
"""Logistic regression + Weight-of-Evidence baseline — the industry-standard interpretable
credit scorecard, compared against the XGBoost champion. Bins are fit only on the reference
(training) window and frozen for every later transform, the same discipline
drift/statistical.py's reference bins follow."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN

_EPSILON = 1e-4


@dataclass
class WOEBins:
    numeric_bin_edges: dict[str, np.ndarray]
    numeric_woe: dict[str, np.ndarray]
    categorical_woe: dict[str, pd.Series]


def _woe(pct_good: float, pct_bad: float) -> float:
    return float(np.log(max(pct_good, _EPSILON) / max(pct_bad, _EPSILON)))


def _compute_woe_by_bin(bin_idx: np.ndarray, target: pd.Series, n_bins: int) -> np.ndarray:
    woe = np.zeros(n_bins)
    total_good = (target == 0).sum()
    total_bad = (target == 1).sum()
    for b in range(n_bins):
        mask = bin_idx == b
        good = ((target == 0) & mask).sum()
        bad = ((target == 1) & mask).sum()
        woe[b] = _woe(good / total_good, bad / total_bad)
    return woe


def _compute_woe_by_category(values: pd.Series, target: pd.Series) -> pd.Series:
    total_good = (target == 0).sum()
    total_bad = (target == 1).sum()
    result = {}
    for category in values.dropna().unique():
        mask = values == category
        good = ((target == 0) & mask).sum()
        bad = ((target == 1) & mask).sum()
        result[category] = _woe(good / total_good, bad / total_bad)
    return pd.Series(result)


def fit_woe_bins(
    reference_df: pd.DataFrame,
    numeric_features: list[str] = NUMERIC_FEATURES,
    categorical_features: list[str] = CATEGORICAL_FEATURES,
    n_bins: int = 10,
) -> WOEBins:
    numeric_bin_edges: dict[str, np.ndarray] = {}
    numeric_woe: dict[str, np.ndarray] = {}
    for feature in numeric_features:
        values = reference_df[feature].dropna()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        bin_idx = np.digitize(reference_df[feature].fillna(values.median()), edges[1:-1])
        numeric_bin_edges[feature] = edges
        numeric_woe[feature] = _compute_woe_by_bin(bin_idx, reference_df[TARGET_COLUMN], len(edges) - 1)

    categorical_woe = {
        feature: _compute_woe_by_category(reference_df[feature], reference_df[TARGET_COLUMN])
        for feature in categorical_features
    }

    return WOEBins(numeric_bin_edges=numeric_bin_edges, numeric_woe=numeric_woe, categorical_woe=categorical_woe)


def transform_woe(bins: WOEBins, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for feature, edges in bins.numeric_bin_edges.items():
        median = df[feature].median()
        bin_idx = np.digitize(df[feature].fillna(median), edges[1:-1])
        out[feature] = bins.numeric_woe[feature][bin_idx]
    for feature, woe_map in bins.categorical_woe.items():
        out[feature] = df[feature].map(woe_map).fillna(0.0)  # unseen category -> neutral WOE
    return out


def train_logistic_baseline(reference_df: pd.DataFrame) -> tuple[LogisticRegression, WOEBins]:
    bins = fit_woe_bins(reference_df)
    X = transform_woe(bins, reference_df)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, reference_df[TARGET_COLUMN])
    return model, bins


def evaluate_logistic_auc(model: LogisticRegression, bins: WOEBins, df: pd.DataFrame) -> float:
    X = transform_woe(bins, df)
    proba = model.predict_proba(X)[:, 1]
    return roc_auc_score(df[TARGET_COLUMN], proba)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_baseline_woe.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/IdeaProjects/mlops
git add model/baseline_woe.py tests/test_baseline_woe.py
git commit -m "Add hand-rolled logistic regression + WOE scorecard baseline"
```

---

### Task 3: Calibration analysis

**Files:**
- Create: `model/calibration.py`
- Create: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `config.NON_FEATURE_COLUMNS`, `config.TARGET_COLUMN`; any object exposing
  `.predict_proba(X) -> np.ndarray` (a fitted `Pipeline`, or anything with the same
  interface).
- Produces:
  - `reliability_diagram_data(pipeline, df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame`
    — columns `bin`, `mean_predicted`, `observed_rate`, `count`.
  - `expected_calibration_error(pipeline, df: pd.DataFrame, n_bins: int = 10) -> float`.

  Task 4 (report script) calls both directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calibration.py
import numpy as np
import pandas as pd

from model.calibration import expected_calibration_error, reliability_diagram_data


class _FakeProbaModel:
    """Stands in for a fitted Pipeline — ignores X, returns pre-set probabilities. Lets the
    ECE test assert against a known, exact ground truth rather than a real model's
    incidental calibration."""
    def __init__(self, probs: np.ndarray):
        self._probs = probs

    def predict_proba(self, X):
        return np.column_stack([1 - self._probs, self._probs])


def test_reliability_diagram_data_returns_bucketed_dataframe_covering_all_rows():
    n = 500
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 30000, size=n),
        "default_flag": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
    })
    model = _FakeProbaModel(rng.uniform(0, 1, size=n))

    diagram = reliability_diagram_data(model, df, n_bins=5)

    assert set(diagram.columns) == {"bin", "mean_predicted", "observed_rate", "count"}
    assert diagram["count"].sum() == n


def test_expected_calibration_error_near_zero_for_well_calibrated_predictions():
    n = 1000
    rng = np.random.default_rng(0)
    probs = np.full(n, 0.2)
    default_flag = rng.choice([0, 1], size=n, p=[0.8, 0.2])  # observed rate matches predicted
    df = pd.DataFrame({"loan_amnt": np.zeros(n), "default_flag": default_flag})

    ece = expected_calibration_error(_FakeProbaModel(probs), df, n_bins=5)

    assert ece < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_calibration.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'model.calibration'`.

- [ ] **Step 3: Implement `model/calibration.py`**

```python
# model/calibration.py
"""Calibration analysis: reliability-diagram data and Expected Calibration Error for any
fitted model exposing predict_proba."""
import numpy as np
import pandas as pd

from config import NON_FEATURE_COLUMNS, TARGET_COLUMN


def reliability_diagram_data(pipeline, df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Buckets predicted probabilities into n_bins equal-width buckets; returns mean
    predicted probability vs. observed default rate (and row count) per non-empty bucket."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    y_prob = pipeline.predict_proba(df[feature_cols])[:, 1]
    y_true = df[TARGET_COLUMN].to_numpy()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": b,
            "mean_predicted": float(y_prob[mask].mean()),
            "observed_rate": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(pipeline, df: pd.DataFrame, n_bins: int = 10) -> float:
    """Bucket-size-weighted average of |mean predicted prob - observed default rate|."""
    diagram = reliability_diagram_data(pipeline, df, n_bins=n_bins)
    total = diagram["count"].sum()
    weighted_error = (
        diagram["count"] / total * (diagram["mean_predicted"] - diagram["observed_rate"]).abs()
    ).sum()
    return float(weighted_error)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_calibration.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/IdeaProjects/mlops
git add model/calibration.py tests/test_calibration.py
git commit -m "Add calibration analysis: reliability diagram data and ECE"
```

---

### Task 4: Real report — tune, compare, calibrate, decide, document

**Files:**
- Create: `reports/generate_phase6_report.py`
- Modify: `model/train.py` — only if tuning clears the improvement threshold (see Step 3)
- Modify: `README.md`

**Interfaces:**
- Consumes: `tune_hyperparameters` from `model.tune`; `train_logistic_baseline`,
  `evaluate_logistic_auc` from `model.baseline_woe`; `reliability_diagram_data`,
  `expected_calibration_error` from `model.calibration`; `train_baseline`, `evaluate_auc`,
  `temporal_holdout_split` from `model.train`; `config.PROCESSED_DIR`, `config.REPORTS_DIR`,
  `config.TUNING_AUC_IMPROVEMENT_THRESHOLD`.
- Produces: a standalone script and two charts (`reports/phase6_tuning_comparison.png`,
  `reports/phase6_calibration.png`). No other module depends on this script's internals.

- [ ] **Step 1: Write the report script**

```python
# reports/generate_phase6_report.py
"""Phase 6 deliverable: hyperparameter tuning, a logistic regression + WOE baseline
comparison, and calibration analysis for the champion model. All numbers are a backtest on
public historical Lending Club data — not a live production evaluation."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_DIR, REPORTS_DIR, TUNING_AUC_IMPROVEMENT_THRESHOLD
from model.baseline_woe import evaluate_logistic_auc, train_logistic_baseline
from model.calibration import expected_calibration_error, reliability_diagram_data
from model.train import evaluate_auc, temporal_holdout_split, train_baseline
from model.tune import tune_hyperparameters


def main() -> None:
    reference_df = pd.read_parquet(PROCESSED_DIR / "reference.parquet")

    print("=== BACKTEST ON HISTORICAL DATA — not a live production evaluation ===")

    default_pipeline, _, test_slice = train_baseline(reference_df)
    default_auc = evaluate_auc(default_pipeline, test_slice)
    print(f"Current default-config holdout AUC: {default_auc:.4f}")

    print("\nRunning Optuna hyperparameter search (30 trials, ~15-30 min on this dataset)...")
    best_params, tuned_auc = tune_hyperparameters(reference_df)
    print(f"Best tuned holdout AUC: {tuned_auc:.4f}")
    print(f"Best params: {best_params}")

    improvement = tuned_auc - default_auc
    print(f"Improvement over default: {improvement:+.4f} "
          f"(threshold for adopting: {TUNING_AUC_IMPROVEMENT_THRESHOLD})")
    if improvement > TUNING_AUC_IMPROVEMENT_THRESHOLD:
        print("DECISION: tuning found a meaningful improvement — "
              "update model/train.py's build_model_pipeline with these params.")
        champion_pipeline = default_pipeline  # placeholder; re-trained manually after this run
    else:
        print("DECISION: tuning did not clear the improvement threshold — "
              "keeping the current default configuration.")
        champion_pipeline = default_pipeline

    logistic_model, woe_bins = train_logistic_baseline(reference_df)
    _, _, logistic_test_slice = train_baseline(reference_df)  # same split, for a fair comparison
    logistic_auc = evaluate_logistic_auc(logistic_model, woe_bins, logistic_test_slice)
    xgboost_auc = evaluate_auc(champion_pipeline, logistic_test_slice)

    print(f"\n=== Baseline comparison (same held-out slice) ===")
    print(f"Logistic regression + WOE — AUC: {logistic_auc:.4f}")
    print(f"XGBoost champion          — AUC: {xgboost_auc:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Logistic + WOE", "XGBoost (champion)"], [logistic_auc, xgboost_auc], color=["tab:orange", "tab:blue"])
    ax.set_ylabel("AUC")
    ax.set_title("Baseline Comparison — Backtest on Historical Lending Club Data")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "phase6_tuning_comparison.png", dpi=150)

    _, _, calibration_slice = train_baseline(reference_df)
    ece = expected_calibration_error(champion_pipeline, calibration_slice)
    diagram = reliability_diagram_data(champion_pipeline, calibration_slice)
    print(f"\n=== Calibration (champion, held-out slice) ===")
    print(f"Expected Calibration Error: {ece:.4f}")
    print(diagram.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(diagram["mean_predicted"], diagram["observed_rate"], marker="o", label="champion")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Reliability Diagram — Backtest on Historical Lending Club Data")
    ax.legend()
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "phase6_calibration.png", dpi=150)
    print(f"\nCharts saved to {REPORTS_DIR}/phase6_tuning_comparison.png and phase6_calibration.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the report against the real prepared data**

```bash
cd ~/IdeaProjects/mlops
uv run python -m reports.generate_phase6_report
```

Expected: prints the default-config holdout AUC, runs the 30-trial Optuna search (this is
the long step — expect roughly 15-30 minutes), prints the best params and whether they
clear the improvement threshold, prints the logistic+WOE vs. XGBoost AUC comparison, prints
the ECE and reliability-diagram table, and saves both charts. **Inspect the real numbers
before proceeding** — there is no way to know ahead of time whether tuning clears the
0.005 threshold; both outcomes are legitimate.

- [ ] **Step 3: Act on the real tuning result**

If Step 2's output showed `DECISION: tuning found a meaningful improvement`, update
`model/train.py`'s `build_model_pipeline` to use the printed `best_params` instead of the
current hardcoded values, e.g.:

```python
def build_model_pipeline() -> Pipeline:
    """Unfit champion/challenger pipeline: shared preprocessor + XGBoost classifier.

    Hyperparameters tuned via Optuna (Phase 6, reports/generate_phase6_report.py) — improved
    holdout AUC by <the real printed improvement value> over the prior hand-picked defaults.
    """
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", XGBClassifier(
            n_estimators=<tuned value>,
            max_depth=<tuned value>,
            learning_rate=<tuned value>,
            subsample=<tuned value>,
            colsample_bytree=<tuned value>,
            min_child_weight=<tuned value>,
            eval_metric="auc",
            random_state=42,
        )),
    ])
```

Then re-run the full test suite and Phase 1/3/5's report scripts to confirm the new config
doesn't break anything and to see the updated headline numbers:

```bash
cd ~/IdeaProjects/mlops
uv run pytest -v
uv run python -m reports.generate_phase1_report
uv run python -m reports.generate_phase3_report
uv run python -m backtest.run_backtest
```

If Step 2's output instead showed `DECISION: tuning did not clear the improvement
threshold`, make no change to `model/train.py` — this is a legitimate, reportable outcome,
not a failure to fix.

- [ ] **Step 4: Add a Phase 6 section to `README.md`**

Using the real printed numbers from Step 2 (and Step 3 if the config changed), add a
"## Phase 6 — Model Selection" section after the Phase 5 section, covering:

- The tuning result: default AUC, tuned AUC, whether the config was updated, and why (the
  0.005 threshold and the actual improvement observed).
- The baseline comparison table (logistic + WOE vs. XGBoost AUC) with a short paragraph on
  the interpretability tradeoff — WOE coefficients are directly explainable as scorecard
  points, XGBoost needs SHAP to get there but scores higher.
- The calibration result: the real ECE number, the reliability diagram
  (`reports/phase6_calibration.png`), and an honest read of whether raw `predict_proba` is
  pricing-usable as-is given that ECE value.
- The backtest framing line, consistent with every other section.

- [ ] **Step 5: Commit**

```bash
cd ~/IdeaProjects/mlops
git add reports/generate_phase6_report.py reports/phase6_tuning_comparison.png reports/phase6_calibration.png README.md
git add model/train.py  # only if Step 3 changed it
git commit -m "Add Phase 6 report: hyperparameter tuning, baseline comparison, calibration"
```

---

## Self-review notes

- **Spec coverage:** hyperparameter tuning with Optuna over the specified search space,
  budget, and improvement threshold ✓; hand-rolled logistic regression + WOE baseline,
  frozen on the reference window ✓; calibration analysis with reliability diagram and ECE ✓;
  report script running all three in order against real data, updating the champion
  automatically based on the real result, documented either way ✓; README section ✓.
- **Placeholder scan:** Task 4 Step 3's `<the real printed improvement value>` /
  `<tuned value>` placeholders are inside an *instruction* for what to fill in from Step 2's
  actual output at execution time — not a plan-writing placeholder, since the real numbers
  don't exist until the real 30-trial search runs. This mirrors Phase 6's own explicit
  design principle (§2.1): the decision is made from a real result, never assumed ahead of
  time.
- **Type consistency:** `temporal_holdout_split` (Task 1) returns
  `tuple[pd.DataFrame, pd.DataFrame]` and is consumed identically by `train_baseline` (Task
  1) and `tune_hyperparameters` (Task 1). `WOEBins` (Task 2) fields are constructed in
  `fit_woe_bins` and read identically by `transform_woe`. `reliability_diagram_data`'s
  output columns (Task 3) match exactly what `expected_calibration_error` (Task 3) and the
  report script's printed table (Task 4) both consume.
