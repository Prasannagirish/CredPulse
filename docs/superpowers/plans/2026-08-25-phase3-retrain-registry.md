# CreditPulse Phase 3 — Retraining + MLflow Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user is
> executing this plan to learn the system, one step at a time — narrate what each step is
> doing and why before running it, don't just execute silently.

**Goal:** Retrain a challenger model on a trailing window of recent loans, compare it against
the Phase 1 champion on a genuinely held-out slice, and wire both into an MLflow model
registry whose `promote_challenger`/`rollback` operations verifiably refuse to run without
explicit `confirm=True` — the project's core safety property (spec §2: no silent
auto-promotion, ever).

**Architecture:** `lifecycle/retrain.py` selects a trailing window and fits a fresh model with
the same architecture as the Phase 1 champion (extracted into a shared factory in
`model/train.py`). `lifecycle/evaluate.py` scores champion and challenger on the same held-out
slice (AUC, F1, Brier score). `lifecycle/registry.py` wraps MLflow's alias-based model
registry (`champion`/`challenger` aliases, not the deprecated stage API) with identity-checked
confirmation gating. A Phase 3 report script runs the whole lifecycle once against real data.

**Tech Stack:** Python 3.12, pandas, scikit-learn, xgboost (reusing `model/train.py`'s
pipeline), mlflow (`MlflowClient`, model registry, SQLite backend already configured), pytest.

## Global Constraints

- `promote_challenger(confirm: bool)` and `rollback(confirm: bool)` must raise
  `ConfirmationRequired` unless `confirm is True` **exactly** — an identity check, not a
  truthiness check, so `1`, `"true"`, `"yes"` are all refused (design doc §6). This is a
  deliberate safety property, not a missing feature (spec §2).
- The registry uses MLflow **aliases** (`champion`, `challenger`), not the deprecated
  `stage` API.
- `promote_challenger` must record the displaced champion version before reassigning the
  alias, so `rollback` has a well-defined target.
- Champion and challenger are always evaluated on a window **neither** was trained on (spec
  Phase 3: "never train and evaluate on the same window").
- `tests/test_lifecycle_gating.py` (spec §6's required filename) must assert both that a
  call without `confirm=True` raises, **and** that registry state is unchanged afterward —
  a refusal that still mutates state would defeat the whole point.
- OpenMP note carried over from Phase 2: any script combining `xgboost` and `torch` (none of
  Phase 3's own modules use `torch`, but the Phase 3 report script doesn't need it either) —
  not applicable here, no `OMP_NUM_THREADS` guard needed in this phase's files.
- Every printed report output is a backtest on historical, resolved-outcome data — label it
  as such, consistent with Phases 1-2.

---

### Task 1: Extract shared model factory; add `lifecycle.retrain`

**Files:**
- Modify: `model/train.py` — extract `build_model_pipeline()`
- Create: `lifecycle/__init__.py`
- Create: `lifecycle/retrain.py`
- Create: `tests/test_lifecycle.py`
- Modify: `config.py` — add `RETRAIN_WINDOW_MONTHS`

**Interfaces:**
- Produces:
  - `model.train.build_model_pipeline() -> Pipeline` — unfit `Pipeline([("preprocess",
    build_preprocessor()), ("model", XGBClassifier(...))])` with the exact hyperparameters
    `train_baseline` already used. Both `train_baseline` (Phase 1) and
    `lifecycle.retrain.retrain_challenger` (this task) call this, so champion and challenger
    always share architecture — only the training data differs.
  - `lifecycle.retrain.select_recent_window(df: pd.DataFrame, as_of: pd.Timestamp, months:
    int = RETRAIN_WINDOW_MONTHS, date_col: str = "issue_d") -> pd.DataFrame` — rows with
    `date_col` in `[as_of - months, as_of)`.
  - `lifecycle.retrain.retrain_challenger(training_df: pd.DataFrame) -> Pipeline` — fits a
    fresh `build_model_pipeline()` on all of `training_df` (no internal holdout — the caller
    is responsible for ensuring the evaluation window, handled in Task 2, doesn't overlap).
    Task 2 (`lifecycle/evaluate.py`) and Task 4 (report script) both call this directly.

- [ ] **Step 1: Add `RETRAIN_WINDOW_MONTHS` to `config.py`**

Append to `config.py`:

```python
# Default trailing window (in months) a challenger retrains on, counting back from the
# simulated "as of" retrain date.
RETRAIN_WINDOW_MONTHS = 12
```

- [ ] **Step 2: Extract `build_model_pipeline()` in `model/train.py`**

Find this block in `model/train.py` (inside `train_baseline`):

```python
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="auc",
            random_state=42,
        )),
    ])
```

Replace the whole `train_baseline` function with:

```python
def build_model_pipeline() -> Pipeline:
    """Unfit champion/challenger pipeline: shared preprocessor + XGBoost classifier.

    Used by both the Phase 1 champion (train_baseline, below) and Phase 3 challenger
    retraining (lifecycle.retrain.retrain_challenger) so both are trained with identical
    architecture and hyperparameters — only the training data differs.
    """
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="auc",
            random_state=42,
        )),
    ])


def train_baseline(reference_df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """Fit the baseline champion on a temporal holdout of the reference window.

    The test slice is the chronologically last 10% of reference_df, not a random sample —
    a random split would let the model implicitly see "future" reference-window loans
    during training, which is a milder version of the same leak the reference/eval split
    exists to prevent.
    """
    sorted_df = reference_df.sort_values("issue_d").reset_index(drop=True)
    split_idx = int(len(sorted_df) * 0.9)
    train_slice = sorted_df.iloc[:split_idx]
    test_slice = sorted_df.iloc[split_idx:]

    pipeline = build_model_pipeline()
    feature_cols = [c for c in reference_df.columns if c not in (TARGET_COLUMN, "issue_d")]
    pipeline.fit(train_slice[feature_cols], train_slice[TARGET_COLUMN])

    return pipeline, train_slice, test_slice
```

- [ ] **Step 3: Run the Phase 1/2 test suite to confirm the refactor didn't break anything**

```bash
cd ~/IdeaProjects/mlops
uv run pytest -v
```

Expected: all 21 previously-passing tests still pass (this is a pure extraction — identical
behavior, no new logic).

- [ ] **Step 4: Write the failing tests for `lifecycle.retrain`**

```python
# tests/test_lifecycle.py
import numpy as np
import pandas as pd

from lifecycle.retrain import retrain_challenger, select_recent_window


def _synthetic_df(n: int = 200, start: str = "2019-01-01") -> pd.DataFrame:
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


def test_select_recent_window_keeps_only_trailing_months():
    df = _synthetic_df(n=200, start="2019-01-01")  # spans ~19.7 months at 3-day spacing
    as_of = pd.Timestamp("2020-01-01")
    window = select_recent_window(df, as_of, months=6)
    assert window["issue_d"].min() >= as_of - pd.DateOffset(months=6)
    assert window["issue_d"].max() < as_of
    assert len(window) < len(df)


def test_retrain_challenger_returns_fitted_pipeline_that_predicts_probabilities():
    df = _synthetic_df()
    pipeline = retrain_challenger(df)
    feature_cols = [c for c in df.columns if c not in ("default_flag", "issue_d")]
    predictions = pipeline.predict_proba(df[feature_cols])
    assert predictions.shape == (len(df), 2)
```

- [ ] **Step 5: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_lifecycle.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lifecycle'`.

- [ ] **Step 6: Create the `lifecycle` package and implement `lifecycle/retrain.py`**

```bash
cd ~/IdeaProjects/mlops
mkdir -p lifecycle
touch lifecycle/__init__.py
```

```python
# lifecycle/retrain.py
"""Retrain a credit-risk challenger on the most recent available window of resolved loans."""
import pandas as pd
from sklearn.pipeline import Pipeline

from config import RETRAIN_WINDOW_MONTHS, TARGET_COLUMN
from model.train import build_model_pipeline


def select_recent_window(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    months: int = RETRAIN_WINDOW_MONTHS,
    date_col: str = "issue_d",
) -> pd.DataFrame:
    """Rows with date_col in [as_of - months, as_of) — the trailing window "available" at
    as_of in a walk-forward simulation."""
    window_start = as_of - pd.DateOffset(months=months)
    return df[(df[date_col] >= window_start) & (df[date_col] < as_of)].copy()


def retrain_challenger(training_df: pd.DataFrame) -> Pipeline:
    """Train a fresh challenger on training_df. Same architecture as the Phase 1 champion
    (model.train.build_model_pipeline) but trained on whatever window the caller selects —
    no internal holdout here; evaluation happens separately via lifecycle.evaluate, on a
    window the caller guarantees wasn't used for this training (spec Phase 3)."""
    pipeline = build_model_pipeline()
    feature_cols = [c for c in training_df.columns if c not in (TARGET_COLUMN, "issue_d")]
    pipeline.fit(training_df[feature_cols], training_df[TARGET_COLUMN])
    return pipeline
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_lifecycle.py -v
```

Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
cd ~/IdeaProjects/mlops
git add model/train.py lifecycle/__init__.py lifecycle/retrain.py tests/test_lifecycle.py config.py
git commit -m "Extract shared model factory; add challenger retraining"
```

---

### Task 2: Champion vs. challenger comparison

**Files:**
- Create: `lifecycle/evaluate.py`
- Test: `tests/test_lifecycle.py` (append)

**Interfaces:**
- Consumes: `TARGET_COLUMN` from `config`; fitted `Pipeline` objects from `model.train` /
  `lifecycle.retrain` (both produce the same `Pipeline` shape: `predict_proba` over all
  columns except `TARGET_COLUMN` and `"issue_d"`).
- Produces:
  - `ChampionChallengerComparison` (dataclass) — fields `champion_auc: float`,
    `challenger_auc: float`, `champion_f1: float`, `challenger_f1: float`, `champion_brier:
    float`, `challenger_brier: float`, `challenger_wins: bool` (`True` iff
    `challenger_auc > champion_auc`).
  - `compare_champion_challenger(champion: Pipeline, challenger: Pipeline, holdout_df:
    pd.DataFrame) -> ChampionChallengerComparison`. Task 4 (report script) calls this
    directly, once per retrain simulation.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_lifecycle.py
from lifecycle.evaluate import compare_champion_challenger


def test_compare_champion_challenger_returns_metrics_for_both():
    train_df = _synthetic_df(n=200, start="2018-01-01")
    later_train_df = _synthetic_df(n=200, start="2019-01-01")
    holdout_df = _synthetic_df(n=100, start="2019-09-01")

    champion = retrain_challenger(train_df)
    challenger = retrain_challenger(later_train_df)

    comparison = compare_champion_challenger(champion, challenger, holdout_df)

    assert 0.0 <= comparison.champion_auc <= 1.0
    assert 0.0 <= comparison.challenger_auc <= 1.0
    assert 0.0 <= comparison.champion_f1 <= 1.0
    assert 0.0 <= comparison.challenger_f1 <= 1.0
    assert 0.0 <= comparison.champion_brier <= 1.0
    assert 0.0 <= comparison.challenger_brier <= 1.0
    assert comparison.challenger_wins == (comparison.challenger_auc > comparison.champion_auc)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_lifecycle.py -v -k compare_champion_challenger
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lifecycle.evaluate'`.

- [ ] **Step 3: Implement `lifecycle/evaluate.py`**

```python
# lifecycle/evaluate.py
"""Score a champion and a challenger on the same held-out window, never one either was
trained on (spec Phase 3)."""
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from config import TARGET_COLUMN


@dataclass
class ChampionChallengerComparison:
    champion_auc: float
    challenger_auc: float
    champion_f1: float
    challenger_f1: float
    champion_brier: float
    challenger_brier: float
    challenger_wins: bool


def _score(pipeline: Pipeline, df: pd.DataFrame) -> tuple[float, float, float]:
    feature_cols = [c for c in df.columns if c not in (TARGET_COLUMN, "issue_d")]
    y_true = df[TARGET_COLUMN]
    y_prob = pipeline.predict_proba(df[feature_cols])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return (
        roc_auc_score(y_true, y_prob),
        f1_score(y_true, y_pred),
        brier_score_loss(y_true, y_prob),
    )


def compare_champion_challenger(
    champion: Pipeline, challenger: Pipeline, holdout_df: pd.DataFrame
) -> ChampionChallengerComparison:
    champion_auc, champion_f1, champion_brier = _score(champion, holdout_df)
    challenger_auc, challenger_f1, challenger_brier = _score(challenger, holdout_df)
    return ChampionChallengerComparison(
        champion_auc=champion_auc,
        challenger_auc=challenger_auc,
        champion_f1=champion_f1,
        challenger_f1=challenger_f1,
        champion_brier=champion_brier,
        challenger_brier=challenger_brier,
        challenger_wins=challenger_auc > champion_auc,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_lifecycle.py -v
```

Expected: 3 passed (2 from Task 1, 1 from this task).

- [ ] **Step 5: Commit**

```bash
cd ~/IdeaProjects/mlops
git add lifecycle/evaluate.py tests/test_lifecycle.py
git commit -m "Add champion vs challenger comparison (AUC, F1, Brier)"
```

---

### Task 3: MLflow registry with confirm-gated promote/rollback

**Files:**
- Create: `lifecycle/registry.py`
- Create: `tests/test_lifecycle_gating.py`
- Modify: `config.py` — add `MODEL_NAME`

**Interfaces:**
- Consumes: `config.MLFLOW_TRACKING_URI`, `config.MODEL_NAME` (looked up as `config.ATTR`
  inside each function body, not via `from config import ATTR` at module level — this is
  required so tests can `monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", ...)` per test
  for isolation; a module-level import would freeze the value at import time and
  monkeypatching would have no effect).
- Produces:
  - `ConfirmationRequired(Exception)` — raised by `promote_challenger`/`rollback` when
    `confirm is not True`.
  - `register_model(model_uri: str, name: str | None = None) -> str` — registers an MLflow
    model (logged via `mlflow.sklearn.log_model`, referenced as `f"runs:/{run_id}/model"`) as
    a new version under `name` (defaults to `config.MODEL_NAME`). Returns the version number
    as a string. Creates the registered model if it doesn't exist yet.
  - `set_challenger(version: str, name: str | None = None) -> None` — sets the `challenger`
    alias to `version`.
  - `get_champion_version(name: str | None = None) -> str | None` — `None` if no champion
    alias is set (including if the registered model doesn't exist yet).
  - `get_challenger_version(name: str | None = None) -> str | None` — same, for `challenger`.
  - `promote_challenger(confirm: bool, name: str | None = None) -> str` — refuses unless
    `confirm is True`. Records the current champion version (if any) as a
    `previous_champion_version` tag, reassigns the `champion` alias to the current challenger
    version, deletes the `challenger` alias, and returns the newly-promoted version.
  - `rollback(confirm: bool, name: str | None = None) -> str` — refuses unless `confirm is
    True`. Reassigns `champion` back to the version recorded in `previous_champion_version`
    and returns it. Raises `ValueError` if no previous champion was recorded.

  Task 4 (report script) and `tests/test_lifecycle_gating.py` both call all of the above
  directly.

- [ ] **Step 1: Add `MODEL_NAME` to `config.py`**

Append to `config.py`:

```python
MODEL_NAME = "creditpulse-credit-risk"
```

- [ ] **Step 2: Write the failing gating tests**

```python
# tests/test_lifecycle_gating.py
"""Asserts promote_challenger/rollback refuse to act without confirm=True, and that
registry state is unchanged after a refusal — spec §2's core safety property."""
import mlflow
import mlflow.sklearn
import pytest
from sklearn.linear_model import LogisticRegression

import config
from lifecycle.registry import (
    ConfirmationRequired,
    get_champion_version,
    get_challenger_version,
    promote_challenger,
    register_model,
    rollback,
    set_challenger,
)


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'test_mlflow.db'}")
    monkeypatch.setattr(config, "MODEL_NAME", "test-model")


def _log_and_register_dummy_model() -> str:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    with mlflow.start_run() as run:
        model = LogisticRegression().fit([[0], [1]], [0, 1])
        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
    return register_model(model_uri)


def test_promote_challenger_refuses_without_confirm_true():
    version = _log_and_register_dummy_model()
    set_challenger(version)

    with pytest.raises(ConfirmationRequired):
        promote_challenger(confirm=False)

    assert get_champion_version() is None
    assert get_challenger_version() == version


def test_promote_challenger_refuses_truthy_non_bool_values():
    version = _log_and_register_dummy_model()
    set_challenger(version)

    for bad_value in [1, "true", "yes"]:
        with pytest.raises(ConfirmationRequired):
            promote_challenger(confirm=bad_value)
        assert get_champion_version() is None


def test_promote_challenger_succeeds_with_confirm_true():
    version = _log_and_register_dummy_model()
    set_challenger(version)

    promoted_version = promote_challenger(confirm=True)

    assert promoted_version == version
    assert get_champion_version() == version
    assert get_challenger_version() is None


def test_rollback_refuses_without_confirm_true():
    v1 = _log_and_register_dummy_model()
    set_challenger(v1)
    promote_challenger(confirm=True)

    v2 = _log_and_register_dummy_model()
    set_challenger(v2)
    promote_challenger(confirm=True)

    with pytest.raises(ConfirmationRequired):
        rollback(confirm=False)

    assert get_champion_version() == v2


def test_rollback_reverts_to_previous_champion_with_confirm_true():
    v1 = _log_and_register_dummy_model()
    set_challenger(v1)
    promote_challenger(confirm=True)

    v2 = _log_and_register_dummy_model()
    set_challenger(v2)
    promote_challenger(confirm=True)

    reverted_version = rollback(confirm=True)

    assert reverted_version == v1
    assert get_champion_version() == v1
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_lifecycle_gating.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lifecycle.registry'`.

- [ ] **Step 4: Implement `lifecycle/registry.py`**

```python
# lifecycle/registry.py
"""Thin wrapper over MLflow's model registry for champion/challenger get/register/promote/
rollback. Uses aliases (champion/challenger), not the deprecated stage API."""
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

import config

_CHAMPION_ALIAS = "champion"
_CHALLENGER_ALIAS = "challenger"
_PREVIOUS_CHAMPION_TAG = "previous_champion_version"


class ConfirmationRequired(Exception):
    """Raised when a lifecycle-mutating call is attempted without confirm=True."""


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return MlflowClient()


def _resolve_name(name: str | None) -> str:
    return name if name is not None else config.MODEL_NAME


def register_model(model_uri: str, name: str | None = None) -> str:
    """Register a logged MLflow model (by its run URI, e.g. f"runs:/{run_id}/model") as a
    new version under `name`. Creates the registered model if it doesn't exist yet."""
    name = _resolve_name(name)
    client = _client()
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)
    result = mlflow.register_model(model_uri, name)
    return result.version


def set_challenger(version: str, name: str | None = None) -> None:
    name = _resolve_name(name)
    _client().set_registered_model_alias(name, _CHALLENGER_ALIAS, version)


def get_champion_version(name: str | None = None) -> str | None:
    name = _resolve_name(name)
    try:
        return _client().get_model_version_by_alias(name, _CHAMPION_ALIAS).version
    except MlflowException:
        return None


def get_challenger_version(name: str | None = None) -> str | None:
    name = _resolve_name(name)
    try:
        return _client().get_model_version_by_alias(name, _CHALLENGER_ALIAS).version
    except MlflowException:
        return None


def promote_challenger(confirm: bool, name: str | None = None) -> str:
    """Promote the current challenger to champion. Refuses unless confirm is exactly True —
    a deliberate safety property (spec §2): no silent auto-promotion, ever."""
    if confirm is not True:
        raise ConfirmationRequired("promote_challenger requires confirm=True")

    name = _resolve_name(name)
    client = _client()
    challenger_version = get_challenger_version(name)
    if challenger_version is None:
        raise ValueError(f"No challenger registered for model '{name}'")

    previous_champion_version = get_champion_version(name)
    if previous_champion_version is not None:
        client.set_registered_model_tag(name, _PREVIOUS_CHAMPION_TAG, previous_champion_version)

    client.set_registered_model_alias(name, _CHAMPION_ALIAS, challenger_version)
    client.delete_registered_model_alias(name, _CHALLENGER_ALIAS)
    return challenger_version


def rollback(confirm: bool, name: str | None = None) -> str:
    """Revert to the previous champion version. Refuses unless confirm is exactly True."""
    if confirm is not True:
        raise ConfirmationRequired("rollback requires confirm=True")

    name = _resolve_name(name)
    client = _client()
    model = client.get_registered_model(name)
    previous_version = model.tags.get(_PREVIOUS_CHAMPION_TAG)
    if previous_version is None:
        raise ValueError(f"No previous champion recorded for model '{name}' to roll back to")

    client.set_registered_model_alias(name, _CHAMPION_ALIAS, previous_version)
    return previous_version
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_lifecycle_gating.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Run the full test suite**

```bash
cd ~/IdeaProjects/mlops
uv run pytest -v
```

Expected: all tests pass (24 from before + these new ones).

- [ ] **Step 7: Commit**

```bash
cd ~/IdeaProjects/mlops
git add lifecycle/registry.py tests/test_lifecycle_gating.py config.py
git commit -m "Add MLflow registry with confirm-gated promote/rollback

promote_challenger/rollback raise ConfirmationRequired unless confirm is
exactly True (identity check, not truthiness) - covered by
tests/test_lifecycle_gating.py, which asserts both the refusal and that
registry state is unchanged afterward."
```

---

### Task 4: Real end-to-end lifecycle run

**Files:**
- Create: `reports/generate_phase3_report.py`

**Interfaces:**
- Consumes: `train_baseline` from `model.train`; `select_recent_window`,
  `retrain_challenger` from `lifecycle.retrain`; `compare_champion_challenger` from
  `lifecycle.evaluate`; `ConfirmationRequired`, `get_champion_version`, `promote_challenger`,
  `register_model`, `rollback`, `set_challenger` from `lifecycle.registry`;
  `config.MLFLOW_TRACKING_URI`, `config.PROCESSED_DIR`.
- Produces: a standalone script demonstrating the full lifecycle against real data. No other
  module depends on its internals.

- [ ] **Step 1: Write the report script**

```python
# reports/generate_phase3_report.py
"""Phase 3 deliverable: retrain a challenger on a trailing window, compare it against the
Phase 1 champion on a genuinely held-out slice, register both in MLflow, and demonstrate the
confirm-gated promote/rollback lifecycle end-to-end. All numbers are a backtest on public
historical Lending Club data — not a live production decision."""
import mlflow
import mlflow.sklearn
import pandas as pd

from config import MLFLOW_TRACKING_URI, PROCESSED_DIR
from lifecycle.evaluate import compare_champion_challenger
from lifecycle.registry import (
    ConfirmationRequired,
    get_champion_version,
    promote_challenger,
    register_model,
    rollback,
    set_challenger,
)
from lifecycle.retrain import retrain_challenger, select_recent_window
from model.train import train_baseline

RETRAIN_AS_OF = pd.Timestamp("2020-01-01")
HOLDOUT_MONTHS = 2


def main() -> None:
    reference_df = pd.read_parquet(PROCESSED_DIR / "reference.parquet")
    eval_df = pd.read_parquet(PROCESSED_DIR / "eval.parquet")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("creditpulse-phase3-lifecycle")

    print("=== BACKTEST ON HISTORICAL DATA — not a live production decision ===")

    champion, _, _ = train_baseline(reference_df)

    training_window = select_recent_window(eval_df, RETRAIN_AS_OF, months=12)
    holdout_start = RETRAIN_AS_OF
    holdout_end = RETRAIN_AS_OF + pd.DateOffset(months=HOLDOUT_MONTHS)
    holdout_df = eval_df[(eval_df["issue_d"] >= holdout_start) & (eval_df["issue_d"] < holdout_end)]

    print(f"Retrain-as-of: {RETRAIN_AS_OF.date()}, trailing window: {len(training_window)} loans "
          f"({training_window['issue_d'].min().date()} to {training_window['issue_d'].max().date()})")
    print(f"Held-out comparison slice: {len(holdout_df)} loans "
          f"({holdout_start.date()} to {holdout_end.date()}, not used in either model's training)")

    challenger = retrain_challenger(training_window)
    comparison = compare_champion_challenger(champion, challenger, holdout_df)

    print(f"\nChampion   — AUC: {comparison.champion_auc:.4f}  F1: {comparison.champion_f1:.4f}  "
          f"Brier: {comparison.champion_brier:.4f}")
    print(f"Challenger — AUC: {comparison.challenger_auc:.4f}  F1: {comparison.challenger_f1:.4f}  "
          f"Brier: {comparison.challenger_brier:.4f}")
    print(f"Challenger wins on AUC: {comparison.challenger_wins}")

    with mlflow.start_run(run_name="champion-reference") as champion_run:
        mlflow.sklearn.log_model(champion, "model")
        mlflow.log_metric("auc", comparison.champion_auc)
        champion_uri = f"runs:/{champion_run.info.run_id}/model"
    with mlflow.start_run(run_name=f"challenger-as-of-{RETRAIN_AS_OF.date()}") as challenger_run:
        mlflow.sklearn.log_model(challenger, "model")
        mlflow.log_metric("auc", comparison.challenger_auc)
        challenger_uri = f"runs:/{challenger_run.info.run_id}/model"

    champion_version = register_model(champion_uri)
    if get_champion_version() is None:
        # Bootstrap: promote_challenger requires an existing challenger alias, so run the
        # very first champion through the same gated path rather than special-casing it —
        # there's simply no "previous champion" to protect on the first-ever registration.
        set_challenger(champion_version)
        promote_challenger(confirm=True)
    print(f"\nRegistered champion as version {champion_version} (bootstrapped as registry champion)")

    challenger_version = register_model(challenger_uri)
    set_challenger(challenger_version)
    print(f"Registered challenger as version {challenger_version}")

    print("\n--- Demonstrating confirm-gating (spec §2: no silent auto-promotion) ---")
    try:
        promote_challenger(confirm=False)
        raise AssertionError("promote_challenger should have refused without confirm=True")
    except ConfirmationRequired as e:
        print(f"promote_challenger(confirm=False) correctly refused: {e}")

    if comparison.challenger_wins:
        new_champion_version = promote_challenger(confirm=True)
        print(f"promote_challenger(confirm=True) succeeded — new champion is version {new_champion_version}")

        try:
            rollback(confirm=False)
            raise AssertionError("rollback should have refused without confirm=True")
        except ConfirmationRequired as e:
            print(f"rollback(confirm=False) correctly refused: {e}")

        reverted_version = rollback(confirm=True)
        print(f"rollback(confirm=True) succeeded — reverted to version {reverted_version}")
    else:
        print(
            "Challenger did not outperform champion on this window — in a real system, "
            "trigger_retrain (Phase 4) would report this and a human/agent would decide "
            "not to promote. No promotion attempted here for the same reason."
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the report against the real prepared data**

```bash
cd ~/IdeaProjects/mlops
uv run python -m reports.generate_phase3_report
```

Expected: prints the training window / holdout window sizes and dates, champion vs.
challenger metrics (AUC/F1/Brier), confirmation that `promote_challenger(confirm=False)` and
(if the challenger wins) `rollback(confirm=False)` both raised `ConfirmationRequired`, and the
final registry state. **Inspect the result**: the retrain-as-of date (2020-01-01) is chosen
to land after Phase 2's drift breach (macro features breached 2019-06) — if the challenger
doesn't outperform the champion here, that's still a legitimate, reportable outcome (not
every retrain necessarily wins), not a bug to chase; the acceptance criterion for this phase
is the gating mechanism working correctly, which Task 3's tests already verify independently
of what this particular retrain's numbers turn out to be.

- [ ] **Step 3: Commit**

```bash
cd ~/IdeaProjects/mlops
git add reports/generate_phase3_report.py
git commit -m "Add end-to-end Phase 3 lifecycle report: retrain, compare, register, gate"
```

---

## Self-review notes

- **Spec coverage:** `lifecycle/retrain.py` (retrain challenger on most recent N months) ✓;
  `lifecycle/evaluate.py` (champion vs challenger on held-out recent slice, never
  train-and-evaluate on the same window) ✓; `lifecycle/registry.py` (MLflow registry wrap for
  get/register/promote/rollback) ✓; acceptance criterion — promotion refuses without
  confirmation, covered by a passing test before moving on — `tests/test_lifecycle_gating.py`
  (Task 3) is exactly the spec's required filename and asserts both the refusal and
  unchanged state ✓.
- **Placeholder scan:** none remaining.
- **Type consistency:** `Pipeline` objects from `model.train.build_model_pipeline` (via both
  `train_baseline` and `lifecycle.retrain.retrain_challenger`) share the same
  `predict_proba`-over-feature-columns interface consumed by `lifecycle.evaluate._score` —
  verified by Task 2's test using both functions together. `ChampionChallengerComparison`
  field names (`champion_auc`, `challenger_auc`, `champion_f1`, `challenger_f1`,
  `champion_brier`, `challenger_brier`, `challenger_wins`) are defined once in Task 2 and
  referenced with matching names in Task 4's report script. `ConfirmationRequired`,
  `promote_challenger`, `rollback`, `register_model`, `set_challenger`,
  `get_champion_version`, `get_challenger_version` are defined once in Task 3 with signatures
  `(confirm: bool, name: str | None = None)` / `(version_or_uri, name: str | None = None)`
  and called identically in Task 3's own tests and Task 4's report script.
