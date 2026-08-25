# CreditPulse Phase 4 — MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user is
> executing this plan to learn the system, one step at a time — narrate what each step is
> doing and why before running it, don't just execute silently.

**Goal:** Expose the whole model lifecycle — health check, drift report, retrain, promote,
rollback, explain-a-prediction — as six MCP tools a real MCP client can drive conversationally,
and prove it works end-to-end with a saved transcript.

**Architecture:** `mcp_server/state.py` bootstraps and caches the expensive, shared runtime
state (the current champion, drift reference, autoencoder, SHAP explainer) once per process.
`mcp_server/schemas.py` defines the Pydantic output models spec §7 calls for. `mcp_server/
server.py` wires six MCP tools, each a thin wrapper around a directly-testable `_impl`
function that reuses Phases 1-3's modules. A small client script drives all six tools against
a running server instance and produces the transcript spec §8 Phase 4 requires.

**Tech Stack:** Python 3.12, the `mcp` SDK (installed version 2.1.0 — note this version's
`FastMCP` was renamed `MCPServer`, at `mcp.server.mcpserver.MCPServer`; the spec's own
description of "FastMCP" refers to the decorator-based ergonomics, which `MCPServer` provides
identically, not a specific class name), pydantic, shap, mlflow, pandas, pytest.

## Global Constraints

- Server class: `from mcp.server.mcpserver import MCPServer; mcp = MCPServer("creditpulse")`.
  Tools are registered with `@mcp.tool(name="...")` — pass `name=` explicitly whenever the
  Python function name would otherwise collide with an imported name (e.g.
  `lifecycle.registry.promote_challenger`). Start with `mcp.run(transport="stdio")`.
- Each tool's docstring is its MCP-visible description (spec §7) — write real, useful
  descriptions, not placeholders.
- `promote_challenger`/`rollback` tools delegate directly to the already-gated
  `lifecycle.registry` functions — do not reimplement the confirm check; inheriting it is
  what guarantees the MCP-exposed tools can't bypass Phase 3's tested safety property.
- All six tools operate on **backtest, historical data** — no tool queries anything outside
  `var/processed/*.parquet` and the local MLflow registry. Every tool's docstring should make
  this framing clear, consistent with every other phase.
- `mlflow.sklearn.load_model(model_uri)` needs no `skops_trusted_types` argument (verified —
  only `log_model`/`save_model` do). `shap.TreeExplainer(model).shap_values(X)` returns a
  plain `(n_samples, n_features)` ndarray for our binary XGBClassifier, not a list.
  `ColumnTransformer.get_feature_names_out()` yields names like `"numeric__loan_amnt"` and
  `"categorical__grade_A"` given `build_preprocessor()`'s transformer names `"numeric"`/
  `"categorical"`.
- Phase 4's acceptance criterion (spec §8): a full walkthrough transcript (health check →
  drift report → retrain → promote) driven from a real MCP client, saved into the README.

---

### Task 1: Retain loan `id` through the data pipeline

**Files:**
- Modify: `config.py` — add `NON_FEATURE_COLUMNS`
- Modify: `data/prepare.py` — retain raw `id` column
- Modify: `model/train.py`, `lifecycle/evaluate.py`, `lifecycle/retrain.py` — use
  `NON_FEATURE_COLUMNS` instead of the inline `(TARGET_COLUMN, "issue_d")` tuple
- Test: `tests/test_prepare.py` (append)

**Interfaces:**
- Produces: `config.NON_FEATURE_COLUMNS: tuple[str, ...]` = `(TARGET_COLUMN, "issue_d",
  "id")` — the single place every "which columns are model features" computation excludes
  non-feature columns from. `run_prepare`'s output parquet gains an `id` column (the raw
  Lending Club loan id) alongside the existing `issue_d`/features/target — Task 5's
  `explain_prediction` tool looks a loan up by this column.

- [ ] **Step 1: Add `NON_FEATURE_COLUMNS` to `config.py`**

Append to `config.py`:

```python
# Columns that are never model features, in every "which columns go into X" computation
# across model/train.py, lifecycle/evaluate.py, lifecycle/retrain.py, and the MCP server.
NON_FEATURE_COLUMNS = (TARGET_COLUMN, "issue_d", "id")
```

- [ ] **Step 2: Retain `id` in `data/prepare.py`**

Find this line in `run_prepare` (in `data/prepare.py`):

```python
    selected["issue_d"] = raw.loc[selected.index, "issue_d"]
```

Add directly below it:

```python
    selected["id"] = raw.loc[selected.index, "id"]
```

- [ ] **Step 3: Write the failing test for the `id` column**

Append to `tests/test_prepare.py`:

```python
def test_run_prepare_retains_loan_id(tmp_path: Path):
    csv_path = tmp_path / "fixture.csv"
    csv_path.write_text(
        "id,loan_amnt,term,int_rate,grade,sub_grade,emp_length,home_ownership,annual_inc,"
        "verification_status,dti,revol_util,fico_range_low,fico_range_high,purpose,"
        "issue_d,loan_status\n"
        "1001,1000, 36 months,10.65%,B,B2,3 years,RENT,50000,Verified,15.0,30%,700,704,"
        "debt_consolidation,Dec-2018,Fully Paid\n"
        "1002,2000, 60 months,15.0%,D,D1,10+ years,MORTGAGE,70000,Not Verified,20.0,50%,650,654,"
        "credit_card,Jun-2019,Charged Off\n"
    )
    output_dir = tmp_path / "processed"

    reference_df, eval_df = run_prepare(raw_csv_path=csv_path, output_dir=output_dir)

    assert list(reference_df["id"]) == [1001]
    assert list(eval_df["id"]) == [1002]
```

- [ ] **Step 4: Run test to verify it fails**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_prepare.py::test_run_prepare_retains_loan_id -v
```

Expected: FAIL — `KeyError: 'id'` (the column doesn't exist yet in the output).

- [ ] **Step 5: Run test to verify it passes** (Step 2's edit already implements this)

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_prepare.py -v
```

Expected: all `test_prepare.py` tests pass, including the new one.

- [ ] **Step 6: Update the three `feature_cols` exclusion sites to use `NON_FEATURE_COLUMNS`**

In `model/train.py`, both occurrences of:

```python
    feature_cols = [c for c in reference_df.columns if c not in (TARGET_COLUMN, "issue_d")]
```

and

```python
    feature_cols = [c for c in df.columns if c not in (TARGET_COLUMN, "issue_d")]
```

become (respectively, keeping each function's own variable name — `reference_df`/`df`):

```python
    feature_cols = [c for c in reference_df.columns if c not in NON_FEATURE_COLUMNS]
```

```python
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
```

Update the import line at the top of `model/train.py`:

```python
from config import NON_FEATURE_COLUMNS, TARGET_COLUMN
```

In `lifecycle/evaluate.py`, inside `_score`:

```python
    feature_cols = [c for c in df.columns if c not in (TARGET_COLUMN, "issue_d")]
```

becomes:

```python
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
```

Update its import line:

```python
from config import NON_FEATURE_COLUMNS, TARGET_COLUMN
```

In `lifecycle/retrain.py`, inside `retrain_challenger`:

```python
    feature_cols = [c for c in training_df.columns if c not in (TARGET_COLUMN, "issue_d")]
```

becomes:

```python
    feature_cols = [c for c in training_df.columns if c not in NON_FEATURE_COLUMNS]
```

Update its import line:

```python
from config import NON_FEATURE_COLUMNS, RETRAIN_WINDOW_MONTHS, TARGET_COLUMN
```

- [ ] **Step 7: Run the full test suite**

```bash
cd ~/IdeaProjects/mlops
uv run pytest -v
```

Expected: all previously-passing tests still pass (synthetic test fixtures don't include an
`id` column, and `NON_FEATURE_COLUMNS` still correctly excludes `TARGET_COLUMN`/`issue_d`
when `id` isn't present — this is a superset-tolerant change).

- [ ] **Step 8: Regenerate the real parquet files with the `id` column**

```bash
cd ~/IdeaProjects/mlops
uv run python -m data.prepare
```

Expected: reprints the schema diagnostics and row counts (should match Phase 1's original
run — 1,781,073 reference rows, 79,691 eval rows), and overwrites `var/processed/
reference.parquet` / `eval.parquet` with the `id` column now included.

- [ ] **Step 9: Sanity-check the regenerated parquet, then re-run Phase 1-3's report scripts**

```bash
cd ~/IdeaProjects/mlops
uv run python -c "
import pandas as pd
from config import PROCESSED_DIR
ref = pd.read_parquet(PROCESSED_DIR / 'reference.parquet')
print('has id column:', 'id' in ref.columns)
print(ref[['id', 'issue_d']].head(3))
"
uv run python -m reports.generate_phase1_report
uv run python -m reports.generate_phase3_report
```

Expected: `has id column: True`; both report scripts complete and print the same figures as
before (0.7102 pre-2019 holdout AUC; challenger beating champion on the Phase 3 comparison) —
confirming the `id` column addition didn't change any model behavior, since `NON_FEATURE_
COLUMNS` keeps it out of every pipeline's feature set.

- [ ] **Step 10: Commit**

```bash
cd ~/IdeaProjects/mlops
git add config.py data/prepare.py model/train.py lifecycle/evaluate.py lifecycle/retrain.py tests/test_prepare.py
git commit -m "Retain loan id through the data pipeline for explain_prediction lookups"
```

---

### Task 2: `get_champion_promoted_at` in the registry

**Files:**
- Modify: `lifecycle/registry.py`
- Test: `tests/test_lifecycle_gating.py` (append)

**Interfaces:**
- Produces: `lifecycle.registry.get_champion_promoted_at(name: str | None = None) -> int |
  None` — the champion model version's `creation_timestamp` (milliseconds since epoch), or
  `None` if there's no champion. Task 3 (`mcp_server/state.py`) uses this for
  `get_model_health`'s "days since last retrain".

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lifecycle_gating.py`:

```python
def test_get_champion_promoted_at_returns_timestamp_after_promotion():
    from lifecycle.registry import get_champion_promoted_at

    assert get_champion_promoted_at() is None

    version = _log_and_register_dummy_model()
    set_challenger(version)
    promote_challenger(confirm=True)

    promoted_at = get_champion_promoted_at()
    assert isinstance(promoted_at, int)
    assert promoted_at > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_lifecycle_gating.py::test_get_champion_promoted_at_returns_timestamp_after_promotion -v
```

Expected: FAIL with `ImportError: cannot import name 'get_champion_promoted_at'`.

- [ ] **Step 3: Implement `get_champion_promoted_at`**

Append to `lifecycle/registry.py`:

```python
def get_champion_promoted_at(name: str | None = None) -> int | None:
    """Milliseconds-since-epoch creation timestamp of the current champion version — used
    by get_model_health to report "days since last retrain". None if there's no champion."""
    name = _resolve_name(name)
    try:
        return _client().get_model_version_by_alias(name, _CHAMPION_ALIAS).creation_timestamp
    except MlflowException:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_lifecycle_gating.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/IdeaProjects/mlops
git add lifecycle/registry.py tests/test_lifecycle_gating.py
git commit -m "Add get_champion_promoted_at for model-health reporting"
```

---

### Task 3: `mcp_server/state.py` — shared runtime state

**Files:**
- Create: `mcp_server/__init__.py`
- Create: `mcp_server/state.py`
- Test: `tests/test_mcp_state.py`

**Interfaces:**
- Consumes: `build_model_pipeline`, `train_baseline` from `model.train`;
  `fit_statistical_reference`, `StatisticalDriftReference` from `drift.statistical`;
  `train_autoencoder`, `reconstruction_error`, `DriftAutoencoder` from `drift.autoencoder`;
  `get_champion_version`, `get_champion_promoted_at`, `promote_challenger`, `register_model`,
  `set_challenger` from `lifecycle.registry`; `config.NON_FEATURE_COLUMNS`,
  `config.NUMERIC_FEATURES`, `config.CATEGORICAL_FEATURES`, `config.PROCESSED_DIR`,
  `config.MLFLOW_TRACKING_URI`, `config.MODEL_NAME`, `config.TARGET_COLUMN`.
- Produces:
  - `ServerState` (dataclass) — fields `champion_pipeline: Pipeline`, `champion_version:
    str`, `champion_promoted_at_ms: int | None`, `reference_df: pd.DataFrame`, `eval_df:
    pd.DataFrame`, `statistical_reference: StatisticalDriftReference`, `autoencoder:
    DriftAutoencoder`, `reference_errors: np.ndarray`, `explainer` (a `shap.TreeExplainer`).
  - `get_state(force_refresh: bool = False) -> ServerState` — lazily builds and caches state
    at module level; rebuilds only if `force_refresh=True` or nothing's cached yet.
  - `refresh_champion() -> None` — after `promote_challenger`/`rollback`, reloads just the
    champion pipeline/version/timestamp/explainer in the cached state, without rebuilding the
    (expensive) drift reference and autoencoder, which don't change when the champion does.
  - `latest_reliable_batch(eval_df: pd.DataFrame, min_defaults: int =
    config.MIN_RELIABLE_DEFAULTS) -> tuple[str, pd.DataFrame]` — the most recent calendar
    month with at least `min_defaults` observed defaults, raising `ValueError` if none
    qualify.
  - `transform_for_drift(champion_pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray` —
    dense float32 array via the champion's own fitted preprocessor, for feeding the
    autoencoder/drift engine — reuses the live model's exact feature space rather than
    fitting an independent transformer (an improvement over Phase 2's report script, which
    fit its own separate preprocessor).
  - `SKOPS_TRUSTED_TYPES: list[str]` — the same three-type whitelist Phase 3 needed for
    `mlflow.sklearn.log_model`. Task 6 (`mcp_server/server.py`, mutating tools) reuses this
    constant directly rather than redefining it.

  Task 5 and 6 (`mcp_server/server.py`) call all of the above directly.

- [ ] **Step 1: Create the `mcp_server` package**

```bash
cd ~/IdeaProjects/mlops
mkdir -p mcp_server
touch mcp_server/__init__.py
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_mcp_state.py
import numpy as np
import pandas as pd

from mcp_server.state import latest_reliable_batch, transform_for_drift
from model.train import build_model_pipeline
from config import CATEGORICAL_FEATURES, NON_FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN


def _synthetic_df(n: int, start: str) -> pd.DataFrame:
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


def test_latest_reliable_batch_picks_most_recent_month_meeting_threshold():
    df = _synthetic_df(n=300, start="2019-01-01")
    df["default_flag"] = 0
    df.loc[df.index[:150], "default_flag"] = 1  # concentrate defaults in the earlier half

    label, batch = latest_reliable_batch(df, min_defaults=5)

    assert batch["default_flag"].sum() >= 5
    assert (batch["issue_d"].dt.to_period("M").astype(str) == label).all()


def test_latest_reliable_batch_raises_when_nothing_qualifies():
    df = _synthetic_df(n=10, start="2019-01-01")
    df["default_flag"] = 0
    try:
        latest_reliable_batch(df, min_defaults=1000)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_transform_for_drift_returns_dense_float32_array():
    df = _synthetic_df(n=50, start="2019-01-01")
    pipeline = build_model_pipeline()
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    pipeline.fit(df[feature_cols], df[TARGET_COLUMN])

    X = transform_for_drift(pipeline, df)

    assert X.dtype == np.float32
    assert X.shape[0] == len(df)
    assert not hasattr(X, "toarray")  # confirms it's dense, not still sparse
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_mcp_state.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mcp_server.state'`.

- [ ] **Step 4: Implement `mcp_server/state.py`**

```python
# mcp_server/state.py
"""Runtime state for the MCP server: loads/bootstraps the champion, fits the drift
reference, and builds a SHAP explainer — computed once per process and cached."""
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

import config
from drift.autoencoder import DriftAutoencoder, reconstruction_error, train_autoencoder
from drift.statistical import StatisticalDriftReference, fit_statistical_reference
from lifecycle.registry import (
    get_champion_promoted_at,
    get_champion_version,
    promote_challenger,
    register_model,
    set_challenger,
)
from model.train import train_baseline

# MLflow's sklearn flavor serializes with skops by default, which refuses to save a
# pipeline containing an XGBClassifier as "untrusted" unless explicitly whitelisted (Phase 3
# finding) — reused here for the bootstrap path and by mcp_server/server.py's mutating tools.
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]


@dataclass
class ServerState:
    champion_pipeline: Pipeline
    champion_version: str
    champion_promoted_at_ms: int | None
    reference_df: pd.DataFrame
    eval_df: pd.DataFrame
    statistical_reference: StatisticalDriftReference
    autoencoder: DriftAutoencoder
    reference_errors: np.ndarray
    explainer: object


_state: ServerState | None = None


def transform_for_drift(champion_pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    preprocessor = champion_pipeline.named_steps["preprocess"]
    X = preprocessor.transform(df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES])
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X.astype(np.float32)


def latest_reliable_batch(
    eval_df: pd.DataFrame, min_defaults: int = config.MIN_RELIABLE_DEFAULTS
) -> tuple[str, pd.DataFrame]:
    """Most recent calendar month with at least min_defaults observed defaults — the same
    reliability bar Phase 1 established to avoid right-censored, near-meaningless metrics
    close to the data cutoff."""
    df = eval_df.copy()
    df["month"] = df["issue_d"].dt.to_period("M").astype(str)
    defaults_by_month = df.groupby("month")[config.TARGET_COLUMN].sum()
    reliable_months = defaults_by_month[defaults_by_month >= min_defaults].index
    if len(reliable_months) == 0:
        raise ValueError("No calendar month has enough observed defaults to be reliable")
    latest_month = max(reliable_months)
    return latest_month, df[df["month"] == latest_month]


def _load_or_bootstrap_champion(reference_df: pd.DataFrame) -> tuple[Pipeline, str, int | None]:
    """Load the current registry champion, or bootstrap-train and register the Phase 1
    baseline if no champion exists yet (e.g. a fresh checkout with an empty registry)."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    version = get_champion_version()
    if version is None:
        pipeline, _, _ = train_baseline(reference_df)
        with mlflow.start_run(run_name="mcp-bootstrap-champion") as run:
            mlflow.sklearn.log_model(pipeline, "model", skops_trusted_types=SKOPS_TRUSTED_TYPES)
            model_uri = f"runs:/{run.info.run_id}/model"
        version = register_model(model_uri)
        set_challenger(version)
        promote_challenger(confirm=True)
    else:
        pipeline = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@champion")

    return pipeline, version, get_champion_promoted_at()


def _build_state() -> ServerState:
    reference_df = pd.read_parquet(config.PROCESSED_DIR / "reference.parquet")
    eval_df = pd.read_parquet(config.PROCESSED_DIR / "eval.parquet")

    champion_pipeline, champion_version, promoted_at_ms = _load_or_bootstrap_champion(reference_df)

    statistical_reference = fit_statistical_reference(
        reference_df, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES
    )
    reference_X = transform_for_drift(champion_pipeline, reference_df)
    autoencoder = train_autoencoder(reference_X, epochs=10, batch_size=1024)
    reference_errors = reconstruction_error(autoencoder, reference_X)

    explainer = shap.TreeExplainer(champion_pipeline.named_steps["model"])

    return ServerState(
        champion_pipeline=champion_pipeline,
        champion_version=champion_version,
        champion_promoted_at_ms=promoted_at_ms,
        reference_df=reference_df,
        eval_df=eval_df,
        statistical_reference=statistical_reference,
        autoencoder=autoencoder,
        reference_errors=reference_errors,
        explainer=explainer,
    )


def get_state(force_refresh: bool = False) -> ServerState:
    global _state
    if _state is None or force_refresh:
        _state = _build_state()
    return _state


def refresh_champion() -> None:
    """Reload just the champion pipeline/version/timestamp/explainer after promote/rollback
    — the drift reference and autoencoder are fit on the fixed pre-2019 reference window and
    don't change when the champion does, so rebuilding them here would be wasted work."""
    global _state
    if _state is None:
        return
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    pipeline = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@champion")
    _state.champion_pipeline = pipeline
    _state.champion_version = get_champion_version()
    _state.champion_promoted_at_ms = get_champion_promoted_at()
    _state.explainer = shap.TreeExplainer(pipeline.named_steps["model"])
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_mcp_state.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
cd ~/IdeaProjects/mlops
git add mcp_server/__init__.py mcp_server/state.py tests/test_mcp_state.py
git commit -m "Add MCP server runtime state: bootstrap, drift reference, SHAP explainer"
```

---

### Task 4: `mcp_server/schemas.py` — Pydantic output models

**Files:**
- Create: `mcp_server/schemas.py`
- Test: `tests/test_mcp_schemas.py`

**Interfaces:**
- Produces (all `pydantic.BaseModel` subclasses):
  - `ModelHealth`: `champion_version: str`, `days_since_last_retrain: float`,
    `recent_batch_auc: float`, `recent_batch_f1: float`, `recent_batch_label: str`,
    `drift_status: str`.
  - `FeatureDrift`: `feature: str`, `type: str`, `psi: float`, `ks_stat: float | None`.
  - `DriftReportOutput`: `window: str`, `status: str`, `autoencoder_status: str`,
    `autoencoder_fraction_above_threshold: float`, `statistical_status: str`,
    `top_drifting_features: list[FeatureDrift]`.
  - `RetrainResult`: `training_window_start: str`, `training_window_end: str`,
    `holdout_window_start: str`, `holdout_window_end: str`, `champion_auc: float`,
    `challenger_auc: float`, `champion_f1: float`, `challenger_f1: float`, `champion_brier:
    float`, `challenger_brier: float`, `challenger_wins: bool`, `challenger_version: str`.
  - `PromoteResult`: `new_champion_version: str`.
  - `RollbackResult`: `reverted_to_version: str`.
  - `FeatureContribution`: `feature: str`, `value: str`, `shap_contribution: float`.
  - `PredictionExplanation`: `loan_id: str`, `predicted_default_probability: float`,
    `top_contributions: list[FeatureContribution]`.

  Task 5 and 6 (`mcp_server/server.py`) construct and return these directly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcp_schemas.py
from mcp_server.schemas import (
    DriftReportOutput,
    FeatureContribution,
    FeatureDrift,
    ModelHealth,
    PredictionExplanation,
    PromoteResult,
    RetrainResult,
    RollbackResult,
)


def test_model_health_constructs_and_serializes():
    health = ModelHealth(
        champion_version="1", days_since_last_retrain=3.5, recent_batch_auc=0.68,
        recent_batch_f1=0.2, recent_batch_label="2019-10", drift_status="warning",
    )
    assert health.model_dump()["drift_status"] == "warning"


def test_drift_report_output_nests_feature_drift():
    report = DriftReportOutput(
        window="2019-06", status="breach", autoencoder_status="ok",
        autoencoder_fraction_above_threshold=0.01, statistical_status="breach",
        top_drifting_features=[FeatureDrift(feature="grade", type="categorical", psi=0.63, ks_stat=None)],
    )
    assert report.top_drifting_features[0].feature == "grade"


def test_retrain_result_constructs():
    result = RetrainResult(
        training_window_start="2018-11-01", training_window_end="2019-10-01",
        holdout_window_start="2019-11-01", holdout_window_end="2020-01-01",
        champion_auc=0.63, challenger_auc=0.68, champion_f1=0.02, challenger_f1=0.06,
        champion_brier=0.08, challenger_brier=0.07, challenger_wins=True, challenger_version="2",
    )
    assert result.challenger_wins is True


def test_promote_and_rollback_results_construct():
    assert PromoteResult(new_champion_version="2").new_champion_version == "2"
    assert RollbackResult(reverted_to_version="1").reverted_to_version == "1"


def test_prediction_explanation_nests_contributions():
    explanation = PredictionExplanation(
        loan_id="1001", predicted_default_probability=0.42,
        top_contributions=[FeatureContribution(feature="int_rate", value="15.0", shap_contribution=0.31)],
    )
    assert explanation.top_contributions[0].feature == "int_rate"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_mcp_schemas.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mcp_server.schemas'`.

- [ ] **Step 3: Implement `mcp_server/schemas.py`**

```python
# mcp_server/schemas.py
"""Pydantic models for MCP tool structured outputs (spec §7). All figures are computed from
a backtest over historical Lending Club data."""
from pydantic import BaseModel


class ModelHealth(BaseModel):
    champion_version: str
    days_since_last_retrain: float
    recent_batch_auc: float
    recent_batch_f1: float
    recent_batch_label: str
    drift_status: str


class FeatureDrift(BaseModel):
    feature: str
    type: str
    psi: float
    ks_stat: float | None


class DriftReportOutput(BaseModel):
    window: str
    status: str
    autoencoder_status: str
    autoencoder_fraction_above_threshold: float
    statistical_status: str
    top_drifting_features: list[FeatureDrift]


class RetrainResult(BaseModel):
    training_window_start: str
    training_window_end: str
    holdout_window_start: str
    holdout_window_end: str
    champion_auc: float
    challenger_auc: float
    champion_f1: float
    challenger_f1: float
    champion_brier: float
    challenger_brier: float
    challenger_wins: bool
    challenger_version: str


class PromoteResult(BaseModel):
    new_champion_version: str


class RollbackResult(BaseModel):
    reverted_to_version: str


class FeatureContribution(BaseModel):
    feature: str
    value: str
    shap_contribution: float


class PredictionExplanation(BaseModel):
    loan_id: str
    predicted_default_probability: float
    top_contributions: list[FeatureContribution]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_mcp_schemas.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/IdeaProjects/mlops
git add mcp_server/schemas.py tests/test_mcp_schemas.py
git commit -m "Add Pydantic schemas for MCP tool outputs"
```

---

### Task 5: Read-only tools — `get_model_health`, `get_drift_report`, `explain_prediction`

**Files:**
- Create: `mcp_server/server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `ServerState`, `get_state`, `latest_reliable_batch`, `transform_for_drift` from
  `mcp_server.state`; all schemas from `mcp_server.schemas`; `evaluate_drift` from
  `drift.engine`; `config.NON_FEATURE_COLUMNS`, `config.NUMERIC_FEATURES`,
  `config.CATEGORICAL_FEATURES`, `config.TARGET_COLUMN`.
- Produces:
  - `get_model_health_impl() -> ModelHealth`
  - `get_drift_report_impl(window: str = "last_30d") -> DriftReportOutput`
  - `explain_prediction_impl(loan_id: str) -> PredictionExplanation`
  - `mcp = MCPServer("creditpulse")` and the corresponding `@mcp.tool()`-decorated
    `get_model_health`, `get_drift_report`, `explain_prediction` functions, each calling
    straight through to its `_impl`.

  Task 6 appends the three mutating tools to this same file and reuses `mcp`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mcp_server.py
import numpy as np
import pandas as pd
import pytest

import config
from drift.autoencoder import reconstruction_error, train_autoencoder
from drift.statistical import fit_statistical_reference
from mcp_server import server, state
from model.train import build_model_pipeline


def _synthetic_loans(n: int, start: str, id_prefix: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n, freq="3D")
    risk = rng.normal(size=n)
    default_prob = 1 / (1 + np.exp(-risk))
    default_flag = (rng.uniform(size=n) < default_prob).astype(int)
    return pd.DataFrame({
        "id": [f"{id_prefix}{i}" for i in range(n)],
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


@pytest.fixture()
def synthetic_state(monkeypatch):
    reference_df = _synthetic_loans(300, "2017-01-01", "REF")
    eval_df = _synthetic_loans(200, "2019-01-01", "EVAL")

    pipeline = build_model_pipeline()
    feature_cols = [c for c in reference_df.columns if c not in config.NON_FEATURE_COLUMNS]
    pipeline.fit(reference_df[feature_cols], reference_df[config.TARGET_COLUMN])

    statistical_reference = fit_statistical_reference(
        reference_df, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES
    )
    reference_X = state.transform_for_drift(pipeline, reference_df)
    autoencoder = train_autoencoder(reference_X, epochs=3, batch_size=64)
    reference_errors = reconstruction_error(autoencoder, reference_X)

    import shap
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])

    fake_state = state.ServerState(
        champion_pipeline=pipeline,
        champion_version="1",
        champion_promoted_at_ms=0,
        reference_df=reference_df,
        eval_df=eval_df,
        statistical_reference=statistical_reference,
        autoencoder=autoencoder,
        reference_errors=reference_errors,
        explainer=explainer,
    )
    monkeypatch.setattr(state, "get_state", lambda force_refresh=False: fake_state)
    return fake_state


def test_get_model_health_impl_returns_populated_health(synthetic_state):
    health = server.get_model_health_impl()
    assert health.champion_version == "1"
    assert 0.0 <= health.recent_batch_auc <= 1.0
    assert health.drift_status in ("ok", "warning", "breach")


def test_get_drift_report_impl_resolves_explicit_month_window(synthetic_state):
    month = synthetic_state.eval_df["issue_d"].dt.to_period("M").astype(str).iloc[0]
    report = server.get_drift_report_impl(window=month)
    assert report.window == month
    assert report.status in ("ok", "warning", "breach")
    assert len(report.top_drifting_features) > 0


def test_get_drift_report_impl_resolves_last_n_days_window(synthetic_state):
    report = server.get_drift_report_impl(window="last_30d")
    assert report.window == "last_30d"
    assert report.status in ("ok", "warning", "breach")


def test_explain_prediction_impl_returns_top_contributions(synthetic_state):
    loan_id = synthetic_state.eval_df["id"].iloc[0]
    explanation = server.explain_prediction_impl(loan_id)
    assert explanation.loan_id == str(loan_id)
    assert 0.0 <= explanation.predicted_default_probability <= 1.0
    assert len(explanation.top_contributions) == 5


def test_explain_prediction_impl_raises_for_unknown_loan_id(synthetic_state):
    try:
        server.explain_prediction_impl("does-not-exist")
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_mcp_server.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mcp_server.server'`.

- [ ] **Step 3: Implement `mcp_server/server.py` (read-only tools)**

```python
# mcp_server/server.py
"""MCP server exposing CreditPulse's model lifecycle as tools (spec §7). All results are
backtest computations over historical Lending Club data — never a live evaluation."""
import re

import pandas as pd
from mcp.server.mcpserver import MCPServer
from sklearn.metrics import f1_score

import config
from drift.engine import evaluate_drift
from mcp_server.schemas import (
    DriftReportOutput,
    FeatureContribution,
    FeatureDrift,
    ModelHealth,
    PredictionExplanation,
)
from mcp_server.state import get_state, latest_reliable_batch, transform_for_drift
from model.train import evaluate_auc

mcp = MCPServer("creditpulse")


def _raw_feature_for_transformed_name(name: str) -> str:
    """Map a ColumnTransformer output name (e.g. "categorical__grade_A") back to its raw
    feature name ("grade"), so SHAP contributions can be reported in human terms rather than
    one-hot-encoded column names."""
    prefix, _, rest = name.partition("__")
    if prefix == "numeric":
        return rest
    if prefix == "categorical":
        candidates = [f for f in config.CATEGORICAL_FEATURES if rest.startswith(f + "_") or rest == f]
        if candidates:
            return max(candidates, key=len)
    return name


def _resolve_window(eval_df: pd.DataFrame, window: str) -> tuple[str, pd.DataFrame]:
    """Resolve a window spec against the eval data's own max issue_d as "now" — this is a
    backtest over historical data, not a live stream, so "last_30d" means the last 30 days
    of the most recent data available, not 30 real days ago."""
    match = re.fullmatch(r"last_(\d+)d", window)
    if match:
        days = int(match.group(1))
        now = eval_df["issue_d"].max()
        start = now - pd.Timedelta(days=days)
        subset = eval_df[(eval_df["issue_d"] > start) & (eval_df["issue_d"] <= now)]
        return window, subset
    df = eval_df.copy()
    df["month"] = df["issue_d"].dt.to_period("M").astype(str)
    subset = df[df["month"] == window]
    if len(subset) == 0:
        raise ValueError(f"No data found for window={window!r}")
    return window, subset


def get_model_health_impl() -> ModelHealth:
    state = get_state()
    batch_label, batch_df = latest_reliable_batch(state.eval_df)

    recent_auc = evaluate_auc(state.champion_pipeline, batch_df)
    feature_cols = [c for c in batch_df.columns if c not in config.NON_FEATURE_COLUMNS]
    y_prob = state.champion_pipeline.predict_proba(batch_df[feature_cols])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    recent_f1 = f1_score(batch_df[config.TARGET_COLUMN], y_pred)

    comparison_X = transform_for_drift(state.champion_pipeline, batch_df)
    report = evaluate_drift(
        window=batch_label,
        statistical_reference=state.statistical_reference,
        autoencoder=state.autoencoder,
        reference_errors=state.reference_errors,
        comparison_df=batch_df,
        comparison_X=comparison_X,
        numeric_features=config.NUMERIC_FEATURES,
        categorical_features=config.CATEGORICAL_FEATURES,
    )

    if state.champion_promoted_at_ms:
        days_since = (pd.Timestamp.now().timestamp() * 1000 - state.champion_promoted_at_ms) / 86_400_000
    else:
        days_since = 0.0

    return ModelHealth(
        champion_version=state.champion_version,
        days_since_last_retrain=round(days_since, 1),
        recent_batch_auc=recent_auc,
        recent_batch_f1=recent_f1,
        recent_batch_label=batch_label,
        drift_status=report.status,
    )


def get_drift_report_impl(window: str = "last_30d") -> DriftReportOutput:
    state = get_state()
    resolved_window, subset = _resolve_window(state.eval_df, window)
    comparison_X = transform_for_drift(state.champion_pipeline, subset)

    report = evaluate_drift(
        window=resolved_window,
        statistical_reference=state.statistical_reference,
        autoencoder=state.autoencoder,
        reference_errors=state.reference_errors,
        comparison_df=subset,
        comparison_X=comparison_X,
        numeric_features=config.NUMERIC_FEATURES,
        categorical_features=config.CATEGORICAL_FEATURES,
    )

    top_features = [
        FeatureDrift(
            feature=row.feature, type=row.type, psi=row.psi,
            ks_stat=None if pd.isna(row.ks_stat) else row.ks_stat,
        )
        for row in report.feature_drift.head(5).itertuples()
    ]
    return DriftReportOutput(
        window=resolved_window,
        status=report.status,
        autoencoder_status=report.autoencoder_status,
        autoencoder_fraction_above_threshold=report.autoencoder_fraction_above_threshold,
        statistical_status=report.statistical_status,
        top_drifting_features=top_features,
    )


def explain_prediction_impl(loan_id: str) -> PredictionExplanation:
    state = get_state()
    combined = pd.concat([state.reference_df, state.eval_df], ignore_index=True)
    matches = combined[combined["id"].astype(str) == str(loan_id)]
    if len(matches) == 0:
        raise ValueError(f"No loan found with id={loan_id!r}")
    loan = matches.iloc[0]

    feature_cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    X_raw = loan[feature_cols].to_frame().T
    preprocessor = state.champion_pipeline.named_steps["preprocess"]
    X_transformed = preprocessor.transform(X_raw)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    predicted_proba = float(
        state.champion_pipeline.named_steps["model"].predict_proba(X_transformed)[0, 1]
    )
    shap_row = state.explainer.shap_values(X_transformed)[0]
    transformed_names = preprocessor.get_feature_names_out()

    contributions: dict[str, float] = {}
    for name, value in zip(transformed_names, shap_row):
        raw = _raw_feature_for_transformed_name(name)
        contributions[raw] = contributions.get(raw, 0.0) + float(value)

    top = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    top_contributions = [
        FeatureContribution(feature=f, value=str(loan[f]), shap_contribution=v) for f, v in top
    ]

    return PredictionExplanation(
        loan_id=str(loan_id),
        predicted_default_probability=predicted_proba,
        top_contributions=top_contributions,
    )


@mcp.tool()
def get_model_health() -> ModelHealth:
    """Current champion model version, days since last retrain, AUC/F1 on the most recently
    reliably-labeled historical batch, and current overall drift status. Backtest on
    historical Lending Club data, not a live evaluation."""
    return get_model_health_impl()


@mcp.tool()
def get_drift_report(window: str = "last_30d") -> DriftReportOutput:
    """Per-feature PSI/KS drift and the autoencoder's reconstruction-error signal for a
    window of historical loans, plus which features are driving any breach. `window`
    accepts "last_Nd" (relative to the most recent data available) or an explicit "YYYY-MM"
    month."""
    return get_drift_report_impl(window)


@mcp.tool()
def explain_prediction(loan_id: str) -> PredictionExplanation:
    """The champion model's predicted default probability for a given historical loan, plus
    its top SHAP feature contributions — framed like an adverse-action explanation."""
    return explain_prediction_impl(loan_id)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_mcp_server.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/IdeaProjects/mlops
git add mcp_server/server.py tests/test_mcp_server.py
git commit -m "Add read-only MCP tools: get_model_health, get_drift_report, explain_prediction"
```

---

### Task 6: Mutating tools — `trigger_retrain`, `promote_challenger`, `rollback`

**Files:**
- Modify: `mcp_server/server.py`
- Test: `tests/test_mcp_server.py` (append)

**Interfaces:**
- Consumes: `select_recent_window`, `retrain_challenger` from `lifecycle.retrain`;
  `compare_champion_challenger` from `lifecycle.evaluate`; `ConfirmationRequired`,
  `promote_challenger` (as `registry_promote_challenger`), `register_model`, `rollback` (as
  `registry_rollback`), `set_challenger` from `lifecycle.registry`; `refresh_champion`,
  `SKOPS_TRUSTED_TYPES` from `mcp_server.state`; `RetrainResult`, `PromoteResult`,
  `RollbackResult` from `mcp_server.schemas`.
- Produces:
  - `trigger_retrain_impl() -> RetrainResult` — never promotes anything (spec §7 tool 3).
  - `promote_challenger_impl(confirm: bool) -> PromoteResult` — delegates confirm-checking
    entirely to `lifecycle.registry.promote_challenger`.
  - `rollback_impl(confirm: bool) -> RollbackResult` — same, via
    `lifecycle.registry.rollback`.
  - `@mcp.tool(name="trigger_retrain")` `trigger_retrain`, `@mcp.tool(name=
    "promote_challenger")` `promote_challenger_tool`, `@mcp.tool(name="rollback")`
    `rollback_tool` — the explicit `name=` keeps the MCP-visible tool name matching spec §7
    exactly, while the Python function name avoids colliding with the imported
    `lifecycle.registry` functions of the same name.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mcp_server.py`:

```python
import mlflow
import mlflow.sklearn


@pytest.fixture(autouse=True)
def isolated_mlflow_for_mutating_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'test_mlflow.db'}")
    monkeypatch.setattr(config, "MODEL_NAME", "test-mcp-model")
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.create_experiment("test-mcp-experiment", artifact_location=f"file://{tmp_path / 'mlartifacts'}")
    mlflow.set_experiment("test-mcp-experiment")


def test_trigger_retrain_impl_never_promotes(synthetic_state):
    from lifecycle.registry import get_challenger_version, get_champion_version

    result = server.trigger_retrain_impl()

    assert 0.0 <= result.champion_auc <= 1.0
    assert 0.0 <= result.challenger_auc <= 1.0
    assert get_challenger_version() == result.challenger_version
    # trigger_retrain must never touch the champion alias itself (spec: "Does not promote")
    assert get_champion_version() != result.challenger_version or get_champion_version() is None


def test_promote_challenger_impl_refuses_without_confirm(synthetic_state):
    from lifecycle.registry import ConfirmationRequired

    server.trigger_retrain_impl()
    with pytest.raises(ConfirmationRequired):
        server.promote_challenger_impl(confirm=False)


def test_promote_and_rollback_impl_round_trip(synthetic_state):
    from lifecycle.registry import get_champion_version, register_model, set_challenger

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            synthetic_state.champion_pipeline, "model",
            skops_trusted_types=state.SKOPS_TRUSTED_TYPES,
        )
        model_uri = f"runs:/{run.info.run_id}/model"
    bootstrap_version = register_model(model_uri)
    set_challenger(bootstrap_version)
    server.promote_challenger_impl(confirm=True)
    assert get_champion_version() == bootstrap_version

    retrain_result = server.trigger_retrain_impl()
    promote_result = server.promote_challenger_impl(confirm=True)
    assert promote_result.new_champion_version == retrain_result.challenger_version
    assert get_champion_version() == retrain_result.challenger_version

    rollback_result = server.rollback_impl(confirm=True)
    assert rollback_result.reverted_to_version == bootstrap_version
    assert get_champion_version() == bootstrap_version
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_mcp_server.py -v -k "retrain or promote or rollback"
```

Expected: FAIL with `AttributeError: module 'mcp_server.server' has no attribute
'trigger_retrain_impl'`.

- [ ] **Step 3: Append the mutating tools to `mcp_server/server.py`**

Add these imports to the top of `mcp_server/server.py`, alongside the existing ones:

```python
import mlflow
import mlflow.sklearn

from lifecycle.evaluate import compare_champion_challenger
from lifecycle.registry import (
    ConfirmationRequired,
    promote_challenger as registry_promote_challenger,
    register_model,
    rollback as registry_rollback,
    set_challenger,
)
from lifecycle.retrain import retrain_challenger, select_recent_window
from mcp_server.schemas import PromoteResult, RetrainResult, RollbackResult
from mcp_server.state import SKOPS_TRUSTED_TYPES, refresh_champion
```

Append the implementation functions and tools to the end of `mcp_server/server.py`:

```python
def trigger_retrain_impl() -> RetrainResult:
    """Retrain a challenger on the most recent available window and compare it against the
    champion on a held-out slice neither model trained on. Never promotes anything."""
    state_obj = get_state()
    all_loans_df = pd.concat([state_obj.reference_df, state_obj.eval_df], ignore_index=True)
    as_of = state_obj.eval_df["issue_d"].max() - pd.DateOffset(months=2)
    training_window = select_recent_window(all_loans_df, as_of, months=config.RETRAIN_WINDOW_MONTHS)
    holdout_start, holdout_end = as_of, as_of + pd.DateOffset(months=2)
    holdout_df = all_loans_df[
        (all_loans_df["issue_d"] >= holdout_start) & (all_loans_df["issue_d"] < holdout_end)
    ]

    challenger = retrain_challenger(training_window)
    comparison = compare_champion_challenger(state_obj.champion_pipeline, challenger, holdout_df)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name=f"mcp-trigger-retrain-{as_of.date()}") as run:
        mlflow.sklearn.log_model(challenger, "model", skops_trusted_types=SKOPS_TRUSTED_TYPES)
        mlflow.log_metric("auc", comparison.challenger_auc)
        model_uri = f"runs:/{run.info.run_id}/model"
    challenger_version = register_model(model_uri)
    set_challenger(challenger_version)

    return RetrainResult(
        training_window_start=str(training_window["issue_d"].min().date()),
        training_window_end=str(training_window["issue_d"].max().date()),
        holdout_window_start=str(holdout_start.date()),
        holdout_window_end=str(holdout_end.date()),
        champion_auc=comparison.champion_auc,
        challenger_auc=comparison.challenger_auc,
        champion_f1=comparison.champion_f1,
        challenger_f1=comparison.challenger_f1,
        champion_brier=comparison.champion_brier,
        challenger_brier=comparison.challenger_brier,
        challenger_wins=comparison.challenger_wins,
        challenger_version=challenger_version,
    )


def promote_challenger_impl(confirm: bool) -> PromoteResult:
    new_version = registry_promote_challenger(confirm=confirm)
    refresh_champion()
    return PromoteResult(new_champion_version=new_version)


def rollback_impl(confirm: bool) -> RollbackResult:
    reverted_version = registry_rollback(confirm=confirm)
    refresh_champion()
    return RollbackResult(reverted_to_version=reverted_version)


@mcp.tool(name="trigger_retrain")
def trigger_retrain() -> RetrainResult:
    """Retrain a challenger on the most recent available window, log the run to MLflow, and
    return champion-vs-challenger metrics (AUC, F1, calibration). Does NOT promote anything."""
    return trigger_retrain_impl()


@mcp.tool(name="promote_challenger")
def promote_challenger_tool(confirm: bool) -> PromoteResult:
    """Promote the current challenger to champion in the MLflow registry. Refuses unless
    confirm is exactly True — no silent auto-promotion, ever. Returns the new champion's
    version."""
    return promote_challenger_impl(confirm)


@mcp.tool(name="rollback")
def rollback_tool(confirm: bool) -> RollbackResult:
    """Revert to the previous champion version. Refuses unless confirm is exactly True."""
    return rollback_impl(confirm)


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_mcp_server.py -v
```

Expected: 8 passed (5 from Task 5, 3 from this task).

- [ ] **Step 5: Run the full test suite**

```bash
cd ~/IdeaProjects/mlops
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
cd ~/IdeaProjects/mlops
git add mcp_server/server.py tests/test_mcp_server.py
git commit -m "Add mutating MCP tools: trigger_retrain, promote_challenger, rollback"
```

---

### Task 7: Real end-to-end MCP client walkthrough

**Files:**
- Create: `scripts/run_mcp_walkthrough.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `mcp.client.stdio.stdio_client`, `mcp.client.stdio.StdioServerParameters`,
  `mcp.client.session.ClientSession` (installed `mcp` SDK's real client API — verified: both
  expose `.initialize()`, `.list_tools()`, `.call_tool(name, arguments)`).
- Produces: a script that drives the real server via `mcp_server/server.py` as a subprocess
  over stdio, calling all six tools in the order spec §8 Phase 4 requires (health → drift
  report → retrain → review comparison → promote with confirmation), printing a transcript.
  No other module depends on this script.

- [ ] **Step 1: Write the walkthrough script**

```python
# scripts/run_mcp_walkthrough.py
"""Drives the CreditPulse MCP server end-to-end via a real MCP client, over stdio, exactly
as spec §8 Phase 4 requires: health check -> drift report -> retrain -> review -> promote
with confirmation. Prints a transcript suitable for pasting into the README. Everything
below is a backtest over historical Lending Club data, not a live production system."""
import asyncio
import json
import sys

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _print_result(label: str, result: types.CallToolResult) -> None:
    print(f"\n--- {label} ---")
    for block in result.content:
        if isinstance(block, types.TextContent):
            try:
                print(json.dumps(json.loads(block.text), indent=2))
            except (json.JSONDecodeError, TypeError):
                print(block.text)


async def main() -> None:
    server_params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Connected. Available tools:", [t.name for t in tools.tools])

            health = await session.call_tool("get_model_health", {})
            _print_result("get_model_health()", health)

            drift = await session.call_tool("get_drift_report", {"window": "2019-06"})
            _print_result('get_drift_report(window="2019-06")', drift)

            retrain = await session.call_tool("trigger_retrain", {})
            _print_result("trigger_retrain()", retrain)

            refused = await session.call_tool("promote_challenger", {"confirm": False})
            _print_result("promote_challenger(confirm=False) — expected refusal", refused)

            promoted = await session.call_tool("promote_challenger", {"confirm": True})
            _print_result("promote_challenger(confirm=True)", promoted)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run the walkthrough against the real server**

```bash
cd ~/IdeaProjects/mlops
uv run python scripts/run_mcp_walkthrough.py 2>&1 | tee /tmp/mcp_walkthrough_transcript.txt
```

Expected: connects, lists all six tool names, then prints each tool's JSON result in order.
`promote_challenger(confirm=False)` should show an error/refusal result (a real MCP
`CallToolResult` with `isError=True` or an equivalent error payload — inspect the actual
printed output and confirm it clearly indicates refusal, since this is the spec's core
safety property showing up at the client boundary, not just inside the Python test suite).
If the refusal doesn't come through clearly in the tool result, that's a real gap to fix:
FastMCP-style servers typically translate a raised exception into an error `CallToolResult`
automatically, but confirm this by inspecting the actual transcript rather than assuming.

- [ ] **Step 3: Copy the real transcript into the README**

```bash
cd ~/IdeaProjects/mlops
cat /tmp/mcp_walkthrough_transcript.txt
```

Add a `## MCP Lifecycle Walkthrough` section to `README.md` (create `README.md` with at
least a title and this section if it doesn't exist yet) containing the real printed
transcript from Step 2, wrapped in a fenced code block, with a one-line preface stating it
was captured by running `scripts/run_mcp_walkthrough.py` against a real MCP client, and that
every number in it is from a backtest on historical Lending Club data.

- [ ] **Step 4: Commit**

```bash
cd ~/IdeaProjects/mlops
git add scripts/run_mcp_walkthrough.py README.md
git commit -m "Add real MCP client walkthrough script and README transcript"
```

---

## Self-review notes

- **Spec coverage:** all six tools from spec §7 implemented — `get_model_health` ✓,
  `get_drift_report(window)` ✓, `trigger_retrain` (never promotes) ✓, `promote_challenger
  (confirm)` (refuses without confirm=True, delegates to Phase 3's tested gate) ✓, `rollback
  (confirm)` (same) ✓, `explain_prediction(loan_id)` (SHAP top contributions, adverse-action
  framing) ✓. Pydantic structured output for every tool ✓. Phase 4 acceptance criterion — a
  full walkthrough transcript from a real MCP client, saved in the README — is exactly
  Task 7 ✓.
- **Placeholder scan:** none remaining. Task 7 Step 2's instruction to "inspect the actual
  printed output" for the refusal case is a real verification step with a concrete command,
  not a vague hedge — the exact shape of an error `CallToolResult` depends on runtime
  behavior that's more reliable to observe than to guess at plan-writing time (this mirrors
  every prior phase's discipline of running real scripts against real data rather than
  trusting assumptions).
- **Type consistency:** `ServerState` (Task 3) fields are read identically in Task 5 and 6's
  tool implementations. All six `*Result`/`*Output`/`ModelHealth`/`PredictionExplanation`
  schemas (Task 4) are constructed with matching field names in Task 5/6. `get_state`,
  `refresh_champion`, `latest_reliable_batch`, `transform_for_drift`, `SKOPS_TRUSTED_TYPES`
  (Task 3) are imported and called with matching signatures in Tasks 5-6.
