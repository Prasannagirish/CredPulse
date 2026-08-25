# CreditPulse Phase 1 — Data Prep + Baseline Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user is
> executing this plan to learn the system, one step at a time — narrate what each step is
> doing and why before running it, don't just execute silently.

**Goal:** Load the real Lending Club accepted-loans CSV, build a leak-safe temporal train/eval
split, train an XGBoost baseline champion on pre-2019 data, and prove the core premise of the
whole project — that a static model's AUC visibly declines through 2019–2020 without
retraining.

**Architecture:** `config.py` centralizes paths/dates/constants. `model/features.py` defines
an explicit feature **allow-list** and the shared `ColumnTransformer` preprocessing pipeline.
`data/prepare.py` loads the raw CSV, derives the target, applies the temporal split, and writes
partitioned parquet. `model/train.py` fits the baseline XGBoost model on the reference window
and evaluates it, unmodified, against each 2019/2020 quarter.

**Tech Stack:** Python 3.12 (via `uv`), pandas, pyarrow, scikit-learn, xgboost, mlflow
(SQLite backend), matplotlib, pytest.

## Global Constraints

- Python 3.12 pinned — `uv venv --python 3.12` (system default 3.14 is unsupported by
  xgboost/torch/mlflow wheels).
- MLflow backend store must be SQLite (`sqlite:///var/mlflow.db`), not the default file store —
  file store doesn't support the model registry needed in Phase 3/4.
- Raw data source: `~/Downloads/creditpulse-data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv`
  (already downloaded, 2,260,702 rows). Never copy this file into the git repo.
- Target: positive = `loan_status` in `{"Charged Off", "Default"}`; negative = `"Fully Paid"`;
  everything else (`Current`, `Late (*)`, `In Grace Period`, `Does not meet the credit
  policy.*`) is dropped as a non-terminal/unresolved outcome — this must be a named, tested
  function, not inline filtering, because it's the single most safety-critical line in the
  whole project (get it wrong and every downstream number is fiction).
- Features: **allow-list only** — `loan_amnt`, `term`, `int_rate`, `grade`, `sub_grade`,
  `emp_length`, `home_ownership`, `annual_inc`, `verification_status`, `dti`, `revol_util`,
  `fico_range_low`, `fico_range_high`, `purpose`. No other raw column may reach the model.
  This is a safe-by-default list: an unrecognized column is silently excluded rather than
  silently included, so a future schema change can't reintroduce a leaked column.
- Temporal split: reference/training window = loans with `issue_d` before 2019-01-01;
  evaluation window = `issue_d` in 2019-01-01..2020-12-31. Never shuffle across this boundary.
- Every printed/plotted result must be labeled "backtest on historical data" — even at this
  early phase, so the habit is established before Phase 5's headline numbers exist.
- All generated data artifacts (parquet, mlflow.db) live under `var/` at the repo root,
  gitignored. Small chart PNGs under `reports/` are committed as build output.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `config.py`
- Create: `data/__init__.py`, `model/__init__.py`, `tests/__init__.py`
- Create: `var/.gitkeep`, `reports/.gitkeep`

**Interfaces:**
- Produces: `config.py` module with constants `RAW_CSV_PATH: Path`, `PROCESSED_DIR: Path`,
  `MLFLOW_TRACKING_URI: str`, `REPORTS_DIR: Path`, `REFERENCE_END: str` (`"2019-01-01"`),
  `EVAL_START: str` (`"2019-01-01"`), `EVAL_END: str` (`"2021-01-01"`, exclusive upper bound).
  All later tasks import these instead of hardcoding paths/dates.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "creditpulse"
version = "0.1.0"
description = "Agentic MLOps control plane for credit risk models (backtest on historical data)"
requires-python = ">=3.12,<3.13"
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "scipy>=1.13",
    "pyarrow>=16.0",
    "scikit-learn>=1.5",
    "xgboost>=2.0",
    "torch>=2.3",
    "shap>=0.45",
    "mlflow>=2.14",
    "mcp>=1.0",
    "pydantic>=2.7",
    "matplotlib>=3.9",
]

[tool.uv]
dev-dependencies = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["."]
include = ["*.py", "data/**", "model/**"]
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
var/
!var/.gitkeep
.pytest_cache/
*.egg-info/
```

- [ ] **Step 3: Write `config.py`**

```python
"""Central paths and constants shared across data prep, drift, backtest, and the MCP server."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

RAW_CSV_PATH = Path.home() / "Downloads" / "creditpulse-data" / "accepted_2007_to_2018q4.csv" / "accepted_2007_to_2018Q4.csv"

VAR_DIR = REPO_ROOT / "var"
PROCESSED_DIR = VAR_DIR / "processed"
REPORTS_DIR = REPO_ROOT / "reports"

MLFLOW_TRACKING_URI = f"sqlite:///{VAR_DIR / 'mlflow.db'}"

# Temporal split boundaries (spec §3) — loans issued before REFERENCE_END are the
# reference/training window; loans issued in [EVAL_START, EVAL_END) are the drift/eval window.
REFERENCE_END = "2019-01-01"
EVAL_START = "2019-01-01"
EVAL_END = "2021-01-01"

TARGET_COLUMN = "default_flag"

# Safe-by-default: only these raw columns may become model features. Anything not listed
# here is excluded, even if the raw schema changes to add new columns.
FEATURE_COLUMNS = [
    "loan_amnt",
    "term",
    "int_rate",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "dti",
    "revol_util",
    "fico_range_low",
    "fico_range_high",
    "purpose",
]

NUMERIC_FEATURES = [
    "loan_amnt", "term", "int_rate", "emp_length", "annual_inc",
    "dti", "revol_util", "fico_range_low", "fico_range_high",
]
CATEGORICAL_FEATURES = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]
```

- [ ] **Step 4: Create package inits and gitkeep files**

```bash
cd ~/IdeaProjects/mlops
mkdir -p data model tests var reports
touch data/__init__.py model/__init__.py tests/__init__.py var/.gitkeep reports/.gitkeep
```

- [ ] **Step 5: Create the venv and install dependencies**

```bash
cd ~/IdeaProjects/mlops
uv venv --python 3.12
uv sync
```

Expected: `.venv/` created, dependency resolution succeeds, `uv.lock` written.

- [ ] **Step 6: Verify the environment**

```bash
cd ~/IdeaProjects/mlops
uv run python -c "import pandas, xgboost, sklearn, mlflow, torch; print('ok')"
```

Expected: prints `ok` with no import errors.

- [ ] **Step 7: Commit**

```bash
cd ~/IdeaProjects/mlops
git add pyproject.toml .gitignore config.py data/__init__.py model/__init__.py tests/__init__.py var/.gitkeep reports/.gitkeep uv.lock
git commit -m "Scaffold project: pyproject.toml, config, package layout"
```

---

### Task 2: Target derivation

**Files:**
- Create: `model/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing beyond a raw `pandas.DataFrame` with a `loan_status` column.
- Produces: `derive_target(df: pd.DataFrame) -> pd.DataFrame` — returns a **new** DataFrame
  containing only rows with a resolved outcome (`Fully Paid`, `Charged Off`, `Default`), with
  a new integer column named by `config.TARGET_COLUMN` (1 = default, 0 = fully paid). Row
  order and index are not guaranteed to be preserved. Later tasks (Task 3, `prepare.py`) call
  this before feature selection.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features.py
import pandas as pd
import pytest

from model.features import derive_target


def test_derive_target_labels_charged_off_and_default_as_positive():
    df = pd.DataFrame({
        "loan_status": ["Charged Off", "Default", "Fully Paid"],
        "loan_amnt": [1000, 2000, 3000],
    })
    result = derive_target(df)
    assert list(result["default_flag"]) == sorted(list(result["default_flag"]))  # placeholder, replaced below
```

Replace that placeholder assertion with real ones — write the actual test file content:

```python
# tests/test_features.py
import pandas as pd

from model.features import derive_target


def test_derive_target_marks_charged_off_and_default_positive():
    df = pd.DataFrame({
        "loan_status": ["Charged Off", "Default", "Fully Paid"],
        "loan_amnt": [1000, 2000, 3000],
    })
    result = derive_target(df)
    labels = dict(zip(result["loan_amnt"], result["default_flag"]))
    assert labels[1000] == 1
    assert labels[2000] == 1
    assert labels[3000] == 0


def test_derive_target_drops_non_terminal_statuses():
    df = pd.DataFrame({
        "loan_status": [
            "Current", "Late (31-120 days)", "In Grace Period",
            "Late (16-30 days)", "Fully Paid", "Charged Off",
        ],
        "loan_amnt": [1, 2, 3, 4, 5, 6],
    })
    result = derive_target(df)
    assert set(result["loan_amnt"]) == {5, 6}


def test_derive_target_drops_does_not_meet_credit_policy_rows():
    df = pd.DataFrame({
        "loan_status": [
            "Does not meet the credit policy. Status:Fully Paid",
            "Does not meet the credit policy. Status:Charged Off",
            "Fully Paid",
        ],
        "loan_amnt": [1, 2, 3],
    })
    result = derive_target(df)
    assert set(result["loan_amnt"]) == {3}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_features.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError: cannot import name 'derive_target'`.

- [ ] **Step 3: Implement `derive_target`**

```python
# model/features.py
"""Feature allow-list, target derivation, and the shared preprocessing pipeline."""
import pandas as pd

from config import TARGET_COLUMN

_POSITIVE_STATUSES = {"Charged Off", "Default"}
_NEGATIVE_STATUSES = {"Fully Paid"}


def derive_target(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to resolved loans and attach a binary default_flag column.

    Loans still "Current" or in any other non-terminal state (Late, In Grace Period,
    or the legacy "Does not meet the credit policy." variants) are excluded — their
    outcome isn't resolved yet, so including them would leak censored labels into
    training (spec §3).
    """
    resolved = df[df["loan_status"].isin(_POSITIVE_STATUSES | _NEGATIVE_STATUSES)].copy()
    resolved[TARGET_COLUMN] = resolved["loan_status"].isin(_POSITIVE_STATUSES).astype(int)
    return resolved
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_features.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/IdeaProjects/mlops
git add model/features.py tests/test_features.py
git commit -m "Add target derivation with non-terminal-status exclusion"
```

---

### Task 3: Feature allow-list selection and preprocessing pipeline

**Files:**
- Modify: `model/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `config.FEATURE_COLUMNS`, `config.NUMERIC_FEATURES`, `config.CATEGORICAL_FEATURES`,
  `config.TARGET_COLUMN`.
- Produces:
  - `select_features(df: pd.DataFrame) -> pd.DataFrame` — returns a DataFrame containing only
    the columns in `config.FEATURE_COLUMNS` (raw, not yet transformed) plus `TARGET_COLUMN`
    when present in the input.
  - `clean_raw_types(df: pd.DataFrame) -> pd.DataFrame` — parses the raw string-encoded
    columns (`term`, `int_rate`, `revol_util`, `emp_length`) into numeric dtypes in place on
    the returned copy; leaves other columns untouched.
  - `build_preprocessor() -> sklearn.compose.ColumnTransformer` — unfitted transformer:
    median-imputes + standard-scales `NUMERIC_FEATURES`, most-frequent-imputes +
    one-hot-encodes `CATEGORICAL_FEATURES` (with `handle_unknown="ignore"`). Fit once on the
    reference window in Task 5/6 and reused unchanged for every later evaluation — never
    refit per window (design doc §4).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_features.py
import numpy as np

from model.features import select_features, clean_raw_types, build_preprocessor


def test_select_features_keeps_only_allow_listed_columns():
    df = pd.DataFrame({
        "loan_amnt": [1000],
        "term": [" 36 months"],
        "int_rate": ["10.65%"],
        "grade": ["B"],
        "sub_grade": ["B2"],
        "emp_length": ["3 years"],
        "home_ownership": ["RENT"],
        "annual_inc": [50000.0],
        "verification_status": ["Verified"],
        "dti": [15.0],
        "revol_util": ["30%"],
        "fico_range_low": [700],
        "fico_range_high": [704],
        "purpose": ["debt_consolidation"],
        "default_flag": [0],
        "recoveries": [123.0],  # post-origination leak column — must be dropped
        "total_pymnt": [5000.0],  # post-origination leak column — must be dropped
    })
    result = select_features(df)
    assert "recoveries" not in result.columns
    assert "total_pymnt" not in result.columns
    assert "default_flag" in result.columns
    assert set(result.columns) - {"default_flag"} == set(
        ["loan_amnt", "term", "int_rate", "grade", "sub_grade", "emp_length",
         "home_ownership", "annual_inc", "verification_status", "dti", "revol_util",
         "fico_range_low", "fico_range_high", "purpose"]
    )


def test_clean_raw_types_parses_term_rate_util_emp_length():
    df = pd.DataFrame({
        "term": [" 36 months", " 60 months"],
        "int_rate": ["10.65%", "7.2%"],
        "revol_util": ["30%", None],
        "emp_length": ["3 years", "10+ years"],
    })
    result = clean_raw_types(df)
    assert list(result["term"]) == [36, 60]
    assert result["int_rate"].iloc[0] == pytest.approx(10.65)
    assert result["revol_util"].iloc[0] == pytest.approx(30.0)
    assert np.isnan(result["revol_util"].iloc[1])
    assert result["emp_length"].iloc[0] == 3
    assert result["emp_length"].iloc[1] == 10


def test_build_preprocessor_fits_and_transforms():
    df = pd.DataFrame({
        "loan_amnt": [1000, 2000, 3000],
        "term": [36, 60, 36],
        "int_rate": [10.0, 12.0, 9.0],
        "emp_length": [1, 5, 10],
        "annual_inc": [40000.0, 60000.0, 80000.0],
        "dti": [10.0, 20.0, 15.0],
        "revol_util": [30.0, 40.0, 50.0],
        "fico_range_low": [700, 710, 690],
        "fico_range_high": [704, 714, 694],
        "grade": ["A", "B", "C"],
        "sub_grade": ["A1", "B2", "C3"],
        "home_ownership": ["RENT", "OWN", "MORTGAGE"],
        "verification_status": ["Verified", "Not Verified", "Verified"],
        "purpose": ["debt_consolidation", "credit_card", "other"],
    })
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3
```

Add `import pytest` at the top of `tests/test_features.py` if not already present from Task 2
(it isn't — add it now).

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_features.py -v
```

Expected: the three new tests FAIL with `ImportError`.

- [ ] **Step 3: Implement `select_features`, `clean_raw_types`, `build_preprocessor`**

```python
# append to model/features.py
import re

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

_EMP_LENGTH_PATTERN = re.compile(r"(\d+)")


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the allow-listed raw columns, plus the target column if present.

    This is intentionally a whitelist, not a blacklist of known-leaky columns: an
    unrecognized column (including one added by a future schema change) is excluded by
    default rather than silently flowing through into the model.
    """
    keep = [c for c in FEATURE_COLUMNS if c in df.columns]
    if TARGET_COLUMN in df.columns:
        keep = keep + [TARGET_COLUMN]
    return df[keep].copy()


def clean_raw_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse string-encoded raw columns into numeric dtypes."""
    out = df.copy()
    if "term" in out.columns:
        out["term"] = out["term"].astype(str).str.extract(r"(\d+)").astype(float)
    if "int_rate" in out.columns:
        out["int_rate"] = out["int_rate"].astype(str).str.rstrip("%").astype(float)
    if "revol_util" in out.columns:
        out["revol_util"] = out["revol_util"].astype(str).str.rstrip("%")
        out["revol_util"] = pd.to_numeric(out["revol_util"], errors="coerce")
    if "emp_length" in out.columns:
        out["emp_length"] = out["emp_length"].astype(str).str.extract(_EMP_LENGTH_PATTERN)
        out["emp_length"] = pd.to_numeric(out["emp_length"], errors="coerce")
    return out


def build_preprocessor() -> ColumnTransformer:
    """Build the (unfitted) shared preprocessing pipeline.

    Fit this exactly once, on the reference (pre-2019) window, and reuse the fitted
    instance for every later evaluation window and for the Phase 2 autoencoder. Refitting
    per window would let drift leak into preprocessing itself.
    """
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_features.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/IdeaProjects/mlops
git add model/features.py tests/test_features.py
git commit -m "Add feature allow-list selection, type cleaning, and shared preprocessor"
```

---

### Task 4: Temporal split helper

**Files:**
- Create: `data/prepare.py`
- Test: `tests/test_prepare.py`

**Interfaces:**
- Consumes: `config.REFERENCE_END`, `config.EVAL_START`, `config.EVAL_END`.
- Produces: `temporal_split(df: pd.DataFrame, date_col: str = "issue_d") -> tuple[pd.DataFrame, pd.DataFrame]`
  — returns `(reference_df, eval_df)`. `date_col` is parsed as `"%b-%Y"` (e.g. `"Dec-2018"`,
  matching the raw file's `issue_d` format) if not already a datetime dtype. `reference_df`
  contains rows with `issue_d < REFERENCE_END`; `eval_df` contains rows with
  `EVAL_START <= issue_d < EVAL_END`. Rows outside both ranges (e.g. issued in 2021+) are
  dropped from both. Task 6 (`train.py`) consumes this directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prepare.py
import pandas as pd

from data.prepare import temporal_split


def test_temporal_split_separates_reference_and_eval_windows():
    df = pd.DataFrame({
        "issue_d": ["Dec-2018", "Jan-2019", "Jun-2020", "Dec-2020", "Feb-2021"],
        "loan_amnt": [1, 2, 3, 4, 5],
    })
    reference, eval_ = temporal_split(df)
    assert list(reference["loan_amnt"]) == [1]
    assert list(eval_["loan_amnt"]) == [2, 3, 4]


def test_temporal_split_handles_already_parsed_datetime_column():
    df = pd.DataFrame({
        "issue_d": pd.to_datetime(["2018-11-01", "2019-03-01"]),
        "loan_amnt": [1, 2],
    })
    reference, eval_ = temporal_split(df)
    assert list(reference["loan_amnt"]) == [1]
    assert list(eval_["loan_amnt"]) == [2]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_prepare.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `temporal_split`**

```python
# data/prepare.py
"""Load the raw Lending Club CSV, apply the temporal split, and write partitioned parquet."""
import pandas as pd

from config import EVAL_END, EVAL_START, REFERENCE_END


def temporal_split(df: pd.DataFrame, date_col: str = "issue_d") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split loans into the pre-2019 reference window and the 2019-2020 eval window.

    Never shuffle across this boundary — the whole point of the project is measuring
    how a model trained only on the reference window degrades on the eval window.
    """
    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        out[date_col] = pd.to_datetime(out[date_col], format="%b-%Y")

    reference = out[out[date_col] < REFERENCE_END].copy()
    eval_ = out[(out[date_col] >= EVAL_START) & (out[date_col] < EVAL_END)].copy()
    return reference, eval_
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_prepare.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/IdeaProjects/mlops
git add data/prepare.py tests/test_prepare.py
git commit -m "Add temporal split helper for reference/eval windows"
```

---

### Task 5: Full prepare pipeline, run against real data

**Files:**
- Modify: `data/prepare.py`
- Test: `tests/test_prepare.py`

**Interfaces:**
- Consumes: `derive_target`, `select_features`, `clean_raw_types` from `model.features`;
  `temporal_split` from Task 4; `config.RAW_CSV_PATH`, `config.PROCESSED_DIR`.
- Produces: `run_prepare(raw_csv_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]`
  — loads the raw CSV (or an injected path, for testability), prints `df.columns`,
  `df.dtypes`, and `df["loan_status"].value_counts()` before any schema assumption (spec §3
  first-implementation-step requirement), applies `clean_raw_types` → `derive_target` →
  `select_features` → `temporal_split`, writes `reference.parquet` and `eval.parquet` under
  `PROCESSED_DIR`, and returns `(reference_df, eval_df)`. Task 6 (`train.py`) reads the parquet
  files this writes.

- [ ] **Step 1: Write the failing integration test using a small synthetic CSV**

```python
# append to tests/test_prepare.py
from pathlib import Path

from data.prepare import run_prepare


def test_run_prepare_end_to_end(tmp_path: Path):
    csv_path = tmp_path / "fixture.csv"
    csv_path.write_text(
        "loan_amnt,term,int_rate,grade,sub_grade,emp_length,home_ownership,annual_inc,"
        "verification_status,dti,revol_util,fico_range_low,fico_range_high,purpose,"
        "issue_d,loan_status,recoveries\n"
        "1000, 36 months,10.65%,B,B2,3 years,RENT,50000,Verified,15.0,30%,700,704,"
        "debt_consolidation,Dec-2018,Fully Paid,0.0\n"
        "2000, 60 months,15.0%,D,D1,10+ years,MORTGAGE,70000,Not Verified,20.0,50%,650,654,"
        "credit_card,Jun-2019,Charged Off,500.0\n"
        "3000, 36 months,9.0%,A,A1,< 1 year,OWN,90000,Verified,5.0,10%,720,724,"
        "other,Mar-2018,Current,0.0\n"
    )
    reference_df, eval_df = run_prepare(raw_csv_path=csv_path)

    assert len(reference_df) == 1  # only the Dec-2018 Fully Paid row (Current row dropped)
    assert len(eval_df) == 1  # only the Jun-2019 Charged Off row
    assert "recoveries" not in reference_df.columns
    assert "default_flag" in reference_df.columns

    processed_files = list((csv_path.parent).glob("*.parquet"))  # sanity placeholder, replaced below
```

Replace the last line — the real test should check the actual configured output location, but
since `PROCESSED_DIR` is a module-level constant from `config.py` we don't want the test
writing into the real `var/processed/` directory. Rewrite the test to pass an explicit output
directory instead:

```python
# tests/test_prepare.py — final version of this test
def test_run_prepare_end_to_end(tmp_path: Path):
    csv_path = tmp_path / "fixture.csv"
    csv_path.write_text(
        "loan_amnt,term,int_rate,grade,sub_grade,emp_length,home_ownership,annual_inc,"
        "verification_status,dti,revol_util,fico_range_low,fico_range_high,purpose,"
        "issue_d,loan_status,recoveries\n"
        "1000, 36 months,10.65%,B,B2,3 years,RENT,50000,Verified,15.0,30%,700,704,"
        "debt_consolidation,Dec-2018,Fully Paid,0.0\n"
        "2000, 60 months,15.0%,D,D1,10+ years,MORTGAGE,70000,Not Verified,20.0,50%,650,654,"
        "credit_card,Jun-2019,Charged Off,500.0\n"
        "3000, 36 months,9.0%,A,A1,< 1 year,OWN,90000,Verified,5.0,10%,720,724,"
        "other,Mar-2018,Current,0.0\n"
    )
    output_dir = tmp_path / "processed"

    reference_df, eval_df = run_prepare(raw_csv_path=csv_path, output_dir=output_dir)

    assert len(reference_df) == 1
    assert len(eval_df) == 1
    assert "recoveries" not in reference_df.columns
    assert "default_flag" in reference_df.columns
    assert (output_dir / "reference.parquet").exists()
    assert (output_dir / "eval.parquet").exists()
```

Remove the earlier draft version of this test — only the final version above should remain in
the file.

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_prepare.py::test_run_prepare_end_to_end -v
```

Expected: FAIL — `run_prepare()` doesn't exist yet.

- [ ] **Step 3: Implement `run_prepare`**

```python
# append to data/prepare.py
from pathlib import Path

from config import PROCESSED_DIR, RAW_CSV_PATH
from model.features import clean_raw_types, derive_target, select_features


def run_prepare(
    raw_csv_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, split, and persist the reference/eval parquet files.

    Backtest note: everything downstream of this function is a simulation on historical,
    already-resolved loan outcomes — not a live scoring pipeline.
    """
    raw_csv_path = raw_csv_path or RAW_CSV_PATH
    output_dir = output_dir or PROCESSED_DIR

    raw = pd.read_csv(raw_csv_path, low_memory=False)

    print(f"Loaded {len(raw)} rows, {len(raw.columns)} columns from {raw_csv_path}")
    print(raw.dtypes)
    print(raw["loan_status"].value_counts())

    cleaned = clean_raw_types(raw)
    labeled = derive_target(cleaned)
    selected = select_features(labeled)
    selected["issue_d"] = raw.loc[selected.index, "issue_d"]

    reference_df, eval_df = temporal_split(selected)

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_df.to_parquet(output_dir / "reference.parquet", index=False)
    eval_df.to_parquet(output_dir / "eval.parquet", index=False)

    return reference_df, eval_df


if __name__ == "__main__":
    run_prepare()
```

Note the `selected["issue_d"] = raw.loc[selected.index, "issue_d"]` line: `select_features`
only keeps `FEATURE_COLUMNS` + target, but `temporal_split` needs `issue_d`, so it's
reattached here from the original frame by index before splitting, then parquet retains it.
`issue_d` is origination metadata, not a leaky post-origination field, so this is safe.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_prepare.py -v
```

Expected: all tests in the file pass (3 total: the 2 temporal_split tests from Task 4, plus
this one).

- [ ] **Step 5: Run the real pipeline against the downloaded CSV**

```bash
cd ~/IdeaProjects/mlops
uv run python -m data.prepare
```

Expected: prints the real column list, dtypes, and `loan_status` value counts (matches what
was already confirmed manually: Fully Paid 1,059,885 / Current 871,037 / Charged Off 266,014 /
Default 39, plus Late/Grace Period buckets), then writes `var/processed/reference.parquet` and
`var/processed/eval.parquet`. Inspect the row counts printed and sanity check that
`reference.parquet` has roughly 1.1-1.2M rows (pre-2019 resolved loans) and `eval.parquet` has
a smaller count (2019-2020 resolved loans only — most of that window is still "Current" and
gets dropped, which is expected and correct).

- [ ] **Step 6: Commit**

```bash
cd ~/IdeaProjects/mlops
git add data/prepare.py tests/test_prepare.py
git commit -m "Implement full prepare pipeline and run against real Lending Club data"
```

---

### Task 6: Baseline XGBoost model, trained on reference window

**Files:**
- Create: `model/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `build_preprocessor` from `model.features`; `config.NUMERIC_FEATURES`,
  `config.CATEGORICAL_FEATURES`, `config.TARGET_COLUMN`, `config.MLFLOW_TRACKING_URI`.
- Produces:
  - `train_baseline(reference_df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]`
    — splits `reference_df` by time (last 10% of rows by `issue_d`, sorted, as a temporal
    holdout test slice — not a random split, so the "held-out pre-2019 test slice" in the
    spec's Phase 1 acceptance criteria is genuinely unseen-in-time data), fits a
    `sklearn.pipeline.Pipeline` combining `build_preprocessor()` and an
    `xgboost.XGBClassifier`, and returns `(fitted_pipeline, train_slice_df, test_slice_df)`.
  - `evaluate_auc(pipeline: Pipeline, df: pd.DataFrame) -> float` — computes ROC-AUC of
    `pipeline` on `df` using `config.TARGET_COLUMN` as ground truth. Task 7 reuses this for
    per-quarter evaluation.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train.py
import numpy as np
import pandas as pd

from model.train import evaluate_auc, train_baseline


def _synthetic_reference_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=n, freq="7D")
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


def test_train_baseline_returns_fitted_pipeline_and_splits():
    df = _synthetic_reference_df()
    pipeline, train_slice, test_slice = train_baseline(df)
    assert len(train_slice) + len(test_slice) == len(df)
    assert len(test_slice) < len(train_slice)
    assert test_slice["issue_d"].min() >= train_slice["issue_d"].max()


def test_evaluate_auc_returns_value_between_0_and_1():
    df = _synthetic_reference_df()
    pipeline, _, test_slice = train_baseline(df)
    auc = evaluate_auc(pipeline, test_slice)
    assert 0.0 <= auc <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_train.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `train_baseline` and `evaluate_auc`**

```python
# model/train.py
"""Train the champion XGBoost model on the pre-2019 reference window."""
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import TARGET_COLUMN
from model.features import build_preprocessor


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
    feature_cols = [c for c in reference_df.columns if c not in (TARGET_COLUMN, "issue_d")]
    pipeline.fit(train_slice[feature_cols], train_slice[TARGET_COLUMN])

    return pipeline, train_slice, test_slice


def evaluate_auc(pipeline: Pipeline, df: pd.DataFrame) -> float:
    """ROC-AUC of pipeline on df. Backtest metric only — computed on historical, resolved
    loan outcomes, never on live predictions."""
    feature_cols = [c for c in df.columns if c not in (TARGET_COLUMN, "issue_d")]
    predictions = pipeline.predict_proba(df[feature_cols])[:, 1]
    return roc_auc_score(df[TARGET_COLUMN], predictions)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_train.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/IdeaProjects/mlops
git add model/train.py tests/test_train.py
git commit -m "Add baseline XGBoost training with temporal holdout and AUC evaluation"
```

---

### Task 7: Quarterly AUC report, MLflow logging, and the decline chart

**Files:**
- Modify: `model/train.py`
- Create: `reports/generate_phase1_report.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `train_baseline`, `evaluate_auc` from Task 6; `config.MLFLOW_TRACKING_URI`,
  `config.PROCESSED_DIR`, `config.REPORTS_DIR`.
- Produces:
  - `quarterly_auc(pipeline: Pipeline, eval_df: pd.DataFrame) -> pd.DataFrame` — groups
    `eval_df` by calendar quarter of `issue_d`, returns a DataFrame with columns `quarter`
    (e.g. `"2019Q1"`) and `auc`, evaluating the **unmodified** pipeline against each quarter
    without retraining.
  - `reports/generate_phase1_report.py` — standalone script: loads the parquet files, calls
    `train_baseline`, logs the run to MLflow (params: model type, n_estimators, max_depth;
    metrics: reference test AUC, per-quarter AUC), calls `quarterly_auc`, prints a labeled
    summary table, and saves `reports/phase1_auc_decline.png`.

- [ ] **Step 1: Write the failing test for `quarterly_auc`**

```python
# append to tests/test_train.py
from model.train import quarterly_auc


def test_quarterly_auc_groups_by_calendar_quarter():
    df = _synthetic_reference_df(n=400)
    pipeline, _, _ = train_baseline(df)
    eval_df = df.copy()
    eval_df["issue_d"] = pd.date_range("2019-01-01", periods=len(df), freq="7D")
    result = quarterly_auc(pipeline, eval_df)
    assert list(result.columns) == ["quarter", "auc"]
    assert result["quarter"].is_unique
    assert result["auc"].between(0, 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_train.py::test_quarterly_auc_groups_by_calendar_quarter -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `quarterly_auc`**

```python
# append to model/train.py
def quarterly_auc(pipeline: Pipeline, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the unmodified pipeline against each calendar quarter of eval_df.

    This is the core Phase 1 measurement: the model is never retrained here, so a
    declining trend across quarters is the drift problem made concrete (spec Phase 1
    acceptance criteria) — a backtest result on historical, resolved loan outcomes.
    """
    df = eval_df.copy()
    df["quarter"] = pd.PeriodIndex(df["issue_d"], freq="Q").astype(str)

    rows = []
    for quarter, group in df.groupby("quarter"):
        if group[TARGET_COLUMN].nunique() < 2:
            continue  # AUC undefined without both classes present
        rows.append({"quarter": quarter, "auc": evaluate_auc(pipeline, group)})

    return pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_train.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Write the report script**

```python
# reports/generate_phase1_report.py
"""Phase 1 deliverable: baseline AUC on a held-out pre-2019 slice, and the quarterly AUC
decline through 2019-2020 without retraining. All numbers below are a backtest on public
historical Lending Club data — not a live production evaluation."""
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from config import MLFLOW_TRACKING_URI, PROCESSED_DIR, REPORTS_DIR
from model.train import evaluate_auc, quarterly_auc, train_baseline


def main() -> None:
    reference_df = pd.read_parquet(PROCESSED_DIR / "reference.parquet")
    eval_df = pd.read_parquet(PROCESSED_DIR / "eval.parquet")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("creditpulse-phase1-baseline")

    with mlflow.start_run(run_name="baseline-champion"):
        pipeline, _, test_slice = train_baseline(reference_df)
        holdout_auc = evaluate_auc(pipeline, test_slice)
        quarterly = quarterly_auc(pipeline, eval_df)

        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 4)
        mlflow.log_metric("reference_holdout_auc", holdout_auc)
        for _, row in quarterly.iterrows():
            mlflow.log_metric(f"auc_{row['quarter']}", row["auc"])

    print("=== BACKTEST ON HISTORICAL DATA — not a live evaluation ===")
    print(f"Held-out pre-2019 test slice AUC: {holdout_auc:.4f}")
    print("\nQuarterly AUC on 2019-2020 data, static model (no retraining):")
    print(quarterly.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(quarterly["quarter"], quarterly["auc"], marker="o")
    ax.axhline(holdout_auc, color="gray", linestyle="--", label=f"pre-2019 holdout AUC ({holdout_auc:.3f})")
    ax.set_ylabel("AUC")
    ax.set_xlabel("Quarter")
    ax.set_title("Static Model AUC Decline — Backtest on Historical Lending Club Data")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "phase1_auc_decline.png", dpi=150)
    print(f"\nChart saved to {REPORTS_DIR / 'phase1_auc_decline.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the report against the real prepared data**

```bash
cd ~/IdeaProjects/mlops
uv run python -m reports.generate_phase1_report
```

Expected: prints held-out pre-2019 AUC (spec acceptance criterion: should be > 0.65) and the
quarterly AUC table for 2019-2020, saves `reports/phase1_auc_decline.png`. **Stop here and
inspect the numbers before proceeding to Phase 2** — the quarterly AUC should show a visible
decline heading into 2020. If it doesn't, this is the acceptance-gate failure described in the
design doc §8: diagnose before building anything further on top of it (check: is `issue_d`
parsing correctly, is the held-out slice actually chronologically later, is the feature set
too weak, etc.) rather than proceeding.

- [ ] **Step 7: Commit**

```bash
cd ~/IdeaProjects/mlops
git add model/train.py tests/test_train.py reports/generate_phase1_report.py reports/phase1_auc_decline.png
git commit -m "Add quarterly AUC evaluation, MLflow logging, and Phase 1 report"
```

---

## Self-review notes

- **Spec coverage:** `data/prepare.py` (load/clean/target/split/parquet) ✓; `model/train.py`
  baseline XGBoost on pre-2019 ✓; held-out pre-2019 test AUC reported ✓; per-quarter
  2019/2020 AUC without retraining ✓; non-terminal status exclusion documented and tested ✓;
  leak-safe feature selection ✓; MLflow SQLite backend wired for later phases ✓; backtest
  labeling on every printed/plotted result ✓.
- **Placeholder scan:** none remaining — the two draft/placeholder test bodies in Task 5 were
  explicitly superseded by final versions with an instruction to remove the draft.
- **Type consistency:** `TARGET_COLUMN` (from `config.py`) used consistently across
  `features.py`, `prepare.py`, `train.py`. `train_baseline` returns
  `(Pipeline, pd.DataFrame, pd.DataFrame)` in Task 6 and is consumed with that exact shape in
  Task 7. `evaluate_auc(pipeline, df)` signature matches its Task 6 definition everywhere it's
  called in Task 7.
