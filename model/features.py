"""Feature allow-list, target derivation, and the shared preprocessing pipeline."""
import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN

_EMP_LENGTH_PATTERN = re.compile(r"(\d+)")

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
