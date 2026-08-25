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
