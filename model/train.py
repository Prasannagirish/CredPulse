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
