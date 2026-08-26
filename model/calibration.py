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
