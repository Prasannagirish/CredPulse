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
