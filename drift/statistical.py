"""Statistical drift signals: PSI and KS-test, computed against a frozen reference distribution."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

_EPSILON = 1e-4  # smoothing to avoid log(0) / division by zero in PSI
_UNSEEN_CATEGORY = "__UNSEEN__"


@dataclass
class StatisticalDriftReference:
    """Frozen reference distribution for PSI/KS comparisons.

    Bin edges and reference values are computed once from the reference window and reused
    for every comparison window — recomputing them per window (e.g. from quantiles of the
    comparison data itself) is a common bug that makes PSI read ~0 forever, since every
    window would be compared against its own distribution.
    """
    numeric_bin_edges: dict[str, np.ndarray]
    numeric_reference_values: dict[str, np.ndarray]
    categorical_reference_frequencies: dict[str, pd.Series]


def fit_statistical_reference(
    reference_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    n_bins: int = 10,
) -> StatisticalDriftReference:
    numeric_bin_edges: dict[str, np.ndarray] = {}
    numeric_reference_values: dict[str, np.ndarray] = {}
    for feature in numeric_features:
        values = reference_df[feature].dropna().to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        numeric_bin_edges[feature] = edges
        numeric_reference_values[feature] = values

    categorical_reference_frequencies = {
        feature: reference_df[feature].value_counts(normalize=True)
        for feature in categorical_features
    }

    return StatisticalDriftReference(
        numeric_bin_edges=numeric_bin_edges,
        numeric_reference_values=numeric_reference_values,
        categorical_reference_frequencies=categorical_reference_frequencies,
    )


def _psi_from_proportions(ref_props: np.ndarray, comp_props: np.ndarray) -> float:
    ref_props = np.clip(ref_props, _EPSILON, None)
    comp_props = np.clip(comp_props, _EPSILON, None)
    return float(np.sum((comp_props - ref_props) * np.log(comp_props / ref_props)))


def psi_numeric(reference: StatisticalDriftReference, feature: str, comparison: pd.Series) -> float:
    edges = reference.numeric_bin_edges[feature]
    ref_values = reference.numeric_reference_values[feature]
    ref_counts, _ = np.histogram(ref_values, bins=edges)
    comp_counts, _ = np.histogram(comparison.dropna().to_numpy(), bins=edges)
    ref_props = ref_counts / ref_counts.sum()
    comp_props = comp_counts / max(comp_counts.sum(), 1)
    return _psi_from_proportions(ref_props, comp_props)


def psi_categorical(reference: StatisticalDriftReference, feature: str, comparison: pd.Series) -> float:
    ref_freq = reference.categorical_reference_frequencies[feature]
    categories = list(ref_freq.index) + [_UNSEEN_CATEGORY]
    comp_mapped = comparison.where(comparison.isin(ref_freq.index), _UNSEEN_CATEGORY)
    comp_freq = comp_mapped.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    ref_props = ref_freq.reindex(categories, fill_value=0.0).to_numpy()
    return _psi_from_proportions(ref_props, comp_freq.to_numpy())


def ks_statistic(reference: StatisticalDriftReference, feature: str, comparison: pd.Series) -> float:
    ref_values = reference.numeric_reference_values[feature]
    comp_values = comparison.dropna().to_numpy()
    return float(ks_2samp(ref_values, comp_values).statistic)


def compute_feature_drift(
    reference: StatisticalDriftReference,
    comparison_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    rows = []
    for feature in numeric_features:
        rows.append({
            "feature": feature,
            "type": "numeric",
            "psi": psi_numeric(reference, feature, comparison_df[feature]),
            "ks_stat": ks_statistic(reference, feature, comparison_df[feature]),
        })
    for feature in categorical_features:
        rows.append({
            "feature": feature,
            "type": "categorical",
            "psi": psi_categorical(reference, feature, comparison_df[feature]),
            "ks_stat": float("nan"),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
