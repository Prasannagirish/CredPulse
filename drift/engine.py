"""Combines statistical and autoencoder drift signals into a single DriftReport."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    AE_BREACH_FRACTION,
    AE_ERROR_PERCENTILE,
    AE_WARNING_FRACTION,
    PSI_BREACH,
    PSI_WARNING,
)
from drift.autoencoder import DriftAutoencoder, reconstruction_error
from drift.statistical import StatisticalDriftReference, compute_feature_drift

_STATUS_ORDER = {"ok": 0, "warning": 1, "breach": 2}


@dataclass
class DriftReport:
    window: str
    status: str
    feature_drift: pd.DataFrame
    autoencoder_fraction_above_threshold: float
    autoencoder_status: str
    statistical_status: str


def _statistical_status(feature_drift: pd.DataFrame) -> str:
    if (feature_drift["psi"] > PSI_BREACH).any():
        return "breach"
    if (feature_drift["psi"] > PSI_WARNING).any():
        return "warning"
    return "ok"


def _autoencoder_status(fraction_above_threshold: float) -> str:
    if fraction_above_threshold > AE_BREACH_FRACTION:
        return "breach"
    if fraction_above_threshold > AE_WARNING_FRACTION:
        return "warning"
    return "ok"


def evaluate_drift(
    window: str,
    statistical_reference: StatisticalDriftReference,
    autoencoder: DriftAutoencoder,
    reference_errors: np.ndarray,
    comparison_df: pd.DataFrame,
    comparison_X: np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str],
) -> DriftReport:
    feature_drift = compute_feature_drift(
        statistical_reference, comparison_df, numeric_features, categorical_features
    )
    statistical_status = _statistical_status(feature_drift)

    threshold = np.percentile(reference_errors, AE_ERROR_PERCENTILE * 100)
    comparison_errors = reconstruction_error(autoencoder, comparison_X)
    fraction_above_threshold = float((comparison_errors > threshold).mean())
    autoencoder_status = _autoencoder_status(fraction_above_threshold)

    overall_status = max([statistical_status, autoencoder_status], key=lambda s: _STATUS_ORDER[s])

    return DriftReport(
        window=window,
        status=overall_status,
        feature_drift=feature_drift,
        autoencoder_fraction_above_threshold=fraction_above_threshold,
        autoencoder_status=autoencoder_status,
        statistical_status=statistical_status,
    )
