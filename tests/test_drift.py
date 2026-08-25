import numpy as np
import pandas as pd

from drift.autoencoder import reconstruction_error, train_autoencoder
from drift.statistical import (
    compute_feature_drift,
    fit_statistical_reference,
    ks_statistic,
    psi_categorical,
    psi_numeric,
)


def _reference_df(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "income": rng.normal(50000, 10000, size=n),
        "grade": rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2]),
    })


def test_psi_numeric_near_zero_for_identical_distribution():
    reference = fit_statistical_reference(_reference_df(), ["income"], [], n_bins=10)
    same_dist = pd.Series(np.random.default_rng(1).normal(50000, 10000, size=1000))
    assert psi_numeric(reference, "income", same_dist) < 0.05


def test_psi_numeric_large_for_shifted_distribution():
    reference = fit_statistical_reference(_reference_df(), ["income"], [], n_bins=10)
    shifted = pd.Series(np.random.default_rng(1).normal(20000, 10000, size=1000))
    assert psi_numeric(reference, "income", shifted) > 0.25


def test_psi_categorical_flags_unseen_category():
    reference = fit_statistical_reference(_reference_df(), [], ["grade"], n_bins=10)
    comparison = pd.Series(["D"] * 500 + ["A"] * 500)  # "D" never seen in reference
    assert psi_categorical(reference, "grade", comparison) > 0.25


def test_ks_statistic_large_for_shifted_distribution():
    reference = fit_statistical_reference(_reference_df(), ["income"], [], n_bins=10)
    shifted = pd.Series(np.random.default_rng(1).normal(20000, 10000, size=1000))
    assert ks_statistic(reference, "income", shifted) > 0.3


def test_compute_feature_drift_returns_all_features_sorted_by_psi():
    reference = fit_statistical_reference(_reference_df(), ["income"], ["grade"], n_bins=10)
    comparison_df = pd.DataFrame({
        "income": np.random.default_rng(1).normal(50000, 10000, size=1000),
        "grade": pd.Series(np.random.default_rng(1).choice(["A", "B", "C"], size=1000, p=[0.5, 0.3, 0.2])),
    })
    result = compute_feature_drift(reference, comparison_df, ["income"], ["grade"])
    assert set(result["feature"]) == {"income", "grade"}
    assert list(result.columns) == ["feature", "type", "psi", "ks_stat"]
    assert result["psi"].is_monotonic_decreasing


def test_reconstruction_error_lower_on_reference_than_random_noise():
    rng = np.random.default_rng(0)
    X_reference = rng.normal(0, 1, size=(500, 10)).astype(np.float32)
    model = train_autoencoder(X_reference, epochs=20)

    reference_errors = reconstruction_error(model, X_reference)
    noise_errors = reconstruction_error(model, rng.normal(10, 5, size=(500, 10)).astype(np.float32))

    assert reference_errors.mean() < noise_errors.mean()


def test_reconstruction_error_returns_one_value_per_row():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(50, 6)).astype(np.float32)
    model = train_autoencoder(X, epochs=5)
    errors = reconstruction_error(model, X)
    assert errors.shape == (50,)
