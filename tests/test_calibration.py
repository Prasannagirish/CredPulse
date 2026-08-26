import numpy as np
import pandas as pd

from model.calibration import expected_calibration_error, reliability_diagram_data


class _FakeProbaModel:
    """Stands in for a fitted Pipeline — ignores X, returns pre-set probabilities. Lets the
    ECE test assert against a known, exact ground truth rather than a real model's
    incidental calibration."""
    def __init__(self, probs: np.ndarray):
        self._probs = probs

    def predict_proba(self, X):
        return np.column_stack([1 - self._probs, self._probs])


def test_reliability_diagram_data_returns_bucketed_dataframe_covering_all_rows():
    n = 500
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 30000, size=n),
        "default_flag": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
    })
    model = _FakeProbaModel(rng.uniform(0, 1, size=n))

    diagram = reliability_diagram_data(model, df, n_bins=5)

    assert set(diagram.columns) == {"bin", "mean_predicted", "observed_rate", "count"}
    assert diagram["count"].sum() == n


def test_expected_calibration_error_near_zero_for_well_calibrated_predictions():
    n = 1000
    rng = np.random.default_rng(0)
    probs = np.full(n, 0.2)
    default_flag = rng.choice([0, 1], size=n, p=[0.8, 0.2])  # observed rate matches predicted
    df = pd.DataFrame({"loan_amnt": np.zeros(n), "default_flag": default_flag})

    ece = expected_calibration_error(_FakeProbaModel(probs), df, n_bins=5)

    assert ece < 0.05
