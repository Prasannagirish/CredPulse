import numpy as np
import pandas as pd
import pytest

import config
from drift.autoencoder import reconstruction_error, train_autoencoder
from drift.statistical import fit_statistical_reference
from mcp_server import server, state
from model.train import build_model_pipeline


def _synthetic_loans(n: int, start: str, id_prefix: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n, freq="3D")
    risk = rng.normal(size=n)
    default_prob = 1 / (1 + np.exp(-risk))
    default_flag = (rng.uniform(size=n) < default_prob).astype(int)
    return pd.DataFrame({
        "id": [f"{id_prefix}{i}" for i in range(n)],
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


@pytest.fixture()
def synthetic_state(monkeypatch):
    # The synthetic eval_df below is far smaller than real data, so no single month could
    # ever reach the real MIN_RELIABLE_DEFAULTS=100 — lower it for this fixture's scale.
    monkeypatch.setattr(config, "MIN_RELIABLE_DEFAULTS", 5)

    reference_df = _synthetic_loans(300, "2017-01-01", "REF")
    eval_df = _synthetic_loans(200, "2019-01-01", "EVAL")

    pipeline = build_model_pipeline()
    feature_cols = [c for c in reference_df.columns if c not in config.NON_FEATURE_COLUMNS]
    pipeline.fit(reference_df[feature_cols], reference_df[config.TARGET_COLUMN])

    statistical_reference = fit_statistical_reference(
        reference_df, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES
    )
    reference_X = state.transform_for_drift(pipeline, reference_df)
    autoencoder = train_autoencoder(reference_X, epochs=3, batch_size=64)
    reference_errors = reconstruction_error(autoencoder, reference_X)

    import shap
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])

    fake_state = state.ServerState(
        champion_pipeline=pipeline,
        champion_version="1",
        champion_promoted_at_ms=0,
        reference_df=reference_df,
        eval_df=eval_df,
        statistical_reference=statistical_reference,
        autoencoder=autoencoder,
        reference_errors=reference_errors,
        explainer=explainer,
    )
    monkeypatch.setattr(state, "get_state", lambda force_refresh=False: fake_state)
    return fake_state


def test_get_model_health_impl_returns_populated_health(synthetic_state):
    health = server.get_model_health_impl()
    assert health.champion_version == "1"
    assert 0.0 <= health.recent_batch_auc <= 1.0
    assert health.drift_status in ("ok", "warning", "breach")


def test_get_drift_report_impl_resolves_explicit_month_window(synthetic_state):
    month = synthetic_state.eval_df["issue_d"].dt.to_period("M").astype(str).iloc[0]
    report = server.get_drift_report_impl(window=month)
    assert report.window == month
    assert report.status in ("ok", "warning", "breach")
    assert len(report.top_drifting_features) > 0


def test_get_drift_report_impl_resolves_last_n_days_window(synthetic_state):
    report = server.get_drift_report_impl(window="last_30d")
    assert report.window == "last_30d"
    assert report.status in ("ok", "warning", "breach")


def test_explain_prediction_impl_returns_top_contributions(synthetic_state):
    loan_id = synthetic_state.eval_df["id"].iloc[0]
    explanation = server.explain_prediction_impl(loan_id)
    assert explanation.loan_id == str(loan_id)
    assert 0.0 <= explanation.predicted_default_probability <= 1.0
    assert len(explanation.top_contributions) == 5


def test_explain_prediction_impl_raises_for_unknown_loan_id(synthetic_state):
    try:
        server.explain_prediction_impl("does-not-exist")
        assert False, "expected ValueError"
    except ValueError:
        pass
