import numpy as np
import pandas as pd

from lifecycle.evaluate import compare_champion_challenger
from lifecycle.retrain import retrain_challenger, select_recent_window


def _synthetic_df(n: int = 200, start: str = "2019-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n, freq="3D")
    risk = rng.normal(size=n)
    default_prob = 1 / (1 + np.exp(-risk))
    default_flag = (rng.uniform(size=n) < default_prob).astype(int)
    return pd.DataFrame({
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


def test_select_recent_window_keeps_only_trailing_months():
    df = _synthetic_df(n=200, start="2019-01-01")  # spans ~19.7 months at 3-day spacing
    as_of = pd.Timestamp("2020-01-01")
    window = select_recent_window(df, as_of, months=6)
    assert window["issue_d"].min() >= as_of - pd.DateOffset(months=6)
    assert window["issue_d"].max() < as_of
    assert len(window) < len(df)


def test_retrain_challenger_returns_fitted_pipeline_that_predicts_probabilities():
    df = _synthetic_df()
    pipeline = retrain_challenger(df)
    feature_cols = [c for c in df.columns if c not in ("default_flag", "issue_d")]
    predictions = pipeline.predict_proba(df[feature_cols])
    assert predictions.shape == (len(df), 2)


def test_compare_champion_challenger_returns_metrics_for_both():
    train_df = _synthetic_df(n=200, start="2018-01-01")
    later_train_df = _synthetic_df(n=200, start="2019-01-01")
    holdout_df = _synthetic_df(n=100, start="2019-09-01")

    champion = retrain_challenger(train_df)
    challenger = retrain_challenger(later_train_df)

    comparison = compare_champion_challenger(champion, challenger, holdout_df)

    assert 0.0 <= comparison.champion_auc <= 1.0
    assert 0.0 <= comparison.challenger_auc <= 1.0
    assert 0.0 <= comparison.champion_f1 <= 1.0
    assert 0.0 <= comparison.challenger_f1 <= 1.0
    assert 0.0 <= comparison.champion_brier <= 1.0
    assert 0.0 <= comparison.challenger_brier <= 1.0
    assert comparison.challenger_wins == (comparison.challenger_auc > comparison.champion_auc)
