import numpy as np
import pandas as pd

from model.train import evaluate_auc, quarterly_auc, train_baseline


def _synthetic_reference_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=n, freq="7D")
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


def test_train_baseline_returns_fitted_pipeline_and_splits():
    df = _synthetic_reference_df()
    pipeline, train_slice, test_slice = train_baseline(df)
    assert len(train_slice) + len(test_slice) == len(df)
    assert len(test_slice) < len(train_slice)
    assert test_slice["issue_d"].min() >= train_slice["issue_d"].max()


def test_evaluate_auc_returns_value_between_0_and_1():
    df = _synthetic_reference_df()
    pipeline, _, test_slice = train_baseline(df)
    auc = evaluate_auc(pipeline, test_slice)
    assert 0.0 <= auc <= 1.0


def test_quarterly_auc_groups_by_calendar_quarter():
    df = _synthetic_reference_df(n=400)
    pipeline, _, _ = train_baseline(df)
    eval_df = df.copy()
    eval_df["issue_d"] = pd.date_range("2019-01-01", periods=len(df), freq="7D")
    result = quarterly_auc(pipeline, eval_df)
    assert list(result.columns) == ["quarter", "auc"]
    assert result["quarter"].is_unique
    assert result["auc"].between(0, 1).all()
