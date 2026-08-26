import numpy as np
import pandas as pd

from model.baseline_woe import (
    evaluate_logistic_auc,
    fit_woe_bins,
    train_logistic_baseline,
    transform_woe,
)


def _synthetic_df(n: int = 500, start: str = "2017-01-01") -> pd.DataFrame:
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


def test_fit_woe_bins_assigns_lower_woe_to_higher_risk_bin():
    n = 1000
    rng = np.random.default_rng(0)
    int_rate = np.concatenate([np.full(500, 5.0), np.full(500, 25.0)])
    default_flag = np.concatenate([
        rng.choice([0, 1], size=500, p=[0.9, 0.1]),   # low rate -> mostly good (non-default)
        rng.choice([0, 1], size=500, p=[0.3, 0.7]),   # high rate -> mostly bad (default)
    ])
    df = pd.DataFrame({"int_rate": int_rate, "default_flag": default_flag})

    bins = fit_woe_bins(df, numeric_features=["int_rate"], categorical_features=[])

    # WOE = ln(%good/%bad): the low-risk (mostly-good) bin should have a HIGHER WOE than
    # the high-risk (mostly-bad) bin.
    assert bins.numeric_woe["int_rate"][0] > bins.numeric_woe["int_rate"][-1]


def test_transform_woe_handles_unseen_category_gracefully():
    df = _synthetic_df(n=200)
    bins = fit_woe_bins(df)
    test_df = df.copy()
    test_df.loc[test_df.index[0], "grade"] = "ZZ"  # unseen category

    transformed = transform_woe(bins, test_df)

    assert not transformed["grade"].isna().any()


def test_train_logistic_baseline_returns_model_and_bins():
    df = _synthetic_df(n=500)

    model, bins = train_logistic_baseline(df)
    auc = evaluate_logistic_auc(model, bins, df)

    assert 0.0 <= auc <= 1.0
