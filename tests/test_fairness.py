import numpy as np
import pandas as pd
import pytest

from backtest.fairness import add_income_quartile_column, disparate_impact_ratio, group_approval_rates
from lifecycle.retrain import retrain_challenger


def _synthetic_df(n: int = 500, start: str = "2019-01-01") -> pd.DataFrame:
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


def test_group_approval_rates_breaks_out_by_group():
    df = _synthetic_df(n=500)
    pipeline = retrain_challenger(df)

    rates = group_approval_rates(pipeline, df, group_col="home_ownership")

    assert set(rates["group"]) == {"RENT", "OWN", "MORTGAGE"}
    assert (rates["approval_rate"] >= 0).all() and (rates["approval_rate"] <= 1).all()
    assert rates["n_total"].sum() == len(df)


def test_disparate_impact_ratio_flags_large_gap():
    group_rates = pd.DataFrame({"group": ["A", "B"], "approval_rate": [0.9, 0.5]})

    result = disparate_impact_ratio(group_rates)

    assert result["ratio"] == pytest.approx(0.5 / 0.9)
    assert result["flagged"] is True
    assert result["min_group"] == "B"
    assert result["max_group"] == "A"


def test_disparate_impact_ratio_does_not_flag_similar_rates():
    group_rates = pd.DataFrame({"group": ["A", "B"], "approval_rate": [0.85, 0.82]})

    result = disparate_impact_ratio(group_rates)

    assert result["flagged"] is False


def test_add_income_quartile_column_creates_four_groups_from_real_data():
    df = _synthetic_df(n=400)

    result = add_income_quartile_column(df)

    assert result["income_quartile"].nunique() == 4
    assert len(result) == len(df)
