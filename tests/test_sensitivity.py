import numpy as np
import pandas as pd

from backtest.sensitivity import approval_rate_sensitivity
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


def test_approval_rate_sensitivity_sweeps_all_rates_and_accepts_more_at_higher_rates():
    df = _synthetic_df(n=500)
    pipeline = retrain_challenger(df)

    result = approval_rate_sensitivity(pipeline, df, approval_rates=[0.5, 0.8])

    assert list(result["approval_rate"]) == [0.5, 0.8]
    assert (result["n_accepted"] > 0).all()
    assert result.iloc[1]["n_accepted"] >= result.iloc[0]["n_accepted"]
