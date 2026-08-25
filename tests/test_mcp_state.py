import numpy as np
import pandas as pd

from mcp_server.state import latest_reliable_batch, transform_for_drift
from model.train import build_model_pipeline
from config import NON_FEATURE_COLUMNS, TARGET_COLUMN


def _synthetic_df(n: int, start: str) -> pd.DataFrame:
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


def test_latest_reliable_batch_picks_most_recent_month_meeting_threshold():
    df = _synthetic_df(n=300, start="2019-01-01")
    df["default_flag"] = 0
    df.loc[df.index[:150], "default_flag"] = 1  # concentrate defaults in the earlier half

    label, batch = latest_reliable_batch(df, min_defaults=5)

    assert batch["default_flag"].sum() >= 5
    assert (batch["issue_d"].dt.to_period("M").astype(str) == label).all()


def test_latest_reliable_batch_raises_when_nothing_qualifies():
    df = _synthetic_df(n=10, start="2019-01-01")
    df["default_flag"] = 0
    try:
        latest_reliable_batch(df, min_defaults=1000)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_transform_for_drift_returns_dense_float32_array():
    df = _synthetic_df(n=50, start="2019-01-01")
    pipeline = build_model_pipeline()
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    pipeline.fit(df[feature_cols], df[TARGET_COLUMN])

    X = transform_for_drift(pipeline, df)

    assert X.dtype == np.float32
    assert X.shape[0] == len(df)
    assert not hasattr(X, "toarray")  # confirms it's dense, not still sparse
