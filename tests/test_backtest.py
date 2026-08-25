import numpy as np
import pandas as pd

from backtest.arms import ArmState, maybe_retrain_monitored, maybe_retrain_scheduled
from drift.engine import DriftReport
from lifecycle.retrain import retrain_challenger


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


def _dummy_report(status: str) -> DriftReport:
    return DriftReport(
        window="test", status=status, feature_drift=pd.DataFrame(),
        autoencoder_fraction_above_threshold=0.0, autoencoder_status=status, statistical_status=status,
    )


def test_maybe_retrain_scheduled_retrains_after_interval_elapses():
    df = _synthetic_df(600, "2017-01-01")
    initial = retrain_challenger(df)
    arm = ArmState(name="scheduled", pipeline=initial, last_retrain_date=pd.Timestamp("2019-01-01"))

    still_waiting = maybe_retrain_scheduled(arm, pd.Timestamp("2019-02-01"), df)
    assert still_waiting.retrain_count == 0

    retrained = maybe_retrain_scheduled(arm, pd.Timestamp("2019-04-01"), df)
    assert retrained.retrain_count == 1
    assert retrained.last_retrain_date == pd.Timestamp("2019-04-01")


def test_maybe_retrain_monitored_only_retrains_on_breach_after_cooldown():
    df = _synthetic_df(600, "2017-01-01")
    initial = retrain_challenger(df)
    arm = ArmState(name="monitored", pipeline=initial, last_retrain_date=pd.Timestamp("2019-01-01"))

    ok_result = maybe_retrain_monitored(arm, pd.Timestamp("2019-02-01"), df, _dummy_report("ok"))
    assert ok_result.retrain_count == 0

    too_soon = maybe_retrain_monitored(arm, pd.Timestamp("2019-02-01"), df, _dummy_report("breach"))
    assert too_soon.retrain_count == 0  # cooldown not elapsed yet

    retrained = maybe_retrain_monitored(arm, pd.Timestamp("2019-04-01"), df, _dummy_report("breach"))
    assert retrained.retrain_count == 1


from backtest.impact import dollar_impact


def test_dollar_impact_accepts_fixed_fraction_and_computes_loss_per_10k():
    df = _synthetic_df(500, "2019-01-01")
    pipeline = retrain_challenger(df)

    result = dollar_impact(pipeline, df, approval_rate=0.5)

    assert abs(result["n_accepted"] - 250) <= 5
    assert result["n_defaults"] >= 0
    assert result["dollar_loss"] >= 0.0
    assert result["dollar_loss_per_10k_loans"] >= 0.0
