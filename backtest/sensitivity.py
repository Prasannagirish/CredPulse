"""Approval-rate sensitivity: sweeps the dollar-impact calculation across a range of
approval rates instead of a single point estimate."""
import pandas as pd
from sklearn.pipeline import Pipeline

from backtest.impact import dollar_impact
from config import APPROVAL_RATE_SWEEP


def approval_rate_sensitivity(
    pipeline: Pipeline, batch_df: pd.DataFrame, approval_rates: list[float] = APPROVAL_RATE_SWEEP
) -> pd.DataFrame:
    rows = []
    for rate in approval_rates:
        result = dollar_impact(pipeline, batch_df, approval_rate=rate)
        rows.append({"approval_rate": rate, **result})
    return pd.DataFrame(rows)
