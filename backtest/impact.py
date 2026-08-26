"""Dollar-impact estimate: at a fixed approval rate, how much expected loss does each
arm's own model accept among its approved loans? Backtest metric only."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import config


def compute_accepted_mask(
    pipeline: Pipeline, df: pd.DataFrame, approval_rate: float = config.APPROVAL_RATE
) -> np.ndarray:
    """Boolean mask, True for the approval_rate fraction of df with the LOWEST predicted
    default risk — the model picks its own threshold to hit that rate. Shared by
    dollar_impact (below) and backtest.fairness.group_approval_rates, so both analyses use
    the exact same accept/reject decision."""
    feature_cols = [c for c in df.columns if c not in config.NON_FEATURE_COLUMNS]
    risk_scores = pipeline.predict_proba(df[feature_cols])[:, 1]
    threshold = np.quantile(risk_scores, approval_rate)
    return risk_scores <= threshold


def dollar_impact(
    pipeline: Pipeline, batch_df: pd.DataFrame, approval_rate: float = config.APPROVAL_RATE
) -> dict:
    """Accept the approval_rate fraction of batch_df with the LOWEST predicted default risk
    — each model picks its own threshold to hit that rate. This is what keeps the
    comparison fair: a model can't "win" simply by rejecting more loans than another."""
    accepted_mask = compute_accepted_mask(pipeline, batch_df, approval_rate)
    accepted = batch_df[accepted_mask]

    n_accepted = len(accepted)
    n_defaults = int(accepted[config.TARGET_COLUMN].sum())
    dollar_loss = float(
        accepted.loc[accepted[config.TARGET_COLUMN] == 1, "loan_amnt"].sum() * config.LOSS_GIVEN_DEFAULT
    )
    per_10k = (dollar_loss / n_accepted * 10_000) if n_accepted else 0.0

    return {
        "n_accepted": n_accepted,
        "n_defaults": n_defaults,
        "default_rate": (n_defaults / n_accepted) if n_accepted else 0.0,
        "dollar_loss": dollar_loss,
        "dollar_loss_per_10k_loans": per_10k,
    }
