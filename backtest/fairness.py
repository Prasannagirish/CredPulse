"""Disparate-impact screening across documented proxy groups.

LIMITATION: Lending Club's public dataset contains no protected-class labels (race, sex,
age, national origin, marital status) — genuine fair-lending analysis is not possible on
this data. This module uses two literature-grounded PROXIES instead — home_ownership and
income quartiles of annual_inc — and its results are a screening heuristic, not a legal or
regulatory determination.
"""
import pandas as pd
from sklearn.pipeline import Pipeline

import config
from backtest.impact import compute_accepted_mask


def group_approval_rates(
    pipeline: Pipeline, df: pd.DataFrame, group_col: str, approval_rate: float = config.APPROVAL_RATE
) -> pd.DataFrame:
    """Approval and default rates broken out by group_col, using ONE shared risk threshold
    for the whole df — group_col plays no role in the accept/reject decision itself, only
    in how the single decision's outcomes are reported per group."""
    df = df.reset_index(drop=True)
    accepted_mask = compute_accepted_mask(pipeline, df, approval_rate)

    rows = []
    for group_value in sorted(df[group_col].dropna().unique(), key=str):
        group_mask = (df[group_col] == group_value).to_numpy()
        n_total = int(group_mask.sum())
        group_accepted = group_mask & accepted_mask
        n_accepted = int(group_accepted.sum())
        n_defaults = int(df.loc[group_accepted, config.TARGET_COLUMN].sum())
        rows.append({
            "group": group_value,
            "n_total": n_total,
            "n_accepted": n_accepted,
            "approval_rate": (n_accepted / n_total) if n_total else 0.0,
            "n_defaults": n_defaults,
            "default_rate": (n_defaults / n_accepted) if n_accepted else 0.0,
        })
    return pd.DataFrame(rows)


def disparate_impact_ratio(group_rates: pd.DataFrame) -> dict:
    """Standard US "4/5ths rule" screen: minimum group approval rate divided by maximum.
    Below config.DISPARATE_IMPACT_THRESHOLD is flagged — a screening heuristic, not a legal
    determination."""
    min_row = group_rates.loc[group_rates["approval_rate"].idxmin()]
    max_row = group_rates.loc[group_rates["approval_rate"].idxmax()]
    ratio = (min_row["approval_rate"] / max_row["approval_rate"]) if max_row["approval_rate"] > 0 else 1.0
    return {
        "ratio": float(ratio),
        "min_group": min_row["group"],
        "max_group": max_row["group"],
        "flagged": bool(ratio < config.DISPARATE_IMPACT_THRESHOLD),
    }


def add_income_quartile_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds an income_quartile column computed from this df's own annual_inc distribution —
    never a hardcoded dollar boundary."""
    out = df.copy()
    out["income_quartile"] = pd.qcut(
        out["annual_inc"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    )
    return out
