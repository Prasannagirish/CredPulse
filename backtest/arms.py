"""Three backtest strategies for the monthly walk-forward: static (never retrains,
represented as an ArmState nothing ever touches), scheduled (blind periodic retrain), and
monitored (drift-triggered retrain)."""
from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

from config import MONITORED_RETRAIN_COOLDOWN_MONTHS, RETRAIN_WINDOW_MONTHS, SCHEDULED_RETRAIN_MONTHS
from drift.engine import DriftReport
from lifecycle.retrain import retrain_challenger, select_recent_window


@dataclass
class ArmState:
    name: str
    pipeline: Pipeline
    last_retrain_date: pd.Timestamp
    retrain_count: int = 0


def _months_between(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def maybe_retrain_scheduled(arm: ArmState, month: pd.Timestamp, all_loans_df: pd.DataFrame) -> ArmState:
    """Blindly retrain every SCHEDULED_RETRAIN_MONTHS months, regardless of drift."""
    if _months_between(month, arm.last_retrain_date) < SCHEDULED_RETRAIN_MONTHS:
        return arm
    training_window = select_recent_window(all_loans_df, month, months=RETRAIN_WINDOW_MONTHS)
    new_pipeline = retrain_challenger(training_window)
    return ArmState(arm.name, new_pipeline, month, arm.retrain_count + 1)


def maybe_retrain_monitored(
    arm: ArmState, month: pd.Timestamp, all_loans_df: pd.DataFrame, report: DriftReport
) -> ArmState:
    """Retrain only when this month's drift status is not "ok", and only if the cooldown
    since the last retrain has elapsed — otherwise a persistent breach would trigger a
    retrain every single month, re-fitting on nearly-identical overlapping windows."""
    if report.status == "ok":
        return arm
    if _months_between(month, arm.last_retrain_date) < MONITORED_RETRAIN_COOLDOWN_MONTHS:
        return arm
    training_window = select_recent_window(all_loans_df, month, months=RETRAIN_WINDOW_MONTHS)
    new_pipeline = retrain_challenger(training_window)
    return ArmState(arm.name, new_pipeline, month, arm.retrain_count + 1)
