"""Feature allow-list, target derivation, and the shared preprocessing pipeline."""
import pandas as pd

from config import TARGET_COLUMN

_POSITIVE_STATUSES = {"Charged Off", "Default"}
_NEGATIVE_STATUSES = {"Fully Paid"}


def derive_target(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to resolved loans and attach a binary default_flag column.

    Loans still "Current" or in any other non-terminal state (Late, In Grace Period,
    or the legacy "Does not meet the credit policy." variants) are excluded — their
    outcome isn't resolved yet, so including them would leak censored labels into
    training (spec §3).
    """
    resolved = df[df["loan_status"].isin(_POSITIVE_STATUSES | _NEGATIVE_STATUSES)].copy()
    resolved[TARGET_COLUMN] = resolved["loan_status"].isin(_POSITIVE_STATUSES).astype(int)
    return resolved
