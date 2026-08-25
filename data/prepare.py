"""Load the raw Lending Club CSV, apply the temporal split, and write partitioned parquet."""
import pandas as pd

from config import EVAL_END, EVAL_START, REFERENCE_END


def temporal_split(df: pd.DataFrame, date_col: str = "issue_d") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split loans into the pre-2019 reference window and the 2019-2020 eval window.

    Never shuffle across this boundary — the whole point of the project is measuring
    how a model trained only on the reference window degrades on the eval window.
    """
    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        out[date_col] = pd.to_datetime(out[date_col], format="%b-%Y")

    reference = out[out[date_col] < REFERENCE_END].copy()
    eval_ = out[(out[date_col] >= EVAL_START) & (out[date_col] < EVAL_END)].copy()
    return reference, eval_
