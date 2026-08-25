"""Load the raw Lending Club CSV, apply the temporal split, and write partitioned parquet."""
from pathlib import Path

import pandas as pd

from config import EVAL_END, EVAL_START, PROCESSED_DIR, RAW_CSV_PATH, REFERENCE_END
from model.features import clean_raw_types, derive_target, select_features


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


def run_prepare(
    raw_csv_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, split, and persist the reference/eval parquet files.

    Backtest note: everything downstream of this function is a simulation on historical,
    already-resolved loan outcomes — not a live scoring pipeline.
    """
    raw_csv_path = raw_csv_path or RAW_CSV_PATH
    output_dir = output_dir or PROCESSED_DIR

    raw = pd.read_csv(raw_csv_path, low_memory=False)

    print(f"Loaded {len(raw)} rows, {len(raw.columns)} columns from {raw_csv_path}")
    print(raw.dtypes)
    print(raw["loan_status"].value_counts())

    cleaned = clean_raw_types(raw)
    labeled = derive_target(cleaned)
    selected = select_features(labeled)
    selected["issue_d"] = raw.loc[selected.index, "issue_d"]

    reference_df, eval_df = temporal_split(selected)

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_df.to_parquet(output_dir / "reference.parquet", index=False)
    eval_df.to_parquet(output_dir / "eval.parquet", index=False)

    return reference_df, eval_df


if __name__ == "__main__":
    run_prepare()
