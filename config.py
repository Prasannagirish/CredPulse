"""Central paths and constants shared across data prep, drift, backtest, and the MCP server."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# wordsforthewise/lending-club (accepted_2007_to_2018Q4.csv) only covers loans through
# 2018Q4 — no rows fall in the 2019-2020 eval window at all, so Phase 1 originally used the
# wrong mirror. ethon0426/lending-club-20072020q1 extends through 2020-09 and matches the
# spec's ~2.9M row estimate exactly (2,925,494 rows).
RAW_CSV_PATH = Path.home() / "Downloads" / "creditpulse-data-2020" / "Loan_status_2007-2020Q3.gzip"

VAR_DIR = REPO_ROOT / "var"
PROCESSED_DIR = VAR_DIR / "processed"
REPORTS_DIR = REPO_ROOT / "reports"

MLFLOW_TRACKING_URI = f"sqlite:///{VAR_DIR / 'mlflow.db'}"

# Temporal split boundaries (spec §3) — loans issued before REFERENCE_END are the
# reference/training window; loans issued in [EVAL_START, EVAL_END) are the drift/eval window.
REFERENCE_END = "2019-01-01"
EVAL_START = "2019-01-01"
EVAL_END = "2021-01-01"

TARGET_COLUMN = "default_flag"

# Safe-by-default: only these raw columns may become model features. Anything not listed
# here is excluded, even if the raw schema changes to add new columns.
FEATURE_COLUMNS = [
    "loan_amnt",
    "term",
    "int_rate",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "dti",
    "revol_util",
    "fico_range_low",
    "fico_range_high",
    "purpose",
]

NUMERIC_FEATURES = [
    "loan_amnt", "term", "int_rate", "emp_length", "annual_inc",
    "dti", "revol_util", "fico_range_low", "fico_range_high",
]
CATEGORICAL_FEATURES = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]
