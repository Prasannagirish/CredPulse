"""Central paths and constants shared across data prep, drift, backtest, and the MCP server."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

RAW_CSV_PATH = Path.home() / "Downloads" / "creditpulse-data" / "accepted_2007_to_2018q4.csv" / "accepted_2007_to_2018Q4.csv"

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
