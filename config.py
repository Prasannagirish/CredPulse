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

# Drift thresholds (design doc §5)
PSI_WARNING = 0.10
PSI_BREACH = 0.25

# A quarter's default label is only trustworthy once this many defaults have actually been
# observed — quarters near the eval window's data cutoff are right-censored (loans haven't
# had time to default yet), which is why Phase 1's report treats them as unreliable.
MIN_RELIABLE_DEFAULTS = 100

# Autoencoder drift thresholds. By construction, ~1% of in-distribution rows exceed the
# reference set's own 99th-percentile reconstruction error — these fractions describe how
# much larger than that 1% baseline a window's exceedance rate must be to count as drift.
AE_ERROR_PERCENTILE = 0.99
AE_WARNING_FRACTION = 0.02
AE_BREACH_FRACTION = 0.05

# Default trailing window (in months) a challenger retrains on, counting back from the
# simulated "as of" retrain date.
RETRAIN_WINDOW_MONTHS = 12

MODEL_NAME = "creditpulse-credit-risk"

# Columns that are never model features, in every "which columns go into X" computation
# across model/train.py, lifecycle/evaluate.py, lifecycle/retrain.py, and the MCP server.
NON_FEATURE_COLUMNS = (TARGET_COLUMN, "issue_d", "id")

# Backtest arm cadence (Phase 5). Scheduled retrains on a blind calendar cadence; monitored
# retrains only on drift breach, with a cooldown so a persistent breach signal doesn't
# trigger a retrain every single month.
SCHEDULED_RETRAIN_MONTHS = 3
MONITORED_RETRAIN_COOLDOWN_MONTHS = 2

# Dollar-impact assumptions (spec §8 Phase 5) — shown explicitly in the report, never a
# hidden headline number (spec §2).
APPROVAL_RATE = 0.80  # accept the lowest-predicted-risk 80% of applicants
LOSS_GIVEN_DEFAULT = 1.0  # simplification: a defaulted loan's full principal is the loss —
# no partial-recovery modeling, since recovery amounts are excluded from features as
# leaky post-origination fields (spec §3) and genuinely aren't available here either.
