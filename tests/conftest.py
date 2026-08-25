import os

# Must be set before numpy/xgboost/torch are imported by any test module: on macOS, XGBoost
# and PyTorch each bundle their own OpenMP runtime, and running both in one process without
# this segfaults (SIGSEGV) as soon as XGBoost tries to fit. pytest loads conftest.py before
# collecting/importing test modules, so this runs early enough to matter.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
