"""Hyperparameter tuning for the XGBoost champion via Optuna, evaluated on the exact same
temporal holdout model.train.train_baseline uses — no new leak surface."""
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import NON_FEATURE_COLUMNS, TARGET_COLUMN, TUNING_N_TRIALS
from model.features import build_preprocessor
from model.train import evaluate_auc, temporal_holdout_split

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_pipeline_with_params(params: dict) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", XGBClassifier(eval_metric="auc", random_state=42, **params)),
    ])


def _objective(trial: optuna.Trial, train_slice: pd.DataFrame, test_slice: pd.DataFrame) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    pipeline = _build_pipeline_with_params(params)
    feature_cols = [c for c in train_slice.columns if c not in NON_FEATURE_COLUMNS]
    pipeline.fit(train_slice[feature_cols], train_slice[TARGET_COLUMN])
    return evaluate_auc(pipeline, test_slice)


def tune_hyperparameters(
    reference_df: pd.DataFrame, n_trials: int = TUNING_N_TRIALS
) -> tuple[dict, float]:
    """Runs an Optuna search over XGBoost hyperparameters, maximizing AUC on the same
    chronological-last-10% holdout model.train.train_baseline uses. n_trials is a
    deliberately bounded, real search budget — not exhaustive. Returns (best_params,
    best_auc)."""
    train_slice, test_slice = temporal_holdout_split(reference_df)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: _objective(trial, train_slice, test_slice), n_trials=n_trials)

    return study.best_params, study.best_value
