"""Runtime state for the MCP server: loads/bootstraps the champion, fits the drift
reference, and builds a SHAP explainer — computed once per process and cached."""
import os

# Must be set before numpy/xgboost/torch are imported: on macOS, XGBoost and PyTorch each
# bundle their own OpenMP runtime, and running both in one process without this crashes
# (Phase 2/3 finding) — this module is the first place the MCP server imports both.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

import config
from drift.autoencoder import DriftAutoencoder, reconstruction_error, train_autoencoder
from drift.statistical import StatisticalDriftReference, fit_statistical_reference
from lifecycle.registry import (
    get_champion_promoted_at,
    get_champion_version,
    promote_challenger,
    register_model,
    set_challenger,
)
from model.train import train_baseline

# MLflow's sklearn flavor serializes with skops by default, which refuses to save a
# pipeline containing an XGBClassifier as "untrusted" unless explicitly whitelisted (Phase 3
# finding) — reused here for the bootstrap path and by mcp_server/server.py's mutating tools.
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]


@dataclass
class ServerState:
    champion_pipeline: Pipeline
    champion_version: str
    champion_promoted_at_ms: int | None
    reference_df: pd.DataFrame
    eval_df: pd.DataFrame
    statistical_reference: StatisticalDriftReference
    autoencoder: DriftAutoencoder
    reference_errors: np.ndarray
    explainer: object
    # The pipeline whose preprocessor statistical_reference/autoencoder/reference_errors were
    # fit through — fixed at bootstrap and never reassigned by refresh_champion(). Drift
    # comparisons must always transform through THIS pipeline, never champion_pipeline once
    # it may have been swapped for a differently-fit one by a promotion — otherwise the
    # frozen reference would silently be compared against a different feature space.
    reference_pipeline: Pipeline


_state: ServerState | None = None


def transform_for_drift(champion_pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    preprocessor = champion_pipeline.named_steps["preprocess"]
    X = preprocessor.transform(df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES])
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X.astype(np.float32)


def latest_reliable_batch(
    eval_df: pd.DataFrame, min_defaults: int | None = None
) -> tuple[str, pd.DataFrame]:
    """Most recent calendar month with at least min_defaults observed defaults — the same
    reliability bar Phase 1 established to avoid right-censored, near-meaningless metrics
    close to the data cutoff. min_defaults defaults to config.MIN_RELIABLE_DEFAULTS, looked
    up here rather than as a function-definition-time default so tests can monkeypatch it."""
    min_defaults = config.MIN_RELIABLE_DEFAULTS if min_defaults is None else min_defaults
    df = eval_df.copy()
    df["month"] = df["issue_d"].dt.to_period("M").astype(str)
    defaults_by_month = df.groupby("month")[config.TARGET_COLUMN].sum()
    reliable_months = defaults_by_month[defaults_by_month >= min_defaults].index
    if len(reliable_months) == 0:
        raise ValueError("No calendar month has enough observed defaults to be reliable")
    latest_month = max(reliable_months)
    return latest_month, df[df["month"] == latest_month]


def _load_or_bootstrap_champion(reference_df: pd.DataFrame) -> tuple[Pipeline, str, int | None]:
    """Load the current registry champion, or bootstrap-train and register the Phase 1
    baseline if no champion exists yet (e.g. a fresh checkout with an empty registry)."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    version = get_champion_version()
    if version is None:
        pipeline, _, _ = train_baseline(reference_df)
        with mlflow.start_run(run_name="mcp-bootstrap-champion") as run:
            mlflow.sklearn.log_model(pipeline, "model", skops_trusted_types=SKOPS_TRUSTED_TYPES)
            model_uri = f"runs:/{run.info.run_id}/model"
        version = register_model(model_uri)
        set_challenger(version)
        promote_challenger(confirm=True)
    else:
        pipeline = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@champion")

    return pipeline, version, get_champion_promoted_at()


def _build_state() -> ServerState:
    reference_df = pd.read_parquet(config.PROCESSED_DIR / "reference.parquet")
    eval_df = pd.read_parquet(config.PROCESSED_DIR / "eval.parquet")

    champion_pipeline, champion_version, promoted_at_ms = _load_or_bootstrap_champion(reference_df)

    statistical_reference = fit_statistical_reference(
        reference_df, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES
    )
    reference_X = transform_for_drift(champion_pipeline, reference_df)
    autoencoder = train_autoencoder(reference_X, epochs=10, batch_size=1024)
    reference_errors = reconstruction_error(autoencoder, reference_X)

    explainer = shap.TreeExplainer(champion_pipeline.named_steps["model"])

    return ServerState(
        champion_pipeline=champion_pipeline,
        champion_version=champion_version,
        champion_promoted_at_ms=promoted_at_ms,
        reference_df=reference_df,
        eval_df=eval_df,
        statistical_reference=statistical_reference,
        autoencoder=autoencoder,
        reference_errors=reference_errors,
        explainer=explainer,
        reference_pipeline=champion_pipeline,
    )


def get_state(force_refresh: bool = False) -> ServerState:
    global _state
    if _state is None or force_refresh:
        _state = _build_state()
    return _state


def refresh_champion() -> None:
    """Reload just the champion pipeline/version/timestamp/explainer after promote/rollback
    — the drift reference and autoencoder are fit on the fixed pre-2019 reference window and
    don't change when the champion does, so rebuilding them here would be wasted work."""
    global _state
    if _state is None:
        return
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    pipeline = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@champion")
    _state.champion_pipeline = pipeline
    _state.champion_version = get_champion_version()
    _state.champion_promoted_at_ms = get_champion_promoted_at()
    _state.explainer = shap.TreeExplainer(pipeline.named_steps["model"])
