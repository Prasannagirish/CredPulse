"""Pydantic models for MCP tool structured outputs (spec §7). All figures are computed from
a backtest over historical Lending Club data."""
from pydantic import BaseModel


class ModelHealth(BaseModel):
    champion_version: str
    days_since_last_retrain: float
    recent_batch_auc: float
    recent_batch_f1: float
    recent_batch_label: str
    drift_status: str


class FeatureDrift(BaseModel):
    feature: str
    type: str
    psi: float
    ks_stat: float | None


class DriftReportOutput(BaseModel):
    window: str
    status: str
    autoencoder_status: str
    autoencoder_fraction_above_threshold: float
    statistical_status: str
    top_drifting_features: list[FeatureDrift]


class RetrainResult(BaseModel):
    training_window_start: str
    training_window_end: str
    holdout_window_start: str
    holdout_window_end: str
    champion_auc: float
    challenger_auc: float
    champion_f1: float
    challenger_f1: float
    champion_brier: float
    challenger_brier: float
    challenger_wins: bool
    challenger_version: str


class PromoteResult(BaseModel):
    new_champion_version: str


class RollbackResult(BaseModel):
    reverted_to_version: str


class FeatureContribution(BaseModel):
    feature: str
    value: str
    shap_contribution: float


class PredictionExplanation(BaseModel):
    loan_id: str
    predicted_default_probability: float
    top_contributions: list[FeatureContribution]
