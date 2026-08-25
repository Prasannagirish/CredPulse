"""MCP server exposing CreditPulse's model lifecycle as tools (spec §7). All results are
backtest computations over historical Lending Club data — never a live evaluation."""
import re

import pandas as pd
from mcp.server.mcpserver import MCPServer
from sklearn.metrics import f1_score

import config
from drift.engine import evaluate_drift
from mcp_server.schemas import (
    DriftReportOutput,
    FeatureContribution,
    FeatureDrift,
    ModelHealth,
    PredictionExplanation,
)
from mcp_server import state
from model.train import evaluate_auc

mcp = MCPServer("creditpulse")


def _raw_feature_for_transformed_name(name: str) -> str:
    """Map a ColumnTransformer output name (e.g. "categorical__grade_A") back to its raw
    feature name ("grade"), so SHAP contributions can be reported in human terms rather than
    one-hot-encoded column names."""
    prefix, _, rest = name.partition("__")
    if prefix == "numeric":
        return rest
    if prefix == "categorical":
        candidates = [f for f in config.CATEGORICAL_FEATURES if rest.startswith(f + "_") or rest == f]
        if candidates:
            return max(candidates, key=len)
    return name


def _resolve_window(eval_df: pd.DataFrame, window: str) -> tuple[str, pd.DataFrame]:
    """Resolve a window spec against the eval data's own max issue_d as "now" — this is a
    backtest over historical data, not a live stream, so "last_30d" means the last 30 days
    of the most recent data available, not 30 real days ago."""
    match = re.fullmatch(r"last_(\d+)d", window)
    if match:
        days = int(match.group(1))
        now = eval_df["issue_d"].max()
        start = now - pd.Timedelta(days=days)
        subset = eval_df[(eval_df["issue_d"] > start) & (eval_df["issue_d"] <= now)]
        return window, subset
    df = eval_df.copy()
    df["month"] = df["issue_d"].dt.to_period("M").astype(str)
    subset = df[df["month"] == window]
    if len(subset) == 0:
        raise ValueError(f"No data found for window={window!r}")
    return window, subset


def get_model_health_impl() -> ModelHealth:
    srv_state = state.get_state()
    batch_label, batch_df = state.latest_reliable_batch(srv_state.eval_df)

    recent_auc = evaluate_auc(srv_state.champion_pipeline, batch_df)
    feature_cols = [c for c in batch_df.columns if c not in config.NON_FEATURE_COLUMNS]
    y_prob = srv_state.champion_pipeline.predict_proba(batch_df[feature_cols])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    recent_f1 = f1_score(batch_df[config.TARGET_COLUMN], y_pred)

    comparison_X = state.transform_for_drift(srv_state.champion_pipeline, batch_df)
    report = evaluate_drift(
        window=batch_label,
        statistical_reference=srv_state.statistical_reference,
        autoencoder=srv_state.autoencoder,
        reference_errors=srv_state.reference_errors,
        comparison_df=batch_df,
        comparison_X=comparison_X,
        numeric_features=config.NUMERIC_FEATURES,
        categorical_features=config.CATEGORICAL_FEATURES,
    )

    if srv_state.champion_promoted_at_ms:
        days_since = (pd.Timestamp.now().timestamp() * 1000 - srv_state.champion_promoted_at_ms) / 86_400_000
    else:
        days_since = 0.0

    return ModelHealth(
        champion_version=srv_state.champion_version,
        days_since_last_retrain=round(days_since, 1),
        recent_batch_auc=recent_auc,
        recent_batch_f1=recent_f1,
        recent_batch_label=batch_label,
        drift_status=report.status,
    )


def get_drift_report_impl(window: str = "last_30d") -> DriftReportOutput:
    srv_state = state.get_state()
    resolved_window, subset = _resolve_window(srv_state.eval_df, window)
    comparison_X = state.transform_for_drift(srv_state.champion_pipeline, subset)

    report = evaluate_drift(
        window=resolved_window,
        statistical_reference=srv_state.statistical_reference,
        autoencoder=srv_state.autoencoder,
        reference_errors=srv_state.reference_errors,
        comparison_df=subset,
        comparison_X=comparison_X,
        numeric_features=config.NUMERIC_FEATURES,
        categorical_features=config.CATEGORICAL_FEATURES,
    )

    top_features = [
        FeatureDrift(
            feature=row.feature, type=row.type, psi=row.psi,
            ks_stat=None if pd.isna(row.ks_stat) else row.ks_stat,
        )
        for row in report.feature_drift.head(5).itertuples()
    ]
    return DriftReportOutput(
        window=resolved_window,
        status=report.status,
        autoencoder_status=report.autoencoder_status,
        autoencoder_fraction_above_threshold=report.autoencoder_fraction_above_threshold,
        statistical_status=report.statistical_status,
        top_drifting_features=top_features,
    )


def explain_prediction_impl(loan_id: str) -> PredictionExplanation:
    srv_state = state.get_state()
    combined = pd.concat([srv_state.reference_df, srv_state.eval_df], ignore_index=True)
    matches = combined[combined["id"].astype(str) == str(loan_id)]
    if len(matches) == 0:
        raise ValueError(f"No loan found with id={loan_id!r}")
    loan = matches.iloc[0]

    feature_cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    X_raw = loan[feature_cols].to_frame().T
    preprocessor = srv_state.champion_pipeline.named_steps["preprocess"]
    X_transformed = preprocessor.transform(X_raw)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    predicted_proba = float(
        srv_state.champion_pipeline.named_steps["model"].predict_proba(X_transformed)[0, 1]
    )
    shap_row = srv_state.explainer.shap_values(X_transformed)[0]
    transformed_names = preprocessor.get_feature_names_out()

    contributions: dict[str, float] = {}
    for name, value in zip(transformed_names, shap_row):
        raw = _raw_feature_for_transformed_name(name)
        contributions[raw] = contributions.get(raw, 0.0) + float(value)

    top = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    top_contributions = [
        FeatureContribution(feature=f, value=str(loan[f]), shap_contribution=v) for f, v in top
    ]

    return PredictionExplanation(
        loan_id=str(loan_id),
        predicted_default_probability=predicted_proba,
        top_contributions=top_contributions,
    )


@mcp.tool()
def get_model_health() -> ModelHealth:
    """Current champion model version, days since last retrain, AUC/F1 on the most recently
    reliably-labeled historical batch, and current overall drift status. Backtest on
    historical Lending Club data, not a live evaluation."""
    return get_model_health_impl()


@mcp.tool()
def get_drift_report(window: str = "last_30d") -> DriftReportOutput:
    """Per-feature PSI/KS drift and the autoencoder's reconstruction-error signal for a
    window of historical loans, plus which features are driving any breach. `window`
    accepts "last_Nd" (relative to the most recent data available) or an explicit "YYYY-MM"
    month."""
    return get_drift_report_impl(window)


@mcp.tool()
def explain_prediction(loan_id: str) -> PredictionExplanation:
    """The champion model's predicted default probability for a given historical loan, plus
    its top SHAP feature contributions — framed like an adverse-action explanation."""
    return explain_prediction_impl(loan_id)
