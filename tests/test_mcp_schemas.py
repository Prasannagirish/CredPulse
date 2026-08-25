from mcp_server.schemas import (
    DriftReportOutput,
    FeatureContribution,
    FeatureDrift,
    ModelHealth,
    PredictionExplanation,
    PromoteResult,
    RetrainResult,
    RollbackResult,
)


def test_model_health_constructs_and_serializes():
    health = ModelHealth(
        champion_version="1", days_since_last_retrain=3.5, recent_batch_auc=0.68,
        recent_batch_f1=0.2, recent_batch_label="2019-10", drift_status="warning",
    )
    assert health.model_dump()["drift_status"] == "warning"


def test_drift_report_output_nests_feature_drift():
    report = DriftReportOutput(
        window="2019-06", status="breach", autoencoder_status="ok",
        autoencoder_fraction_above_threshold=0.01, statistical_status="breach",
        top_drifting_features=[FeatureDrift(feature="grade", type="categorical", psi=0.63, ks_stat=None)],
    )
    assert report.top_drifting_features[0].feature == "grade"


def test_retrain_result_constructs():
    result = RetrainResult(
        training_window_start="2018-11-01", training_window_end="2019-10-01",
        holdout_window_start="2019-11-01", holdout_window_end="2020-01-01",
        champion_auc=0.63, challenger_auc=0.68, champion_f1=0.02, challenger_f1=0.06,
        champion_brier=0.08, challenger_brier=0.07, challenger_wins=True, challenger_version="2",
    )
    assert result.challenger_wins is True


def test_promote_and_rollback_results_construct():
    assert PromoteResult(new_champion_version="2").new_champion_version == "2"
    assert RollbackResult(reverted_to_version="1").reverted_to_version == "1"


def test_prediction_explanation_nests_contributions():
    explanation = PredictionExplanation(
        loan_id="1001", predicted_default_probability=0.42,
        top_contributions=[FeatureContribution(feature="int_rate", value="15.0", shap_contribution=0.31)],
    )
    assert explanation.top_contributions[0].feature == "int_rate"
