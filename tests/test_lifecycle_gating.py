"""Asserts promote_challenger/rollback refuse to act without confirm=True, and that
registry state is unchanged after a refusal — spec §2's core safety property."""
import mlflow
import mlflow.sklearn
import pytest
from sklearn.linear_model import LogisticRegression

import config
from lifecycle.registry import (
    ConfirmationRequired,
    get_champion_version,
    get_challenger_version,
    promote_challenger,
    register_model,
    rollback,
    set_challenger,
)


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'test_mlflow.db'}")
    monkeypatch.setattr(config, "MODEL_NAME", "test-model")
    # MLflow's Default experiment gets an artifact_location resolved relative to cwd
    # (./mlruns) the first time a fresh tracking DB is touched, regardless of the sqlite
    # backend URI above — without this, every test run leaves an mlruns/ directory behind
    # in the repo root. Point it at tmp_path instead so it's cleaned up with the test.
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.create_experiment("test-experiment", artifact_location=f"file://{tmp_path / 'mlartifacts'}")
    mlflow.set_experiment("test-experiment")


def _log_and_register_dummy_model() -> str:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    with mlflow.start_run() as run:
        model = LogisticRegression().fit([[0], [1]], [0, 1])
        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
    return register_model(model_uri)


def test_promote_challenger_refuses_without_confirm_true():
    version = _log_and_register_dummy_model()
    set_challenger(version)

    with pytest.raises(ConfirmationRequired):
        promote_challenger(confirm=False)

    assert get_champion_version() is None
    assert get_challenger_version() == version


def test_promote_challenger_refuses_truthy_non_bool_values():
    version = _log_and_register_dummy_model()
    set_challenger(version)

    for bad_value in [1, "true", "yes"]:
        with pytest.raises(ConfirmationRequired):
            promote_challenger(confirm=bad_value)
        assert get_champion_version() is None


def test_promote_challenger_succeeds_with_confirm_true():
    version = _log_and_register_dummy_model()
    set_challenger(version)

    promoted_version = promote_challenger(confirm=True)

    assert promoted_version == version
    assert get_champion_version() == version
    assert get_challenger_version() is None


def test_rollback_refuses_without_confirm_true():
    v1 = _log_and_register_dummy_model()
    set_challenger(v1)
    promote_challenger(confirm=True)

    v2 = _log_and_register_dummy_model()
    set_challenger(v2)
    promote_challenger(confirm=True)

    with pytest.raises(ConfirmationRequired):
        rollback(confirm=False)

    assert get_champion_version() == v2


def test_rollback_reverts_to_previous_champion_with_confirm_true():
    v1 = _log_and_register_dummy_model()
    set_challenger(v1)
    promote_challenger(confirm=True)

    v2 = _log_and_register_dummy_model()
    set_challenger(v2)
    promote_challenger(confirm=True)

    reverted_version = rollback(confirm=True)

    assert reverted_version == v1
    assert get_champion_version() == v1


def test_get_champion_promoted_at_returns_timestamp_after_promotion():
    from lifecycle.registry import get_champion_promoted_at

    assert get_champion_promoted_at() is None

    version = _log_and_register_dummy_model()
    set_challenger(version)
    promote_challenger(confirm=True)

    promoted_at = get_champion_promoted_at()
    assert isinstance(promoted_at, int)
    assert promoted_at > 0
