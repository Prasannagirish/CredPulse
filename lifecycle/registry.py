"""Thin wrapper over MLflow's model registry for champion/challenger get/register/promote/
rollback. Uses aliases (champion/challenger), not the deprecated stage API."""
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

import config

_CHAMPION_ALIAS = "champion"
_CHALLENGER_ALIAS = "challenger"
_PREVIOUS_CHAMPION_TAG = "previous_champion_version"


class ConfirmationRequired(Exception):
    """Raised when a lifecycle-mutating call is attempted without confirm=True."""


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return MlflowClient()


def _resolve_name(name: str | None) -> str:
    return name if name is not None else config.MODEL_NAME


def register_model(model_uri: str, name: str | None = None) -> str:
    """Register a logged MLflow model (by its run URI, e.g. f"runs:/{run_id}/model") as a
    new version under `name`. Creates the registered model if it doesn't exist yet."""
    name = _resolve_name(name)
    client = _client()
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)
    result = mlflow.register_model(model_uri, name)
    # MLflow returns ModelVersion.version as an int, but tags (used below for
    # previous_champion_version) always round-trip as str — normalize to str everywhere in
    # this module so callers never have to care which MLflow API a version number came from.
    return str(result.version)


def set_challenger(version: str, name: str | None = None) -> None:
    name = _resolve_name(name)
    _client().set_registered_model_alias(name, _CHALLENGER_ALIAS, version)


def get_champion_version(name: str | None = None) -> str | None:
    name = _resolve_name(name)
    try:
        return str(_client().get_model_version_by_alias(name, _CHAMPION_ALIAS).version)
    except MlflowException:
        return None


def get_challenger_version(name: str | None = None) -> str | None:
    name = _resolve_name(name)
    try:
        return str(_client().get_model_version_by_alias(name, _CHALLENGER_ALIAS).version)
    except MlflowException:
        return None


def promote_challenger(confirm: bool, name: str | None = None) -> str:
    """Promote the current challenger to champion. Refuses unless confirm is exactly True —
    a deliberate safety property (spec §2): no silent auto-promotion, ever."""
    if confirm is not True:
        raise ConfirmationRequired("promote_challenger requires confirm=True")

    name = _resolve_name(name)
    client = _client()
    challenger_version = get_challenger_version(name)
    if challenger_version is None:
        raise ValueError(f"No challenger registered for model '{name}'")

    previous_champion_version = get_champion_version(name)
    if previous_champion_version is not None:
        client.set_registered_model_tag(name, _PREVIOUS_CHAMPION_TAG, previous_champion_version)

    client.set_registered_model_alias(name, _CHAMPION_ALIAS, challenger_version)
    client.delete_registered_model_alias(name, _CHALLENGER_ALIAS)
    return challenger_version


def rollback(confirm: bool, name: str | None = None) -> str:
    """Revert to the previous champion version. Refuses unless confirm is exactly True."""
    if confirm is not True:
        raise ConfirmationRequired("rollback requires confirm=True")

    name = _resolve_name(name)
    client = _client()
    model = client.get_registered_model(name)
    previous_version = model.tags.get(_PREVIOUS_CHAMPION_TAG)
    if previous_version is None:
        raise ValueError(f"No previous champion recorded for model '{name}' to roll back to")

    client.set_registered_model_alias(name, _CHAMPION_ALIAS, previous_version)
    return previous_version
