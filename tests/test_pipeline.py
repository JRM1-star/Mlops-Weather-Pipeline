from mlflow import MlflowClient

from src import tracking
from src.pipeline import run_all


def test_end_to_end_and_champion_gate(params):
    first = run_all(params)
    assert first["promoted"] and first["candidate_version"] == "1"

    tracking.setup(params)
    name = params["mlflow"]["model_name"]
    assert str(MlflowClient().get_model_version_by_alias(name, "champion").version) == "1"

    # identical data -> identical model -> must NOT replace the champion
    second = run_all(params)
    assert not second["promoted"]
    assert str(MlflowClient().get_model_version_by_alias(name, "champion").version) == "1"
