import os
from pathlib import Path

import mlflow


def setup(params):
    """Point MLflow at the configured store and make sure the experiment exists."""
    cfg = params["mlflow"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
        root = os.getenv("MLFLOW_ARTIFACT_ROOT") or Path("mlartifacts").resolve().as_uri()
        mlflow.create_experiment(cfg["experiment"], artifact_location=root)
    mlflow.set_experiment(cfg["experiment"])
