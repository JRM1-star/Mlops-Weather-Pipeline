"""Champion/challenger gate: promote only if the candidate clearly beats the current champion."""
import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import mean_absolute_error

from src import tracking
from src.config import load_params
from src.features import feature_columns


def should_promote(candidate_mae, champion_mae, baseline_mae, min_improvement):
    if candidate_mae >= baseline_mae:
        return False, "candidate does not beat the persistence baseline"
    if champion_mae is None:
        return True, "no champion yet"
    needed = champion_mae * (1 - min_improvement)
    if candidate_mae <= needed:
        return True, f"candidate MAE {candidate_mae:.3f} <= required {needed:.3f}"
    return False, f"candidate MAE {candidate_mae:.3f} > required {needed:.3f}"


def run(params):
    tracking.setup(params)
    client, name = MlflowClient(), params["mlflow"]["model_name"]
    last = json.loads((Path(params["paths"]["reports_dir"]) / "last_run.json").read_text())
    test = pd.read_csv(Path(params["paths"]["processed_dir"]) / "test.csv", parse_dates=["date"])
    cols = feature_columns(test)

    def mae(uri):
        return float(mean_absolute_error(test["target"], mlflow.sklearn.load_model(uri).predict(test[cols])))

    version = last["model_version"]
    cand = mae(f"models:/{name}/{version}")
    try:
        champ_version = str(client.get_model_version_by_alias(name, "champion").version)
        # score the champion on the SAME fresh test set for a fair comparison
        champ = mae(f"models:/{name}@champion")
    except MlflowException:
        champ_version, champ = None, None

    ok, reason = should_promote(cand, champ, last["metrics"]["baseline_mae"], params["promotion"]["min_improvement"])
    if ok:
        client.set_registered_model_alias(name, "champion", version)
    client.set_model_version_tag(name, version, "decision", "promoted" if ok else "rejected")

    result = {
        "promoted": ok, "reason": reason, "candidate_version": version, "candidate_mae": cand,
        "champion_version": champ_version, "champion_mae": champ,
    }
    (Path(params["paths"]["reports_dir"]) / "promotion.json").write_text(json.dumps(result, indent=2))
    print("promote:", result)
    return result


if __name__ == "__main__":
    run(load_params())
