"""Train a candidate model, log everything to MLflow, register a new model version."""
import hashlib
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import tracking
from src.config import load_params
from src.features import SRC, feature_columns

HGB_TRUSTED = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def build_model(t):
    if t["model"] == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**t["ridge"]))
    return HistGradientBoostingRegressor(**t["hgb"], random_state=t["seed"])


def run(params):
    tracking.setup(params)
    proc = Path(params["paths"]["processed_dir"])
    train = pd.read_csv(proc / "train.csv", parse_dates=["date"])
    test = pd.read_csv(proc / "test.csv", parse_dates=["date"])
    cols = feature_columns(train)
    t = params["train"]
    model = build_model(t)

    cv_mae = float(np.mean([
        mean_absolute_error(
            train.iloc[b]["target"],
            clone(model).fit(train.iloc[a][cols], train.iloc[a]["target"]).predict(train.iloc[b][cols]),
        )
        for a, b in TimeSeriesSplit(n_splits=5).split(train)
    ]))

    model.fit(train[cols], train["target"])
    pred = model.predict(test[cols])
    metrics = {
        "test_mae": float(mean_absolute_error(test["target"], pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(test["target"], pred))),
        "test_r2": float(r2_score(test["target"], pred)),
        "cv_mae": cv_mae,
        # persistence baseline: "tomorrow's max = today's max"
        "baseline_mae": float(mean_absolute_error(test["target"], test[SRC])),
    }
    metrics["skill_vs_baseline"] = 1 - metrics["test_mae"] / metrics["baseline_mae"]

    data_sha = hashlib.sha1(Path(params["paths"]["raw"]).read_bytes()).hexdigest()[:12]
    with mlflow.start_run() as run:
        mlflow.log_params({**{f"{t['model']}_{k}": v for k, v in t[t["model"]].items()}, **params["features"], "model": t["model"], "seed": t["seed"]})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "data_sha": data_sha,
            "data_last_date": str(test["date"].max().date()),
            "git_sha": os.getenv("GITHUB_SHA", "local"),
        })
        info = mlflow.sklearn.log_model(
            model, name="model",
            signature=infer_signature(test[cols], pred), input_example=test[cols].head(3),
            # we trained this model ourselves, so trusting this one type is safe (skops audit)
            skops_trusted_types=HGB_TRUSTED if t["model"] == "hgb" else None,
        )
        mv = mlflow.register_model(info.model_uri, params["mlflow"]["model_name"])

    rep = Path(params["paths"]["reports_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "metrics.json").write_text(json.dumps(metrics, indent=2))
    last = {"run_id": run.info.run_id, "model_version": str(mv.version), "metrics": metrics}
    (rep / "last_run.json").write_text(json.dumps(last, indent=2))
    print(f"train: v{mv.version} test_mae={metrics['test_mae']:.3f} baseline={metrics['baseline_mae']:.3f}")
    return last


if __name__ == "__main__":
    run(load_params())
