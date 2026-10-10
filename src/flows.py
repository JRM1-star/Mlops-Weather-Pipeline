"""Prefect orchestration.

  python -m src.flows run     # one full training run
  python -m src.flows drift   # ingest -> drift check -> retrain only if drift
  python -m src.flows serve   # long-running: weekly retrain + daily drift check
"""
import sys

from prefect import flow, serve, task

from src import drift, evaluate_promote, features, ingest, train, validate
from src.config import load_params


@task(retries=2, retry_delay_seconds=30)
def t_ingest(p):
    return ingest.run(p)


@task
def t_validate(p):
    return validate.run(p)


@task
def t_features(p):
    return features.run(p)


@task
def t_train(p):
    return train.run(p)


@task
def t_promote(p):
    return evaluate_promote.run(p)


@task
def t_drift(p):
    return drift.run(p)


@flow(name="training-pipeline", log_prints=True)
def training_pipeline(refresh_data: bool = True):
    p = load_params()
    if refresh_data:
        t_ingest(p)
    t_validate(p)
    t_features(p)
    t_train(p)
    return t_promote(p)


@flow(name="drift-check-and-retrain", log_prints=True)
def drift_check_and_retrain():
    p = load_params()
    t_ingest(p)
    result = t_drift(p)
    if result["drift_detected"]:
        print(f"drift in {result['flagged']} -> retraining")
        return training_pipeline(refresh_data=False)
    print("no drift, skipping retrain")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        training_pipeline()
    elif cmd == "drift":
        drift_check_and_retrain()
    elif cmd == "serve":
        serve(
            training_pipeline.to_deployment(name="weekly-train", cron="0 3 * * 0"),
            drift_check_and_retrain.to_deployment(name="daily-drift-check", cron="0 4 * * *"),
        )
    else:
        sys.exit(f"unknown command: {cmd}")
