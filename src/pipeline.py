"""Plain-python runner (used by CI). src/flows.py wraps the same steps in Prefect."""
import json

from src import evaluate_promote, features, ingest, train, validate
from src.config import load_params


def run_all(params=None):
    params = params or load_params()
    ingest.run(params)
    validate.run(params)
    features.run(params)
    train.run(params)
    return evaluate_promote.run(params)


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2, default=str))
