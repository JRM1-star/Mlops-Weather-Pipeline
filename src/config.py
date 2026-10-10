import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params(path=None, overrides=None):
    """Load params.yaml. Env vars and dotted overrides ("data.source") win."""
    p = yaml.safe_load(Path(path or ROOT / "params.yaml").read_text())
    if os.getenv("DATA_SOURCE"):
        p["data"]["source"] = os.environ["DATA_SOURCE"]
    if os.getenv("MLFLOW_TRACKING_URI"):
        p["mlflow"]["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    for key, value in (overrides or {}).items():
        section, name = key.split(".", 1)
        p[section][name] = value
    return p
