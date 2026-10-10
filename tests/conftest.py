import pytest

from src.config import load_params


@pytest.fixture
def params(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", (tmp_path / "art").as_uri())
    return load_params(overrides={
        "paths.raw": str(tmp_path / "raw.csv"),
        "paths.processed_dir": str(tmp_path / "processed"),
        "paths.reports_dir": str(tmp_path / "reports"),
        "data.source": "synthetic",
        "data.start_date": "2018-01-01",
        "data.end_date": "2023-12-31",
        "mlflow.tracking_uri": f"sqlite:///{tmp_path}/mlflow.db",
    })
