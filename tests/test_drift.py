import pytest

from src.drift import detect, psi
from src.ingest import make_synthetic


def test_psi_identical_is_zero():
    import numpy as np
    x = np.random.default_rng(0).normal(size=2000)
    assert psi(x, x) < 1e-6


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_no_drift_on_stationary_data(params, seed):
    df = make_synthetic("2018-01-01", "2023-12-31", seed=seed)
    assert not detect(df, params)["drift_detected"]


def test_detects_temperature_shift(params):
    df = make_synthetic("2018-01-01", "2023-12-31", drift_after="2023-11-01", shift=4.0)
    res = detect(df, params)
    assert res["drift_detected"]
    assert "temperature_2m_max" in res["flagged"]
