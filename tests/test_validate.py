from src.ingest import make_synthetic
from src.validate import check


def test_clean_data_passes(params):
    df = make_synthetic("2018-01-01", "2023-12-31")
    assert check(df, params) == []


def test_impossible_temperature_fails(params):
    df = make_synthetic("2018-01-01", "2023-12-31")
    df.loc[10, "temperature_2m_max"] = 999
    assert any("schema" in e for e in check(df, params))


def test_gap_and_nulls_fail(params):
    df = make_synthetic("2018-01-01", "2023-12-31").drop(index=range(50, 55))
    assert any("gap" in e for e in check(df, params))
    df = make_synthetic("2018-01-01", "2023-12-31")
    df.loc[3, "precipitation_sum"] = None
    assert check(df, params)
