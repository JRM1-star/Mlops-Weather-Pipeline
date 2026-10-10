from src.features import build_features, feature_columns
from src.ingest import make_synthetic


def test_no_future_leakage():
    df = make_synthetic("2020-01-01", "2020-12-31")
    base = build_features(df, [1, 2], [3, 7])
    k = 200
    changed = df.copy()
    changed.loc[k + 1 :, df.columns.drop("date")] += 100  # corrupt everything after row k
    mod = build_features(changed, [1, 2], [3, 7])
    cols = feature_columns(base)
    cutoff = base["date"].iloc[100]
    a, b = base[base["date"] <= cutoff], mod[mod["date"] <= cutoff]
    assert a[cols].equals(b[cols])


def test_target_is_next_day():
    df = make_synthetic("2020-01-01", "2020-03-01")
    f = build_features(df, [1], [3]).set_index("date")
    d = df.set_index("date")
    day = f.index[10]
    assert f.loc[day, "target"] == d["temperature_2m_max"].shift(-1).loc[day]
