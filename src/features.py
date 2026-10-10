"""Lag/rolling/seasonal features. Target = next day's max temperature."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_params

SRC = "temperature_2m_max"
BASE = [SRC, "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max", "shortwave_radiation_sum"]


def build_features(df, lags, windows,horizon=1):
    df = df.sort_values("date").reset_index(drop=True)
    out = df[["date"] + BASE].copy()
    for lag in lags:
        out[f"tmax_lag{lag}"] = df[SRC].shift(lag)
    for w in windows:
        out[f"tmax_roll{w}"] = df[SRC].rolling(w).mean()
        out[f"precip_roll{w}"] = df["precipitation_sum"].rolling(w).mean()
    doy = df["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["target"] = df[SRC].shift(-horizon)
    return out.dropna().reset_index(drop=True)


def feature_columns(df):
    return [c for c in df.columns if c not in ("date", "target")]


def run(params):
    f = params["features"]
    df = pd.read_csv(params["paths"]["raw"], parse_dates=["date"])
    feats = build_features(df, f["lags"], f["rolling_windows"], f.get("horizon", 1))
    train, test = feats.iloc[: -f["test_days"]], feats.iloc[-f["test_days"] :]  # time-ordered split
    out = Path(params["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(f"features: train={len(train)} test={len(test)}")
    return {"train": len(train), "test": len(test)}


if __name__ == "__main__":
    run(load_params())
