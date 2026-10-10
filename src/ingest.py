"""Incrementally pull daily weather into data/raw/weather.csv."""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import load_params

API = "https://archive-api.open-meteo.com/v1/archive"
DAILY = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
]


def fetch_open_meteo(lat, lon, start, end):
    resp = requests.get(
        API,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "daily": ",".join(DAILY),
            "timezone": "auto",
        },
        timeout=60,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["daily"]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def make_synthetic(start, end, seed=0, drift_after=None, shift=0.0):
    """Seasonal + AR(1) weather. `shift` degrees are added after `drift_after` to simulate drift."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    doy = dates.dayofyear.values
    season = np.sin(2 * np.pi * (doy - 100) / 365.25)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 1.2)
    tmax = 30 + 5 * season + noise
    if drift_after is not None:
        tmax = tmax + np.where(dates >= pd.Timestamp(drift_after), shift, 0.0)
    return pd.DataFrame(
        {
            "date": dates,
            "temperature_2m_max": tmax,
            "temperature_2m_min": tmax - 8 + rng.normal(0, 1, n),
            "precipitation_sum": rng.gamma(0.3, 6, n) * (season < 0.3),
            "wind_speed_10m_max": np.abs(15 + 3 * rng.normal(size=n)),
            "shortwave_radiation_sum": np.clip(20 + 5 * season + rng.normal(0, 2, n), 0, None),
        }
    )


def run(params):
    d = params["data"]
    raw = Path(params["paths"]["raw"])
    end = d.get("end_date") or (date.today() - timedelta(days=5)).isoformat()
    start = d["start_date"]

    existing = pd.read_csv(raw, parse_dates=["date"]) if raw.exists() else None
    if existing is not None and len(existing):
        # re-fetch the last week: recent reanalysis values get revised
        resume = (existing["date"].max() - timedelta(days=7)).date().isoformat()
        start = max(start, resume)

    if start > end:
        new = pd.DataFrame(columns=["date"] + DAILY)
    elif d["source"] == "synthetic":
        full = make_synthetic(d["start_date"], end)
        new = full[full["date"] >= start]
    else:
        new = fetch_open_meteo(d["latitude"], d["longitude"], start, end)
        new = new.dropna(subset=["temperature_2m_max"])  # latest days may not be published yet

    df = pd.concat([x for x in (existing, new) if x is not None and len(x)])
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    raw.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw, index=False)
    info = {"rows": len(df), "first": str(df["date"].min().date()), "last": str(df["date"].max().date())}
    print("ingest:", info)
    return info


if __name__ == "__main__":
    run(load_params())
