"""Seasonality-aware drift check: compare recent anomalies against historical anomalies via PSI."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_params


def psi(ref, cur, bins=5):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def anomalies(df, col, ref_mask):
    """Value minus smoothed day-of-year climatology, so the seasonal cycle is not flagged as drift."""
    doy = df["date"].dt.dayofyear
    clim = df[ref_mask].groupby(doy[ref_mask])[col].mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    pad = 15
    ext = pd.concat([clim.iloc[-pad:], clim, clim.iloc[:pad]])
    smooth = pd.Series(ext.rolling(15, center=True).mean().values[pad:-pad], index=clim.index)
    return df[col] - doy.map(smooth)


def detect(df, params):
    d = params["drift"]
    df = df.sort_values("date").reset_index(drop=True)
    recent = df.index >= len(df) - d["window_days"]
    report = {}
    for col in d["monitor"]:
        a = anomalies(df, col, ~recent)
        ref, cur = a[~recent].dropna().values, a[recent].dropna().values
        report[col] = {
            "psi": psi(ref, cur, d["bins"]),
            "ks_p": float(stats.ks_2samp(ref, cur).pvalue),
            "mean_shift": float(cur.mean() - ref.mean()),
        }
    flagged = [c for c, r in report.items() if r["psi"] > d["psi_threshold"]]
    return {"drift_detected": bool(flagged), "flagged": flagged, "features": report,
            "window_end": str(df["date"].max().date())}


def run(params):
    df = pd.read_csv(params["paths"]["raw"], parse_dates=["date"])
    result = detect(df, params)
    out = Path(params["paths"]["reports_dir"]) / "drift.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("drift:", "DETECTED " + str(result["flagged"]) if result["drift_detected"] else "none")
    return result


if __name__ == "__main__":
    run(load_params())
