"""Fail the pipeline early if the raw data is bad."""
import json
from datetime import date
from pathlib import Path

import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:  # older pandera
    import pandera as pa

from src.config import load_params


class ValidationError(Exception):
    pass


def _num(lo, hi):
    return pa.Column(float, pa.Check.in_range(lo, hi), nullable=False, coerce=True)


def build_schema():
    return pa.DataFrameSchema(
        {
            "date": pa.Column(pa.DateTime, unique=True, nullable=False),
            "temperature_2m_max": _num(-60, 60),
            "temperature_2m_min": _num(-60, 60),
            "precipitation_sum": _num(0, 1000),
            "wind_speed_10m_max": _num(0, 300),
            "shortwave_radiation_sum": _num(0, 60),
        },
        checks=pa.Check(
            lambda d: bool((d["temperature_2m_min"] <= d["temperature_2m_max"]).all()),
            error="temperature_2m_min > temperature_2m_max",
        ),
        strict=False,
    )


def check(df, params):
    errors = []
    try:
        build_schema().validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        errors.append(f"schema: {e.failure_cases.head(10).to_dict('records')}")
    except pa.errors.SchemaError as e:
        errors.append(f"schema: {e}")

    v = params["validation"]
    if len(df) < v["min_rows"]:
        errors.append(f"only {len(df)} rows, need >= {v['min_rows']}")
    gaps = df["date"].sort_values().diff().dt.days.dropna()
    if (gaps > 1).any():
        errors.append(f"{int((gaps > 1).sum())} gap(s) in the daily series")
    if params["data"]["source"] != "synthetic" and len(df):
        stale = (date.today() - df["date"].max().date()).days
        if stale > v["max_staleness_days"]:
            errors.append(f"data is {stale} days old (max {v['max_staleness_days']})")
    return errors


def run(params):
    df = pd.read_csv(params["paths"]["raw"], parse_dates=["date"])
    errors = check(df, params)
    out = Path(params["paths"]["reports_dir"]) / "validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"passed": not errors, "rows": len(df), "errors": errors}, indent=2))
    if errors:
        raise ValidationError("; ".join(errors))
    print(f"validate: ok ({len(df)} rows)")
    return {"passed": True, "rows": len(df)}


if __name__ == "__main__":
    run(load_params())
