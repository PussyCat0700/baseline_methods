"""Metric definitions and site-first aggregation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


METRICS = ("MAE_percent", "RMSE_percent", "R2", "Corr")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    forecast = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(truth) & np.isfinite(forecast)
    truth = truth[valid]
    forecast = forecast[valid]
    if not truth.size:
        return {"n": 0, "MAE_percent": np.nan, "RMSE_percent": np.nan, "R2": np.nan, "Corr": np.nan}
    error = forecast - truth
    sse = float(np.square(error).sum())
    centered = truth - float(truth.mean())
    sst = float(np.square(centered).sum())
    corr = np.nan
    if truth.size > 1 and float(truth.std()) > 0 and float(forecast.std()) > 0:
        corr = float(np.corrcoef(truth, forecast)[0, 1])
    return {
        "n": int(truth.size),
        "MAE_percent": float(np.abs(error).mean() * 100.0),
        "RMSE_percent": float(math.sqrt(np.square(error).mean()) * 100.0),
        "R2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "Corr": corr,
    }


def site_first_summary(site_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"station_id", "technology", "task", *METRICS}
    missing = sorted(required - set(site_metrics.columns))
    if missing:
        raise ValueError(f"site metric columns missing: {missing}")
    rows: list[dict[str, object]] = []
    for (technology, task), frame in site_metrics.groupby(["technology", "task"], sort=True):
        row: dict[str, object] = {
            "technology": technology,
            "task": task,
            "site_count": int(frame["station_id"].nunique()),
        }
        for metric in METRICS:
            row[metric] = float(frame[metric].mean(skipna=True))
            row[f"valid_{metric}_sites"] = int(frame[metric].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)
