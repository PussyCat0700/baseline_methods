"""Persistence formulations used in the manuscript."""

from __future__ import annotations

import numpy as np
import pandas as pd


STEP = pd.Timedelta(minutes=15)


def _ultra_short(
    power: pd.Series,
    technology: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    true_parts: list[np.ndarray] = []
    forecast_parts: list[np.ndarray] = []
    for issue_time in pd.date_range(start - STEP, end - pd.Timedelta(hours=4), freq="4h"):
        targets = pd.date_range(issue_time + STEP, periods=16, freq=STEP)
        targets = targets[(targets >= start) & (targets <= end)]
        truth = power.reindex(targets).to_numpy(float)
        if technology == "wind":
            forecast = np.full(
                len(targets), float(power.get(issue_time, np.nan))
            )
        else:
            forecast = power.reindex(
                targets - pd.Timedelta(hours=24)
            ).to_numpy(float)
        true_parts.append(truth)
        forecast_parts.append(forecast)
    return np.concatenate(true_parts), np.concatenate(forecast_parts)


def _daily(
    power: pd.Series,
    technology: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    task: str,
):
    lead_days, observation_lag = (
        (1, pd.Timedelta(hours=48))
        if task == "short"
        else (4, pd.Timedelta(hours=120))
    )
    true_parts: list[np.ndarray] = []
    forecast_parts: list[np.ndarray] = []
    for target_day in pd.date_range(start.floor("D"), end.floor("D"), freq="1D"):
        targets = pd.date_range(target_day, periods=96, freq=STEP)
        targets = targets[(targets >= start) & (targets <= end)]
        truth = power.reindex(targets).to_numpy(float)
        if technology == "wind":
            issue_time = (
                target_day
                - pd.Timedelta(days=lead_days)
                + pd.Timedelta(hours=8)
            )
            forecast = np.full(
                len(targets), float(power.get(issue_time, np.nan))
            )
        else:
            forecast = power.reindex(
                targets - observation_lag
            ).to_numpy(float)
        true_parts.append(truth)
        forecast_parts.append(forecast)
    return np.concatenate(true_parts), np.concatenate(forecast_parts)
