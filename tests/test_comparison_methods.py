import numpy as np
import pandas as pd
import pytest
import torch

from comparison_methods.configs import METHOD_SPECS
from comparison_methods.metrics import calculate_metrics, site_first_summary
from comparison_methods.models import build_model
from comparison_methods.persistence import _ultra_short


def test_inventory_and_search_spaces():
    assert len(METHOD_SPECS) == 20
    assert all(len(spec.configs) == 3 for spec in METHOD_SPECS.values())
    assert sum(spec.technology == "wind" for spec in METHOD_SPECS.values()) == 10
    assert sum(spec.technology == "solar" for spec in METHOD_SPECS.values()) == 10


@pytest.mark.parametrize("method", sorted(METHOD_SPECS))
def test_selected_model_interface(method):
    spec = METHOD_SPECS[method]
    feature_count = 15 if spec.technology == "wind" else 12
    model = build_model(method)
    output = model(
        torch.randn(1, 120, feature_count),
        torch.randn(1, 120, feature_count),
        torch.randn(1, 480),
    )
    assert output.shape == (1, 480)
    assert torch.isfinite(output).all()


def test_metrics_and_site_first_aggregation():
    first = calculate_metrics(np.array([0.0, 1.0]), np.array([0.0, 0.5]))
    second = calculate_metrics(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    frame = pd.DataFrame(
        [
            {"station_id": 0, "technology": "wind", "task": "ultra_short", **first},
            {"station_id": 1, "technology": "wind", "task": "ultra_short", **second},
        ]
    )
    summary = site_first_summary(frame).iloc[0]
    assert summary["site_count"] == 2
    assert summary["RMSE_percent"] == pytest.approx((first["RMSE_percent"] + second["RMSE_percent"]) / 2)


def test_wind_persistence_repeats_issue_time_observation():
    index = pd.date_range("2024-12-31", "2025-01-02", freq="15min")
    power = pd.Series(np.arange(len(index), dtype=float), index=index)
    start = pd.Timestamp("2025-01-01 00:00")
    end = pd.Timestamp("2025-01-01 23:45")

    _, issue_time_forecast = _ultra_short(
        power, "wind", start, end
    )

    assert np.unique(issue_time_forecast[:16]).size == 1
    assert np.all(issue_time_forecast[:16] == power.loc[start - pd.Timedelta(minutes=15)])


def test_frozen_selected_configuration_indices():
    # Boldface selections in Supplementary Table 4.
    expected = {
        "ffnn": 3,
        "gefcom12": 3,
        "heftcom24": 2,
        "cnn_rbfnn": 3,
        "hpe": 3,
        "ann": 2,
        "crossvivit": 3,
        "pvtransnet": 3,
        "cnn_lstm_solar": 2,
        "fusionsf": 2,
        "tft": 1,
        "hbola": 2,
        "lssvm_rbfnn": 3,
        "dmom": 3,
        "scvar": 3,
        "wpd_lstm": 2,
        "atcn": 1,
        "lstm_gcn_mlp": 3,
        "narx_ga": 3,
        "rfs_alo": 2,
    }
    assert {name: spec.selected for name, spec in METHOD_SPECS.items()} == expected
