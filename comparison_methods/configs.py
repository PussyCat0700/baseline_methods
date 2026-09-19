"""Paper-aligned comparison-method search spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MethodSpec:
    name: str
    display_name: str
    technology: str
    horizon: str
    family: str
    configs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    selected: int


def _spec(
    name: str,
    display_name: str,
    technology: str,
    horizon: str,
    family: str,
    configs: list[dict[str, Any]],
    selected: int,
) -> MethodSpec:
    if len(configs) != 3 or selected not in (1, 2, 3):
        raise ValueError(f"invalid search specification for {name}")
    return MethodSpec(
        name=name,
        display_name=display_name,
        technology=technology,
        horizon=horizon,
        family=family,
        configs=tuple(configs),
        selected=selected,
    )


METHOD_SPECS: dict[str, MethodSpec] = {
    "ffnn": _spec(
        "ffnn", "FFNN", "wind", "short_medium", "mlp",
        [
            {"hidden_dim": 512, "num_layers": 12, "dropout": 0.3},
            {"hidden_dim": 256, "num_layers": 8, "dropout": 0.3},
            {"hidden_dim": 128, "num_layers": 4, "dropout": 0.3},
        ], 3,
    ),
    "gefcom12": _spec(
        "gefcom12", "GEFCom12", "wind", "short_medium", "ensemble",
        [
            {"n_estimators": 5, "num_leaves": 16},
            {"n_estimators": 10, "num_leaves": 32},
            {"n_estimators": 20, "num_leaves": 64},
        ], 3,
    ),
    "heftcom24": _spec(
        "heftcom24", "HEFTCom24", "wind", "short_medium", "ensemble",
        [
            {"n_estimators": 100, "num_leaves": 200},
            {"n_estimators": 200, "num_leaves": 400},
            {"n_estimators": 400, "num_leaves": 700},
        ], 2,
    ),
    "cnn_rbfnn": _spec(
        "cnn_rbfnn", "CNN-RBFNN", "wind", "short_medium", "rbf",
        [
            {"hidden_dim": 64, "num_layers": 1},
            {"hidden_dim": 128, "num_layers": 2},
            {"hidden_dim": 256, "num_layers": 4},
        ], 3,
    ),
    "hpe": _spec(
        "hpe", "HPE", "wind", "short_medium", "ensemble",
        [
            {"n_estimators": 50, "max_features": 32},
            {"n_estimators": 100, "max_features": 64},
            {"n_estimators": 150, "max_features": 128},
        ], 3,
    ),
    "ann": _spec(
        "ann", "ANN", "solar", "short_medium", "mlp",
        [
            {"hidden_dim": 512, "num_layers": 3, "dropout": 0.3},
            {"hidden_dim": 384, "num_layers": 2, "dropout": 0.2},
            {"hidden_dim": 256, "num_layers": 2, "dropout": 0.2},
        ], 2,
    ),
    "crossvivit": _spec(
        "crossvivit", "CrossViViT", "solar", "short_medium", "attention",
        [
            {"d_model": 384, "depth": 16, "cross_depth": 4, "decoder_depth": 4, "heads": 12},
            {"d_model": 256, "depth": 12, "cross_depth": 3, "decoder_depth": 3, "heads": 8},
            {"d_model": 128, "depth": 8, "cross_depth": 2, "decoder_depth": 2, "heads": 6},
        ], 3,
    ),
    "pvtransnet": _spec(
        "pvtransnet", "PVTransNet", "solar", "short_medium", "attention",
        [
            {"d_model": 128, "depth": 1, "heads": 2, "head_dim": 64},
            {"d_model": 256, "depth": 4, "heads": 4, "head_dim": 128},
            {"d_model": 512, "depth": 12, "heads": 8, "head_dim": 128},
        ], 3,
    ),
    "cnn_lstm_solar": _spec(
        "cnn_lstm_solar", "CNN-LSTM", "solar", "short_medium", "attention",
        [
            {"d_model": 128, "depth": 2, "dropout": 0.1},
            {"d_model": 256, "depth": 3, "dropout": 0.1},
            {"d_model": 128, "depth": 4, "dropout": 0.2},
        ], 2,
    ),
    "fusionsf": _spec(
        "fusionsf", "FusionSF", "solar", "short_medium", "attention",
        [
            {"d_model": 128, "heads": 2, "depth": 3, "look_back_steps": 48, "dropout": 0.1},
            {"d_model": 256, "heads": 4, "depth": 6, "look_back_steps": 96, "dropout": 0.1},
            {"d_model": 512, "heads": 8, "depth": 12, "look_back_steps": 96, "dropout": 0.1},
        ], 2,
    ),
    "tft": _spec(
        "tft", "TFT", "wind", "ultra_short", "tft",
        [
            {"d_model": 112, "num_layers": 9, "heads": 7, "dropout": 0.1},
            {"d_model": 56, "num_layers": 7, "heads": 7, "dropout": 0.1},
            {"d_model": 49, "num_layers": 5, "heads": 7, "dropout": 0.1},
        ], 1,
    ),
    "hbola": _spec(
        "hbola", "HBOLA", "wind", "ultra_short", "recurrent",
        [
            {"hidden_size": 64, "num_layers": 5, "dropout": 0.1},
            {"hidden_size": 96, "num_layers": 10, "dropout": 0.1},
            {"hidden_size": 128, "num_layers": 20, "dropout": 0.15},
        ], 2,
    ),
    "lssvm_rbfnn": _spec(
        "lssvm_rbfnn", "LSSVM-RBFNN", "wind", "ultra_short", "rbf",
        [
            {"embed_dim": 64, "hidden_dim": 128, "num_layers": 2, "num_prototypes": 64},
            {"embed_dim": 128, "hidden_dim": 256, "num_layers": 2, "num_prototypes": 128},
            {"embed_dim": 128, "hidden_dim": 384, "num_layers": 3, "num_prototypes": 256},
        ], 3,
    ),
    "dmom": _spec(
        "dmom", "DMOM", "wind", "ultra_short", "frequency",
        [
            {"hidden_dim": 32, "alpha": 0.3, "dropout": 0.05},
            {"hidden_dim": 48, "alpha": 0.5, "dropout": 0.10},
            {"hidden_dim": 64, "alpha": 0.7, "dropout": 0.15},
        ], 3,
    ),
    "scvar": _spec(
        "scvar", "SC-VAR", "wind", "ultra_short", "frequency",
        [
            {"hidden_dim": 192, "num_layers": 2, "dropout": 0.15},
            {"hidden_dim": 320, "num_layers": 3, "dropout": 0.20},
            {"hidden_dim": 512, "num_layers": 4, "dropout": 0.25},
        ], 3,
    ),
    "wpd_lstm": _spec(
        "wpd_lstm", "WPD-LSTM", "solar", "ultra_short", "frequency",
        [
            {"hidden_dim": 128, "dropout": 0.1},
            {"hidden_dim": 256, "dropout": 0.1},
            {"hidden_dim": 256, "dropout": 0.2},
        ], 2,
    ),
    "atcn": _spec(
        "atcn", "ATCN", "solar", "ultra_short", "frequency",
        [
            {"hidden_dim": 48, "kernel_size": 5, "dropout": 0.1},
            {"hidden_dim": 64, "kernel_size": 5, "dropout": 0.1},
            {"hidden_dim": 80, "kernel_size": 7, "dropout": 0.2},
        ], 1,
    ),
    "lstm_gcn_mlp": _spec(
        "lstm_gcn_mlp", "LSTM-GCN-MLP", "solar", "ultra_short", "recurrent",
        [
            {"hidden_size": 64, "num_layers": 2, "dropout": 0.1},
            {"hidden_size": 96, "num_layers": 3, "dropout": 0.1},
            {"hidden_size": 128, "num_layers": 3, "dropout": 0.15},
        ], 3,
    ),
    "narx_ga": _spec(
        "narx_ga", "NARX-GA", "solar", "ultra_short", "frequency",
        [
            {"hidden_dim": 32, "fc_hidden_dim": 128},
            {"hidden_dim": 64, "fc_hidden_dim": 256},
            {"hidden_dim": 96, "fc_hidden_dim": 256},
        ], 3,
    ),
    "rfs_alo": _spec(
        "rfs_alo", "RFs-ALO", "solar", "ultra_short", "ensemble",
        [
            {"n_estimators": 500, "num_leaves": 10},
            {"n_estimators": 1000, "num_leaves": 5},
            {"n_estimators": 300, "num_leaves": 20},
        ], 2,
    ),
}


ALIASES = {
    "cnn-rbfnn": "cnn_rbfnn",
    "cnn-lstm-solar": "cnn_lstm_solar",
    "lssvm-rbfnn": "lssvm_rbfnn",
    "lstm-gcn-mlp": "lstm_gcn_mlp",
    "rfs-alo": "rfs_alo",
    "sc-var": "scvar",
}


def get_method_spec(name: str) -> MethodSpec:
    key = ALIASES.get(name.lower(), name.lower())
    try:
        return METHOD_SPECS[key]
    except KeyError as exc:
        available = ", ".join(sorted(METHOD_SPECS))
        raise ValueError(f"unknown method {name!r}; choose from {available}") from exc
