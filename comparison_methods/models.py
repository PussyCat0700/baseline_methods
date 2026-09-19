"""Executable implementations with a common forecasting interface.

Every model accepts historical weather, forecast weather and historical power
and returns 480 quarter-hourly values. The implementations retain the main
computational structure of each cited method while using one shared data and
training interface so that the comparison protocol can be rerun consistently.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .configs import MethodSpec, get_method_spec


FORECAST_HOURS = 120
FORECAST_STEPS = 480
POWER_HISTORY = 480
WIND_FEATURES = 15
SOLAR_FEATURES = 12


def _weather_dim(technology: str) -> int:
    return WIND_FEATURES if technology == "wind" else SOLAR_FEATURES


def _quarter_hour_output(hourly: torch.Tensor, projection: nn.Module) -> torch.Tensor:
    batch, hours, _ = hourly.shape
    return projection(hourly).reshape(batch, hours * 4)


def _sinusoidal(length: int, dimension: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(dimension, 1))
    )
    encoding = torch.zeros(length, dimension, device=device)
    encoding[:, 0::2] = torch.sin(position * scale)
    if dimension > 1:
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
    return encoding


class CommonForecastModel(nn.Module):
    def __init__(self, spec: MethodSpec, config: dict[str, Any]):
        super().__init__()
        self.spec = spec
        self.config = dict(config)
        self.weather_dim = _weather_dim(spec.technology)

    def _validate(
        self,
        weather_past: torch.Tensor,
        weather_future: torch.Tensor,
        power_past: torch.Tensor,
    ) -> None:
        expected_weather = (FORECAST_HOURS, self.weather_dim)
        if tuple(weather_past.shape[1:]) != expected_weather:
            raise ValueError(f"weather_past must have trailing shape {expected_weather}")
        if tuple(weather_future.shape[1:]) != expected_weather:
            raise ValueError(f"weather_future must have trailing shape {expected_weather}")
        if tuple(power_past.shape[1:]) != (POWER_HISTORY,):
            raise ValueError("power_past must have trailing shape (480,)")


class MLPForecast(CommonForecastModel):
    def __init__(self, spec: MethodSpec, config: dict[str, Any]):
        super().__init__(spec, config)
        hidden = int(config["hidden_dim"])
        layers = int(config.get("num_layers", 2))
        dropout = float(config.get("dropout", 0.2))
        input_dim = 30 * self.weather_dim + 120
        blocks: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(max(layers - 1, 0)):
            blocks.extend([nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)])
        blocks.append(nn.Linear(hidden, FORECAST_STEPS))
        self.network = nn.Sequential(*blocks)

    def forward(self, weather_past, weather_future, power_past):
        self._validate(weather_past, weather_future, power_past)
        weather = F.adaptive_avg_pool1d(weather_future.transpose(1, 2), 30).flatten(1)
        power = F.adaptive_avg_pool1d(power_past.unsqueeze(1), 120).flatten(1)
        return self.network(torch.cat([weather, power], dim=1))


class RecurrentForecast(CommonForecastModel):
    def __init__(self, spec: MethodSpec, config: dict[str, Any]):
        super().__init__(spec, config)
        hidden = int(config.get("hidden_size", config.get("hidden_dim", 128)))
        layers = int(config.get("num_layers", 2))
        dropout = float(config.get("dropout", 0.1)) if layers > 1 else 0.0
        self.power_encoder = nn.GRU(4, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.weather_encoder = nn.GRU(
            self.weather_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(float(config.get("dropout", 0.1))),
        )
        self.output = nn.Linear(hidden, 4)

    def forward(self, weather_past, weather_future, power_past):
        self._validate(weather_past, weather_future, power_past)
        power_hourly = power_past.reshape(power_past.shape[0], FORECAST_HOURS, 4)
        _, power_state = self.power_encoder(power_hourly)
        weather_state, _ = self.weather_encoder(weather_future)
        context = power_state[-1].unsqueeze(1).expand(-1, FORECAST_HOURS, -1)
        fused = self.fusion(torch.cat([weather_state, context], dim=-1))
        return _quarter_hour_output(fused, self.output)


class AttentionForecast(CommonForecastModel):
    def __init__(self, spec: MethodSpec, config: dict[str, Any]):
        super().__init__(spec, config)
        dimension = int(config.get("d_model", 128))
        heads = int(config.get("heads", 4))
        while dimension % heads:
            heads -= 1
        depth = int(config.get("depth", 2))
        dropout = float(config.get("dropout", 0.1))
        self.weather_projection = nn.Linear(self.weather_dim, dimension)
        self.power_projection = nn.Linear(4, dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=max(heads, 1),
            dim_feedforward=dimension * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.cross_attention = nn.MultiheadAttention(
            dimension,
            max(heads, 1),
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dimension)
        self.output = nn.Linear(dimension, 4)

    def forward(self, weather_past, weather_future, power_past):
        self._validate(weather_past, weather_future, power_past)
        power = power_past.reshape(power_past.shape[0], FORECAST_HOURS, 4)
        memory = self.power_projection(power)
        query = self.weather_projection(weather_future)
        position = _sinusoidal(FORECAST_HOURS, query.shape[-1], query.device)
        query = self.encoder(query + position.unsqueeze(0))
        attended, _ = self.cross_attention(query, memory, memory, need_weights=False)
        return _quarter_hour_output(self.norm(query + attended), self.output)


class FrequencyForecast(CommonForecastModel):
    def __init__(self, spec: MethodSpec, config: dict[str, Any]):
        super().__init__(spec, config)
        hidden = int(config.get("hidden_dim", 64))
        layers = int(config.get("num_layers", 2))
        kernel = int(config.get("kernel_size", 5))
        dropout = float(config.get("dropout", 0.1))
        self.alpha = float(config.get("alpha", 0.5))
        conv: list[nn.Module] = []
        in_channels = 2
        for _ in range(layers):
            conv.extend(
                [
                    nn.Conv1d(in_channels, hidden, kernel, padding=kernel // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = hidden
        self.history_encoder = nn.Sequential(*conv)
        self.weather_projection = nn.Linear(self.weather_dim, hidden)
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, int(config.get("fc_hidden_dim", hidden * 2))),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(config.get("fc_hidden_dim", hidden * 2)), 4),
        )

    def _decompose(self, power: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum = torch.fft.rfft(power, dim=-1)
        cut = max(1, min(spectrum.shape[-1] - 1, int(self.alpha * spectrum.shape[-1])))
        low_spectrum = torch.zeros_like(spectrum)
        low_spectrum[..., :cut] = spectrum[..., :cut]
        low = torch.fft.irfft(low_spectrum, n=power.shape[-1], dim=-1)
        return low, power - low

    def forward(self, weather_past, weather_future, power_past):
        self._validate(weather_past, weather_future, power_past)
        low, high = self._decompose(power_past)
        history = self.history_encoder(torch.stack([low, high], dim=1))
        history = F.adaptive_avg_pool1d(history, FORECAST_HOURS).transpose(1, 2)
        weather = self.weather_projection(weather_future)
        return _quarter_hour_output(torch.cat([history, weather], dim=-1), self.fusion)


class RBFHybridForecast(CommonForecastModel):
    def __init__(self, spec: MethodSpec, config: dict[str, Any]):
        super().__init__(spec, config)
        hidden = int(config.get("hidden_dim", 128))
        embed = int(config.get("embed_dim", 64))
        prototypes = int(config.get("num_prototypes", 64))
        self.gamma = float(config.get("gamma", 0.5))
        self.encoder = nn.Sequential(
            nn.Linear(30 * self.weather_dim + 120, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed),
        )
        self.centres = nn.Parameter(torch.randn(prototypes, embed) * 0.05)
        self.rbf_output = nn.Linear(prototypes, FORECAST_STEPS)
        self.linear_output = nn.Linear(embed, FORECAST_STEPS)

    def forward(self, weather_past, weather_future, power_past):
        self._validate(weather_past, weather_future, power_past)
        weather = F.adaptive_avg_pool1d(weather_future.transpose(1, 2), 30).flatten(1)
        power = F.adaptive_avg_pool1d(power_past.unsqueeze(1), 120).flatten(1)
        embedding = self.encoder(torch.cat([weather, power], dim=1))
        distance = torch.cdist(embedding, self.centres).square()
        rbf = torch.exp(-self.gamma * distance)
        return 0.5 * (self.rbf_output(rbf) + self.linear_output(embedding))


class SoftEnsembleForecast(CommonForecastModel):
    """Vectorized differentiable ensemble for tree-based comparison methods."""

    def __init__(self, spec: MethodSpec, config: dict[str, Any]):
        super().__init__(spec, config)
        estimators = int(config.get("n_estimators", 100))
        embed = int(config.get("embed_dim", 64))
        context = int(config.get("power_ctx_dim", 32))
        hidden = int(config.get("encoder_hidden_dim", 128))
        self.num_leaves = int(config.get("num_leaves", config.get("max_features", 32)))
        self.power_encoder = nn.Sequential(nn.Linear(POWER_HISTORY, hidden), nn.ReLU(), nn.Linear(hidden, context))
        self.feature_encoder = nn.Sequential(nn.Linear(self.weather_dim + context, hidden), nn.ReLU(), nn.Linear(hidden, embed))
        self.thresholds = nn.Parameter(torch.zeros(estimators, embed))
        self.directions = nn.Parameter(torch.randn(estimators, embed) / math.sqrt(embed))
        self.left_values = nn.Parameter(torch.zeros(estimators, 4))
        self.right_values = nn.Parameter(torch.zeros(estimators, 4))

    def forward(self, weather_past, weather_future, power_past):
        self._validate(weather_past, weather_future, power_past)
        context = self.power_encoder(power_past).unsqueeze(1).expand(-1, FORECAST_HOURS, -1)
        features = self.feature_encoder(torch.cat([weather_future, context], dim=-1))
        projection = torch.einsum("bte,ne->btn", features, self.directions)
        threshold = self.thresholds.mean(dim=1).view(1, 1, -1)
        gate = torch.sigmoid(projection - threshold)
        output = torch.einsum("btn,nq->btq", 1.0 - gate, self.left_values)
        output += torch.einsum("btn,nq->btq", gate, self.right_values)
        output /= float(self.directions.shape[0])
        return output.reshape(output.shape[0], FORECAST_STEPS)


class TemporalFusionForecast(CommonForecastModel):
    def __init__(self, spec: MethodSpec, config: dict[str, Any]):
        super().__init__(spec, config)
        dimension = int(config.get("d_model", 56))
        layers = int(config.get("num_layers", 5))
        heads = int(config.get("heads", 7))
        dropout = float(config.get("dropout", 0.1)) if layers > 1 else 0.0
        self.variable_weights = nn.Linear(self.weather_dim, self.weather_dim)
        self.weather_projection = nn.Linear(self.weather_dim, dimension)
        self.power_projection = nn.Linear(4, dimension)
        self.encoder = nn.LSTM(dimension * 2, dimension, layers, batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=float(config.get("dropout", 0.1)), batch_first=True)
        self.gate = nn.Sequential(nn.Linear(dimension, dimension), nn.Sigmoid())
        self.output = nn.Linear(dimension, 4)

    def forward(self, weather_past, weather_future, power_past):
        self._validate(weather_past, weather_future, power_past)
        weights = torch.softmax(self.variable_weights(weather_future), dim=-1)
        weather = self.weather_projection(weather_future * weights)
        power = self.power_projection(power_past.reshape(power_past.shape[0], FORECAST_HOURS, 4))
        encoded, _ = self.encoder(torch.cat([weather, power], dim=-1))
        attended, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        fused = encoded + self.gate(attended) * attended
        return _quarter_hour_output(fused, self.output)


FAMILY_BUILDERS = {
    "mlp": MLPForecast,
    "recurrent": RecurrentForecast,
    "attention": AttentionForecast,
    "frequency": FrequencyForecast,
    "rbf": RBFHybridForecast,
    "ensemble": SoftEnsembleForecast,
    "tft": TemporalFusionForecast,
}


def build_model(method: str, config_index: int | None = None) -> CommonForecastModel:
    spec = get_method_spec(method)
    index = spec.selected if config_index is None else int(config_index)
    if index not in (1, 2, 3):
        raise ValueError("config_index must be 1, 2 or 3")
    model_class = FAMILY_BUILDERS[spec.family]
    return model_class(spec, dict(spec.configs[index - 1]))
