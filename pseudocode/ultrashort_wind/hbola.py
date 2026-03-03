"""
HBOLA - Hybrid LSTM with Online Learning (Ultrashort Wind)

LSTM with online adaptation for ultrashort-term wind forecasting.
"""

class HBOLA:
    def __init__(self, hidden_dim=256, num_layers=2):
        self.lstm = LSTM(15+1, hidden_dim, num_layers=num_layers)
        self.online_layer = AdaptiveLinear(hidden_dim, 480)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 15]
            weather_future: [B, 120, 15]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Resample power to hourly
        power_hourly = resample(power_past, 120)  # [B, 120, 1]

        # Concatenate weather and power
        x = concat([weather_future, power_hourly], dim=-1)  # [B, 120, 16]

        # LSTM encoding
        _, (h_n, _) = self.lstm(x)  # h_n: [num_layers, B, hidden_dim]

        # Online adaptive prediction
        return self.online_layer(h_n[-1])  # [B, 480]
