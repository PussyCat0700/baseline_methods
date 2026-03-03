"""
Solar-MLP - Solar-specific MLP (Short-term Solar)

MLP optimized for solar power forecasting with solar-specific features.
"""

class Solar_MLP:
    def __init__(self, hidden_dim=512, num_layers=4):
        self.feature_extractor = Sequential([
            Linear(12, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        ])
        self.predictor = Sequential([
            Linear(hidden_dim*120 + 480, hidden_dim),
            ReLU(),
            Dropout(0.3),
            *[Linear(hidden_dim, hidden_dim), ReLU(), Dropout(0.3)] * (num_layers-1),
            Linear(hidden_dim, 480)
        ])

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 12]
            weather_future: [B, 120, 12]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Extract solar-specific features (radiation, cloud, temp)
        solar_features = self.feature_extractor(weather_future)  # [B, 120, hidden_dim]

        # Concatenate with power history
        x = concat([
            flatten(solar_features),
            power_past
        ])  # [B, hidden_dim*120 + 480]

        return self.predictor(x)  # [B, 480]
