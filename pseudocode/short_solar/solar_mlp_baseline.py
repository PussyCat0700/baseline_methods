"""
Baseline Method: Solar-MLP
Task: Short-term Solar Power Forecasting

MLP optimized for solar power forecasting with solar-specific features.

Tested Configurations:
    Config 1: hidden_dim=512, num_hidden_layers=3, dropout=0.3, time_encoding_dim=32
    Config 2: hidden_dim=384, num_hidden_layers=2, dropout=0.2, time_encoding_dim=24  ✓ BEST
    Config 3: hidden_dim=256, num_hidden_layers=2, dropout=0.2, time_encoding_dim=16

Best Configuration: Config 2
    - hidden_dim: 384
    - num_hidden_layers: 2
    - dropout: 0.2
    - time_encoding_dim: 24
"""

class Solar_MLP:
    def __init__(self, hidden_dim=384, num_hidden_layers=2, dropout=0.2, time_encoding_dim=24):
        self.time_encoder = Linear(1, time_encoding_dim)
        self.feature_extractor = Sequential([
            Linear(12 + time_encoding_dim, hidden_dim),
            ReLU(),
            Dropout(dropout),
            *[Linear(hidden_dim, hidden_dim), ReLU(), Dropout(dropout)] * (num_hidden_layers - 1)
        ])
        self.predictor = Sequential([
            Linear(hidden_dim * 120 + 480, hidden_dim),
            ReLU(),
            Dropout(dropout),
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
        # Time encoding
        time_feat = self.time_encoder(time_index)  # [B, 120, time_encoding_dim]

        # Extract solar-specific features
        weather_with_time = concat([weather_future, time_feat], dim=-1)  # [B, 120, 12+time_encoding_dim]
        solar_features = self.feature_extractor(weather_with_time)  # [B, 120, hidden_dim]

        # Concatenate with power history
        x = concat([
            flatten(solar_features),
            power_past
        ])  # [B, hidden_dim*120 + 480]

        return self.predictor(x)  # [B, 480]
