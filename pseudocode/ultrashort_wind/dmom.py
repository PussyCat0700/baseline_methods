"""
DMOM - Decomposition-based Multi-Objective Model (Ultrashort Wind)

Uses wavelet decomposition for multi-scale ultrashort-term forecasting.
"""

class DMOM:
    def __init__(self, wavelet='db4', levels=3, hidden_dim=128):
        self.wavelet = WaveletTransform(wavelet, levels)
        self.component_models = [
            LSTM(1, hidden_dim, num_layers=2)
            for _ in range(levels + 1)  # +1 for approximation
        ]
        self.fc = Linear(hidden_dim * (levels + 1), 480)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 15]
            weather_future: [B, 120, 15]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Wavelet decomposition
        components = self.wavelet.decompose(power_past)  # List of [B, T_i]

        # Model each component
        features = []
        for model, component in zip(self.component_models, components):
            _, (h_n, _) = model(component.unsqueeze(-1))  # [1, B, hidden_dim]
            features.append(h_n[-1])  # [B, hidden_dim]

        # Concatenate and predict
        x = concat(features, dim=-1)  # [B, hidden_dim * (levels+1)]
        return self.fc(x)  # [B, 480]
