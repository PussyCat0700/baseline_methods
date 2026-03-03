"""
WPD-LSTM - Wavelet Packet Decomposition + LSTM (Ultrashort Solar)

Uses wavelet packet decomposition for multi-resolution ultrashort solar forecasting.
"""

class WPD_LSTM:
    def __init__(self, wavelet='db4', levels=3, hidden_dim=128):
        self.wpd = WaveletPacketDecomposition(wavelet, levels)
        self.lstm = LSTM(2**levels, hidden_dim, num_layers=2)
        self.fc = Linear(hidden_dim, 480)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 12]
            weather_future: [B, 120, 12]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Wavelet packet decomposition
        components = self.wpd.decompose(power_past)  # [B, T, 2**levels]

        # LSTM modeling
        _, (h_n, _) = self.lstm(components)  # h_n: [2, B, hidden_dim]

        # Output projection
        return self.fc(h_n[-1])  # [B, 480]
