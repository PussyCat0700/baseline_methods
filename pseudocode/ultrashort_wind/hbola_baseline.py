"""
Baseline Method: HBOLA (Hybrid LSTM with Online Learning)
Task: Ultrashort Wind Power Forecasting

LSTM with online adaptation for ultrashort-term wind forecasting.

Tested Configurations:
    Config 1: num_layers=5, hidden_size=64, dropout=0.1
    Config 2: num_layers=10, hidden_size=96, dropout=0.1  ✓ BEST
    Config 3: num_layers=20, hidden_size=128, dropout=0.15

Best Configuration: Config 2
    - num_layers: 10
    - hidden_size: 96
    - dropout: 0.1
"""

class HBOLA:
    def __init__(self, num_layers=10, hidden_size=96, dropout=0.1):
        self.encoder_lstm = LSTM(15, hidden_size, num_layers=num_layers, dropout=dropout)
        self.decoder_lstm = LSTM(15, hidden_size, num_layers=num_layers, dropout=dropout)
        self.online_layer = AdaptiveLinear(hidden_size, 480)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 15]
            weather_future: [B, 120, 15]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Encode past context
        _, (h_n, c_n) = self.encoder_lstm(weather_past)  # h_n: [num_layers, B, hidden_size]

        # Decode future with initial state from encoder
        decoder_out, _ = self.decoder_lstm(weather_future, (h_n, c_n))  # [B, 120, hidden_size]

        # Online adaptive prediction
        return self.online_layer(decoder_out[:, -1, :])  # [B, 480]
