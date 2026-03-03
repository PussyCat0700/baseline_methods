"""
TFT - Temporal Fusion Transformer (Ultrashort Wind)

Temporal Fusion Transformer with variable selection for ultrashort forecasting.
"""

class TFT:
    def __init__(self, hidden_dim=256, num_heads=4, num_layers=3):
        self.variable_selection = VariableSelectionNetwork(hidden_dim)
        self.lstm_encoder = LSTM(hidden_dim, hidden_dim, num_layers=2)
        self.attention = MultiHeadAttention(num_heads, hidden_dim)
        self.gating = GatedResidualNetwork(hidden_dim)
        self.fc = Linear(hidden_dim, 480)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 15]
            weather_future: [B, 120, 15]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Variable selection
        selected_features = self.variable_selection([
            weather_past,
            weather_future,
            power_past
        ])  # [B, T, hidden_dim]

        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(selected_features)  # [B, T, hidden_dim]

        # Multi-head attention
        attn_out = self.attention(lstm_out, lstm_out, lstm_out)  # [B, T, hidden_dim]

        # Gating
        gated_out = self.gating(attn_out)  # [B, T, hidden_dim]

        # Output projection
        return self.fc(gated_out.mean(dim=1))  # [B, 480]
