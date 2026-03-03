"""
Baseline Method: TFT (Temporal Fusion Transformer)
Task: Ultrashort Wind Power Forecasting

Temporal Fusion Transformer with variable selection for ultrashort forecasting.

Tested Configurations:
    Config 1: d_model=112, lstm_layers=9, num_heads=7, dropout=0.1  ✓ BEST
    Config 2: d_model=56, lstm_layers=7, num_heads=7, dropout=0.1
    Config 3: d_model=49, lstm_layers=5, num_heads=7, dropout=0.1

Best Configuration: Config 1
    - d_model: 112
    - lstm_layers: 9
    - num_heads: 7
    - dropout: 0.1
"""

class TFT:
    def __init__(self, d_model=112, lstm_layers=9, num_heads=7, dropout=0.1):
        self.variable_selection = VariableSelectionNetwork(d_model)
        self.lstm_encoder = LSTM(d_model, d_model, num_layers=lstm_layers, dropout=dropout)
        self.attention = MultiHeadAttention(num_heads, d_model)
        self.gating = GatedResidualNetwork(d_model)
        self.fc = Linear(d_model, 480)

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
        ])  # [B, T, d_model]

        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(selected_features)  # [B, T, d_model]

        # Multi-head attention
        attn_out = self.attention(lstm_out, lstm_out, lstm_out)  # [B, T, d_model]

        # Gating
        gated_out = self.gating(attn_out)  # [B, T, d_model]

        # Output projection
        return self.fc(gated_out.mean(dim=1))  # [B, 480]
