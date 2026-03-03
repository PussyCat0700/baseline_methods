"""
PVTransNet - PV Transformer Network (Short-term Solar)

Transformer-based model specifically designed for PV power forecasting.
"""

class PVTransNet:
    def __init__(self, hidden_dim=256, num_heads=8, num_layers=4):
        self.embed = Linear(12+1, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.transformer = TransformerEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
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
        # Concatenate weather and power
        power_resampled = resample(power_past, 120)  # [B, 120, 1]
        x = concat([weather_future, power_resampled], dim=-1)  # [B, 120, 13]

        # Embed and add positional encoding
        x = self.embed(x)  # [B, 120, hidden_dim]
        x = self.pos_encoding(x)  # [B, 120, hidden_dim]

        # Transformer encoding
        x = self.transformer(x)  # [B, 120, hidden_dim]

        # Output projection
        return self.fc(x.mean(dim=1))  # [B, 480]
