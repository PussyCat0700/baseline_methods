"""
CrossViViT - Cross-attention Vision Transformer (Short-term Solar)

Uses cross-attention between weather and power for solar forecasting.
"""

class CrossViViT:
    def __init__(self, hidden_dim=256, num_heads=8, num_layers=4):
        self.weather_embed = PatchEmbedding(12, hidden_dim)
        self.power_embed = Linear(1, hidden_dim)
        self.cross_attn_layers = [
            CrossAttentionLayer(num_heads, hidden_dim)
            for _ in range(num_layers)
        ]
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
        # Embed weather as tokens
        w_tokens = self.weather_embed(weather_future)  # [B, 120, hidden_dim]

        # Embed power as tokens
        p_tokens = self.power_embed(power_past.unsqueeze(-1))  # [B, 480, hidden_dim]

        # Cross-attention: power queries weather
        x = p_tokens
        for layer in self.cross_attn_layers:
            x = layer(query=x, key=w_tokens, value=w_tokens)  # [B, 480, hidden_dim]

        # Output projection
        return self.fc(x.mean(dim=1))  # [B, 480]
