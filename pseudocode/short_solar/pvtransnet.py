"""
Baseline Method: PVTransNet (PV Transformer Network)
Task: Short-term Solar Power Forecasting

Transformer-based model specifically designed for PV power forecasting.

Tested Configurations:
    Config 1: emb_dim=128, num_layers=1, num_heads=2, head_dim=64
    Config 2: emb_dim=256, num_layers=4, num_heads=4, head_dim=128
    Config 3: emb_dim=512, num_layers=12, num_heads=8, head_dim=128  ✓ BEST

Best Configuration: Config 3
    - emb_dim: 512
    - num_layers: 12
    - num_heads: 8
    - head_dim: 128
    - mlp_dim: 2048 (4 * emb_dim)
"""

class PVTransNet:
    def __init__(self, emb_dim=512, num_layers=12, num_heads=8, head_dim=128, dropout=0.1):
        self.embed = Linear(12, emb_dim)
        self.pos_encoding = PositionalEncoding(emb_dim)
        self.transformer = TransformerEncoder(
            dim=emb_dim,
            depth=num_layers,
            heads=num_heads,
            dim_head=head_dim,
            mlp_dim=4 * emb_dim,
            dropout=dropout
        )
        self.fc = Linear(emb_dim, 480)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 12]
            weather_future: [B, 120, 12]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Embed weather
        x = self.embed(weather_future)  # [B, 120, emb_dim]
        x = self.pos_encoding(x)  # [B, 120, emb_dim]

        # Transformer encoding
        x = self.transformer(x)  # [B, 120, emb_dim]

        # Output projection
        return self.fc(x.mean(dim=1))  # [B, 480]
