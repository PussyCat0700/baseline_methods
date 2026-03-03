"""
Baseline Method: CrossViViT (Cross-attention Vision Transformer)
Task: Short-term Solar Power Forecasting

Uses cross-attention between weather and power for solar forecasting.

Tested Configurations:
    Config 1: dim=384, depth=16, heads=12, dim_head=64, mlp_ratio=4, dropout=0.4, use_glu=True,
              depth_cross=4, decoder_dim=128, decoder_depth=4, decoder_heads=6, decoder_dim_head=128
    Config 2: dim=256, depth=12, heads=8, dim_head=64, mlp_ratio=4, dropout=0.3, use_glu=True,
              depth_cross=3, decoder_dim=96, decoder_depth=3, decoder_heads=4, decoder_dim_head=96
    Config 3: dim=128, depth=8, heads=6, dim_head=64, mlp_ratio=4, dropout=0.2, use_glu=True,
              depth_cross=2, decoder_dim=64, decoder_depth=2, decoder_heads=3, decoder_dim_head=64  ✓ BEST

Best Configuration: Config 3
    - dim: 128
    - depth: 8
    - heads: 6
    - dim_head: 64
    - mlp_ratio: 4
    - dropout: 0.2
    - use_glu: True
    - depth_cross: 2
    - decoder_dim: 64
    - decoder_depth: 2
    - decoder_heads: 3
    - decoder_dim_head: 64
"""

class CrossViViT:
    def __init__(self, dim=128, depth=8, heads=6, dim_head=64, mlp_ratio=4, dropout=0.2,
                 use_glu=True, depth_cross=2, decoder_dim=64, decoder_depth=2,
                 decoder_heads=3, decoder_dim_head=64):
        self.weather_embed = PatchEmbedding(12, dim)
        self.power_embed = Linear(1, dim)
        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_ratio, dropout, use_glu)
        self.cross_attn_layers = [
            CrossAttentionLayer(heads, dim)
            for _ in range(depth_cross)
        ]
        self.decoder = TransformerDecoder(decoder_dim, decoder_depth, decoder_heads, decoder_dim_head)
        self.fc = Linear(decoder_dim, 480)

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
        w_tokens = self.weather_embed(weather_future)  # [B, 120, dim]
        w_tokens = self.encoder(w_tokens)  # [B, 120, dim]

        # Embed power as tokens
        p_tokens = self.power_embed(power_past.unsqueeze(-1))  # [B, 480, dim]

        # Cross-attention: power queries weather
        x = p_tokens
        for layer in self.cross_attn_layers:
            x = layer(query=x, key=w_tokens, value=w_tokens)  # [B, 480, dim]

        # Decode
        x = self.decoder(x)  # [B, 480, decoder_dim]

        # Output projection
        return self.fc(x.mean(dim=1))  # [B, 480]
