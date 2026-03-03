"""
FusionSF - Fusion of Spatial Features (Short-term Solar)

Fuses spatial and temporal features for solar power forecasting.
"""

class FusionSF:
    def __init__(self, hidden_dim=256):
        self.spatial_encoder = Conv2D([32, 64, 128])
        self.temporal_encoder = LSTM(hidden_dim, num_layers=2)
        self.fusion = AttentionFusion(hidden_dim)
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
        # Spatial feature extraction (reshape to 2D)
        spatial_feat = weather_future.reshape(B, 10, 12, 12)  # [B, 10, 12, 12]
        spatial_feat = self.spatial_encoder(spatial_feat)  # [B, 128, H', W']
        spatial_feat = spatial_feat.mean(dim=[2,3])  # [B, 128]

        # Temporal feature extraction
        temporal_feat = self.temporal_encoder(power_past.unsqueeze(-1))  # [B, 480, hidden_dim]
        temporal_feat = temporal_feat[:, -1, :]  # [B, hidden_dim]

        # Fusion
        fused = self.fusion(spatial_feat, temporal_feat)  # [B, hidden_dim]

        return self.fc(fused)  # [B, 480]
