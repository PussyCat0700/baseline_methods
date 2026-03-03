"""
Baseline Method: GREEK (Graph-based Recurrent Network)
Task: Short-term Wind Power Forecasting

Uses graph convolution and GRU for spatial-temporal modeling.

Tested Configurations:
    Config 1: n_estimators=50, max_features=32, embed_dim=64, power_ctx_dim=32, encoder_hidden_dim=128
    Config 2: n_estimators=100, max_features=64, embed_dim=128, power_ctx_dim=64, encoder_hidden_dim=256
    Config 3: n_estimators=150, max_features=128, embed_dim=256, power_ctx_dim=128, encoder_hidden_dim=512  ✓ BEST

Best Configuration: Config 3
    - n_estimators: 150
    - max_features: 128
    - embed_dim: 256
    - power_ctx_dim: 128
    - encoder_hidden_dim: 512
"""

class GREEK:
    def __init__(self, n_estimators=150, max_features=128, embed_dim=256,
                 power_ctx_dim=128, encoder_hidden_dim=512):
        self.embed = Linear(input_dim, embed_dim)
        self.power_encoder = LSTM(power_ctx_dim, encoder_hidden_dim)
        self.graph_conv = GraphConvolution(embed_dim, encoder_hidden_dim)
        self.ensemble = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features=max_features
        )

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 15]
            weather_future: [B, 120, 15]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Embed weather features
        weather_feat = self.embed(flatten(weather_future))  # [B, embed_dim]

        # Encode power context
        power_feat = self.power_encoder(power_past.unsqueeze(-1))[-1]  # [B, encoder_hidden_dim]

        # Graph convolution
        graph_feat = self.graph_conv(weather_feat, adjacency_matrix)  # [B, encoder_hidden_dim]

        # Concatenate features
        features = concat([graph_feat, power_feat])  # [B, encoder_hidden_dim * 2]

        return self.ensemble.predict(features)  # [B, 480]
