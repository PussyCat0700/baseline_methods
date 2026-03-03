"""
Baseline Method: GEFCom12 (Gradient Boosting)
Task: Short-term Wind Power Forecasting

GEFCom 2012 competition winner using Gradient Boosting Decision Trees.

Tested Configurations:
    Config 1: n_estimators=5, num_leaves=16, embed_dim=64, power_ctx_dim=32, encoder_hidden_dim=128
    Config 2: n_estimators=10, num_leaves=32, embed_dim=128, power_ctx_dim=64, encoder_hidden_dim=256
    Config 3: n_estimators=20, num_leaves=64, embed_dim=256, power_ctx_dim=128, encoder_hidden_dim=512  ✓ BEST

Best Configuration: Config 3
    - n_estimators: 20
    - num_leaves: 64
    - embed_dim: 256
    - power_ctx_dim: 128
    - encoder_hidden_dim: 512
"""

class GEFCom12:
    def __init__(self, n_estimators=20, num_leaves=64, embed_dim=256,
                 power_ctx_dim=128, encoder_hidden_dim=512):
        self.embed = Linear(input_dim, embed_dim)
        self.power_encoder = LSTM(power_ctx_dim, encoder_hidden_dim)
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            num_leaves=num_leaves
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

        # Concatenate features
        features = concat([weather_feat, power_feat])  # [B, embed_dim + encoder_hidden_dim]

        return self.model.predict(features)  # [B, 480]
