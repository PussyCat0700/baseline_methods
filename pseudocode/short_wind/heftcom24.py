"""
Baseline Method: HEFTCom24 (LightGBM)
Task: Short-term Wind Power Forecasting

HEFTCom 2024 competition winner using LightGBM gradient boosting.

Tested Configurations:
    Config 1: n_estimators=100, num_leaves=200, embed_dim=128, power_ctx_dim=64, encoder_hidden_dim=256
    Config 2: n_estimators=200, num_leaves=400, embed_dim=128, power_ctx_dim=64, encoder_hidden_dim=256  ✓ BEST
    Config 3: n_estimators=400, num_leaves=700, embed_dim=64, power_ctx_dim=32, encoder_hidden_dim=128

Best Configuration: Config 2
    - n_estimators: 200
    - num_leaves: 400
    - embed_dim: 128
    - power_ctx_dim: 64
    - encoder_hidden_dim: 256
"""

class HEFTCom24:
    def __init__(self, n_estimators=200, num_leaves=400, embed_dim=128,
                 power_ctx_dim=64, encoder_hidden_dim=256):
        self.embed = Linear(input_dim, embed_dim)
        self.power_encoder = LSTM(power_ctx_dim, encoder_hidden_dim)
        self.model = LightGBM(
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
