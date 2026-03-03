"""
HEFTCom24 - LightGBM Baseline (Short-term Wind)

Competition winner from HEFTCom 2024.
Uses LightGBM gradient boosting for power forecasting.

Tested Configurations:
    Config 1: n_estimators=100, num_leaves=200, embed_dim=128, power_ctx_dim=64, encoder_hidden_dim=256
    Config 2: n_estimators=200, num_leaves=400, embed_dim=128, power_ctx_dim=64, encoder_hidden_dim=256  ✓ BEST
    Config 3: (additional config from original implementation)

Best Config: 2
"""

class HEFTCom24:
    def __init__(self, n_estimators=200, num_leaves=400, embed_dim=128,
                 power_ctx_dim=64, encoder_hidden_dim=256):  # Best config
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
        # Flatten and concatenate features
        features = concat([
            flatten(weather_future),  # [B, 1800]
            flatten(power_past)       # [B, 480]
        ])  # [B, 2280]

        # LightGBM prediction
        return self.model.predict(features)  # [B, 480]
