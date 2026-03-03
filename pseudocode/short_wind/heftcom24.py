"""
HEFTCom24 - LightGBM Baseline (Short-term Wind)

Competition winner from HEFTCom 2024.
Uses LightGBM gradient boosting for power forecasting.
"""

class HEFTCom24:
    def __init__(self, n_estimators=500, num_leaves=31, learning_rate=0.05):
        self.model = LightGBM(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate
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
