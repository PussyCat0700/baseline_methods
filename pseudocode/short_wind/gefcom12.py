"""
GEFCom12 - Gradient Boosting Baseline (Short-term Wind)

Competition winner from GEFCom 2012.
Uses Gradient Boosting Decision Trees for power forecasting.
"""

class GEFCom12:
    def __init__(self, n_estimators=1000, max_depth=10, learning_rate=0.1):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
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
        features = concat([
            flatten(weather_future),
            flatten(power_past)
        ])  # [B, 2280]

        return self.model.predict(features)  # [B, 480]
