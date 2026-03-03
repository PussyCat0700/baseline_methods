"""
RFs-ALO - Random Forests with Ant Lion Optimizer (Ultrashort Solar)

Random Forest optimized with Ant Lion Optimizer for ultrashort solar forecasting.
"""

class RFs_ALO:
    def __init__(self, n_estimators=100, n_ants=50, max_iter=100):
        self.rf = RandomForestRegressor(n_estimators=n_estimators)
        self.alo = AntLionOptimizer(n_ants=n_ants, max_iter=max_iter)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 12]
            weather_future: [B, 120, 12]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Flatten features
        features = concat([
            flatten(weather_future),
            flatten(power_past)
        ])  # [B, 1920]

        # ALO optimizes RF hyperparameters
        # (max_depth, min_samples_split, min_samples_leaf)
        optimized_params = self.alo.optimize(
            objective=lambda params: rf_cv_score(self.rf, features, params),
            bounds=[(3, 20), (2, 20), (1, 10)]
        )

        # Set optimized parameters
        self.rf.set_params(**optimized_params)

        # Random Forest prediction
        return self.rf.predict(features)  # [B, 480]
