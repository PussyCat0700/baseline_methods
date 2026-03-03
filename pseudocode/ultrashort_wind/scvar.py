"""
SC-VAR - Spatially Correlated Vector Autoregression (Ultrashort Wind)

VAR model with spatial correlation for ultrashort-term wind forecasting.
"""

class SC_VAR:
    def __init__(self, lag_order=24, spatial_dim=8):
        self.var_model = VARModel(lag_order=lag_order)
        self.spatial_weights = SpatialWeightMatrix(spatial_dim)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 15]
            weather_future: [B, 120, 15]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Average weather over spatial dimension
        weather_avg = weather_future.mean(dim=-1)  # [B, 120]

        # Concatenate with power
        x = concat([
            weather_avg.unsqueeze(-1),
            power_past.unsqueeze(-1)
        ], dim=-1)  # [B, max(120,480), 2]

        # Apply spatial correlation
        x_spatial = self.spatial_weights(x)  # [B, T, 2]

        # VAR forecasting
        return self.var_model.forecast(x_spatial)  # [B, 480]
