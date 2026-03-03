"""
LSSVM-RBFNN - Least Squares SVM + RBF Neural Network (Ultrashort Wind)

Combines LSSVM and RBFNN for ultrashort-term wind forecasting.
"""

class LSSVM_RBFNN:
    def __init__(self, n_centers=100, gamma=0.1):
        self.lssvm = LSSVM(gamma=gamma)
        self.rbfnn = RBFNetwork(n_centers=n_centers)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 15]
            weather_future: [B, 120, 15]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Flatten features
        features = concat([
            flatten(weather_future),
            flatten(power_past)
        ])  # [B, 2280]

        # LSSVM prediction
        svm_out = self.lssvm.predict(features)  # [B, 480]

        # RBFNN prediction
        rbf_out = self.rbfnn(features)  # [B, 480]

        # Ensemble
        return (svm_out + rbf_out) / 2  # [B, 480]
