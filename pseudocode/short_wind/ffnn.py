"""
FFNN - Feed-Forward Neural Network (Short-term Wind)

Simple multi-layer perceptron baseline.
"""

class FFNN:
    def __init__(self, hidden_dim=512, num_layers=3):
        self.layers = Sequential([
            Linear(input_dim, hidden_dim),
            ReLU(),
            Dropout(0.3),
            *[Linear(hidden_dim, hidden_dim), ReLU(), Dropout(0.3)] * (num_layers-1),
            Linear(hidden_dim, 480)
        ])

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 15]
            weather_future: [B, 120, 15]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        x = concat([
            flatten(weather_future),
            flatten(power_past)
        ])  # [B, 2280]

        return self.layers(x)  # [B, 480]
