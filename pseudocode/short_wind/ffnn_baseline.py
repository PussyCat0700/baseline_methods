"""
Baseline Method: FFNN (Feed-Forward Neural Network)
Task: Short-term Wind Power Forecasting

Simple multi-layer perceptron baseline.

Tested Configurations:
    Config 1: hidden_dim=512, num_layers=12
    Config 2: hidden_dim=256, num_layers=8
    Config 3: hidden_dim=128, num_layers=4  ✓ BEST

Best Configuration: Config 3
    - hidden_dim: 128
    - num_layers: 4
    - dropout: 0.3
"""

class FFNN:
    def __init__(self, hidden_dim=128, num_layers=4, dropout=0.3):
        self.layers = Sequential([
            Linear(input_dim, hidden_dim),
            ReLU(),
            Dropout(dropout),
            *[Linear(hidden_dim, hidden_dim), ReLU(), Dropout(dropout)] * (num_layers-1),
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
