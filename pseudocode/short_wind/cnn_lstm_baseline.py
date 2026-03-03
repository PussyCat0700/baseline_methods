"""
Baseline Method: CNN-LSTM Hybrid
Task: Short-term Wind Power Forecasting

Combines CNN for feature extraction with LSTM for temporal modeling.

Tested Configurations:
    Config 1: hidden_size=64, num_layers=1
    Config 2: hidden_size=128, num_layers=2
    Config 3: hidden_size=256, num_layers=4  ✓ BEST

Best Configuration: Config 3
    - hidden_size: 256
    - num_layers: 4
    - dropout: 0.5
"""

class CNN_LSTM:
    def __init__(self, hidden_size=256, num_layers=4, dropout=0.5):
        self.cnn = Sequential([
            Conv1D(15, 32, kernel_size=3),
            ReLU(),
            Conv1D(32, 64, kernel_size=3),
            ReLU(),
            Conv1D(64, 128, kernel_size=3),
            ReLU()
        ])
        self.lstm = LSTM(128, hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = Linear(hidden_size, 480)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 15]
            weather_future: [B, 120, 15]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # CNN feature extraction
        x = weather_future.transpose(1, 2)  # [B, 15, 120]
        x = self.cnn(x)  # [B, 128, T']
        x = x.transpose(1, 2)  # [B, T', 128]

        # LSTM temporal modeling
        _, (h_n, _) = self.lstm(x)  # h_n: [num_layers, B, hidden_size]

        # Output projection
        return self.fc(h_n[-1])  # [B, 480]
