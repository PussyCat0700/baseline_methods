"""
CNN-LSTM - CNN-LSTM Hybrid (Short-term Wind)

Combines CNN for feature extraction with LSTM for temporal modeling.
"""

class CNN_LSTM:
    def __init__(self, cnn_channels=[32,64,128], lstm_hidden=256):
        self.cnn = Sequential([
            Conv1D(15, cnn_channels[0], kernel_size=3),
            ReLU(),
            Conv1D(cnn_channels[0], cnn_channels[1], kernel_size=3),
            ReLU(),
            Conv1D(cnn_channels[1], cnn_channels[2], kernel_size=3),
            ReLU()
        ])
        self.lstm = LSTM(cnn_channels[2], lstm_hidden, num_layers=2)
        self.fc = Linear(lstm_hidden, 480)

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
        _, (h_n, _) = self.lstm(x)  # h_n: [2, B, 256]

        # Output projection
        return self.fc(h_n[-1])  # [B, 480]
