"""
Baseline Method: LSTM-GCN-MLP Hybrid
Task: Ultrashort Solar Power Forecasting

Combines soft-mask frequency decomposition with CNN for ultrashort solar forecasting.

Tested Configurations:
    Config 1: hidden_dim=64, num_layers=2, dropout=0.1, conv1_channels=16, conv2_channels=32,
              cutoff_frac=0.10, soft_temp=1.5
    Config 2: hidden_dim=96, num_layers=3, dropout=0.1, conv1_channels=24, conv2_channels=48,
              cutoff_frac=0.12, soft_temp=2.0
    Config 3: hidden_dim=128, num_layers=3, dropout=0.15, conv1_channels=32, conv2_channels=64,
              cutoff_frac=0.15, soft_temp=2.5  ✓ BEST

Best Configuration: Config 3
    - hidden_dim: 128
    - num_layers: 3
    - dropout: 0.15
    - conv1_channels: 32
    - conv2_channels: 64
    - cutoff_frac: 0.15
    - soft_temp: 2.5
"""

class LSTM_GCN_MLP:
    def __init__(self, hidden_dim=128, num_layers=3, dropout=0.15, conv1_channels=32,
                 conv2_channels=64, cutoff_frac=0.15, soft_temp=2.5):
        self.fft = FFT()
        self.soft_mask = SoftMaskLayer(temperature=soft_temp)
        self.cnn = Sequential([
            Conv1D(1, conv1_channels, kernel_size=3),
            ReLU(),
            Dropout(dropout),
            Conv1D(conv1_channels, conv2_channels, kernel_size=3),
            ReLU(),
            Dropout(dropout)
        ])
        self.lstm = LSTM(conv2_channels, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.fc = Linear(hidden_dim, 480)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 12]
            weather_future: [B, 120, 12]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # FFT and soft masking
        freq = self.fft(power_past)  # [B, 480] complex
        masked_freq = self.soft_mask(freq)  # [B, 480]

        # CNN processing
        x = masked_freq.unsqueeze(1)  # [B, 1, 480]
        x = self.cnn(x)  # [B, conv2_channels, T']
        x = x.transpose(1, 2)  # [B, T', conv2_channels]

        # LSTM encoding
        _, (h_n, _) = self.lstm(x)  # h_n: [num_layers, B, hidden_dim]

        # Output projection
        return self.fc(h_n[-1])  # [B, 480]
