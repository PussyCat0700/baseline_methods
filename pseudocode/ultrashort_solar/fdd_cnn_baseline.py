"""
Baseline Method: FDD-CNN (Frequency Domain Decomposition + CNN)
Task: Ultrashort Solar Power Forecasting

Uses FFT and CNN for frequency-domain ultrashort solar forecasting.

Tested Configurations:
    Config 1: cutoff_frac=0.08, conv1_channels=16, conv2_channels=32, fc_hidden_dim=128
    Config 2: cutoff_frac=0.12, conv1_channels=32, conv2_channels=64, fc_hidden_dim=256
    Config 3: cutoff_frac=0.16, conv1_channels=32, conv2_channels=96, fc_hidden_dim=256  ✓ BEST

Best Configuration: Config 3
    - cutoff_frac: 0.16
    - conv1_channels: 32
    - conv2_channels: 96
    - fc_hidden_dim: 256
"""

class FDD_CNN:
    def __init__(self, cutoff_frac=0.16, conv1_channels=32, conv2_channels=96, fc_hidden_dim=256):
        self.fft = FFT()
        self.cutoff_frac = cutoff_frac

        self.low_branch = Sequential([
            Conv1D(1, conv1_channels, kernel_size=5),
            ReLU(),
            Conv1D(conv1_channels, conv2_channels, kernel_size=5),
            ReLU()
        ])
        self.high_branch = Sequential([
            Conv1D(1, conv1_channels, kernel_size=3),
            ReLU(),
            Conv1D(conv1_channels, conv2_channels, kernel_size=3),
            ReLU()
        ])
        self.fc = Sequential([
            Linear(conv2_channels * 2, fc_hidden_dim),
            ReLU(),
            Linear(fc_hidden_dim, 480)
        ])

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 12]
            weather_future: [B, 120, 12]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # FFT
        freq = self.fft(power_past)  # [B, 480] complex

        # Split into low and high frequency
        low_freq, high_freq = split_frequency(freq, cutoff_frac=self.cutoff_frac)

        # CNN for each band
        low_feat = self.low_branch(low_freq.unsqueeze(1))  # [B, conv2_channels, T']
        high_feat = self.high_branch(high_freq.unsqueeze(1))  # [B, conv2_channels, T']

        # Concatenate and predict
        features = concat([low_feat.mean(dim=-1), high_feat.mean(dim=-1)])  # [B, conv2_channels*2]
        return self.fc(features)  # [B, 480]
