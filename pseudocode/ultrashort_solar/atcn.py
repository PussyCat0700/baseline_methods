"""
Baseline Method: ATCN (Attention-based Temporal Convolutional Network)
Task: Ultrashort Solar Power Forecasting

Uses tri-band frequency decomposition with dilated CNN for ultrashort solar forecasting.

Tested Configurations:
    Config 1: cutoff1_frac=0.06, cutoff2_frac=0.18, conv1_channels=16, conv2_channels=32,
              fc_hidden_dim=128, near_horizon_len=16  ✓ BEST
    Config 2: cutoff1_frac=0.08, cutoff2_frac=0.20, conv1_channels=32, conv2_channels=64,
              fc_hidden_dim=256, near_horizon_len=16
    Config 3: cutoff1_frac=0.10, cutoff2_frac=0.24, conv1_channels=32, conv2_channels=96,
              fc_hidden_dim=256, near_horizon_len=8, dropout=0.2

Best Configuration: Config 1
    - cutoff1_frac: 0.06
    - cutoff2_frac: 0.18
    - conv1_channels: 16
    - conv2_channels: 32
    - fc_hidden_dim: 128
    - near_horizon_len: 16
"""

class ATCN:
    def __init__(self, cutoff1_frac=0.06, cutoff2_frac=0.18, conv1_channels=16,
                 conv2_channels=32, fc_hidden_dim=128, near_horizon_len=16):
        self.fft = FFT()
        self.cutoff1_frac = cutoff1_frac
        self.cutoff2_frac = cutoff2_frac

        # Three branches for three frequency bands
        self.low_branch = DilatedCNN(1, conv1_channels, conv2_channels, dilation=1)
        self.mid_branch = DilatedCNN(1, conv1_channels, conv2_channels, dilation=2)
        self.high_branch = DilatedCNN(1, conv1_channels, conv2_channels, dilation=4)

        self.fc = Sequential([
            Linear(conv2_channels * 3, fc_hidden_dim),
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
        # FFT and tri-band decomposition
        freq = self.fft(power_past)  # [B, 480] complex
        low, mid, high = split_tri_band(freq, self.cutoff1_frac, self.cutoff2_frac)

        # Process each band
        low_feat = self.low_branch(low.unsqueeze(1))  # [B, conv2_channels, T']
        mid_feat = self.mid_branch(mid.unsqueeze(1))  # [B, conv2_channels, T']
        high_feat = self.high_branch(high.unsqueeze(1))  # [B, conv2_channels, T']

        # Concatenate and predict
        features = concat([
            low_feat.mean(dim=-1),
            mid_feat.mean(dim=-1),
            high_feat.mean(dim=-1)
        ])  # [B, conv2_channels * 3]

        return self.fc(features)  # [B, 480]
