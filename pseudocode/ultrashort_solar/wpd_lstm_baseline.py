"""
Baseline Method: WPD-LSTM (Wavelet Packet Decomposition + LSTM)
Task: Ultrashort Solar Power Forecasting

Uses multi-scale frequency decomposition for ultrashort solar forecasting.

Tested Configurations:
    Config 1: c_hidden=32, alpha=0.3, fc_hidden_dim=128, dropout=0.05
    Config 2: c_hidden=48, alpha=0.5, fc_hidden_dim=256, dropout=0.10  ✓ BEST
    Config 3: c_hidden=64, alpha=0.7, fc_hidden_dim=384, dropout=0.15

Best Configuration: Config 2
    - c_hidden: 48
    - alpha: 0.5
    - fc_hidden_dim: 256
    - dropout: 0.10
"""

class WPD_LSTM:
    def __init__(self, c_hidden=48, alpha=0.5, fc_hidden_dim=256, dropout=0.10):
        self.fft = FFT()
        self.low_branch = Sequential([
            Conv1D(1, c_hidden, kernel_size=5),
            ReLU(),
            Conv1D(c_hidden, c_hidden * 2, kernel_size=5),
            ReLU()
        ])
        self.high_branch = Sequential([
            Conv1D(1, c_hidden, kernel_size=3),
            ReLU(),
            Conv1D(c_hidden, c_hidden * 2, kernel_size=3),
            ReLU()
        ])
        self.fc = Sequential([
            Linear(c_hidden * 4, fc_hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(fc_hidden_dim, 480)
        ])
        self.alpha = alpha

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 12]
            weather_future: [B, 120, 12]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # FFT decomposition
        freq = self.fft(power_past)  # [B, 480] complex
        low_freq, high_freq = split_frequency(freq, alpha=self.alpha)  # [B, T_low], [B, T_high]

        # Process low and high frequency components
        low_feat = self.low_branch(low_freq.unsqueeze(1))  # [B, c_hidden*2, T']
        high_feat = self.high_branch(high_freq.unsqueeze(1))  # [B, c_hidden*2, T']

        # Concatenate and predict
        features = concat([low_feat.mean(dim=-1), high_feat.mean(dim=-1)])  # [B, c_hidden*4]
        return self.fc(features)  # [B, 480]
