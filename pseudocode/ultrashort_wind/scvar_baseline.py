"""
Baseline Method: SC-VAR (Spatially Correlated Vector Autoregression)
Task: Ultrashort Wind Power Forecasting

Soft-mask frequency decomposition with CNN for ultrashort-term forecasting.

Tested Configurations:
    Config 1: hidden_dim=192, num_layers=2, dropout=0.15
    Config 2: hidden_dim=320, num_layers=3, dropout=0.20
    Config 3: hidden_dim=512, num_layers=4, dropout=0.25  ✓ BEST

Best Configuration: Config 3
    - hidden_dim: 512
    - num_layers: 4
    - dropout: 0.25
"""

class SC_VAR:
    def __init__(self, hidden_dim=512, num_layers=4, dropout=0.25):
        self.fft = FFT()
        self.soft_mask = SoftMaskLayer(temperature=2.0)
        self.cnn_layers = Sequential([
            Conv1D(1, hidden_dim // 4, kernel_size=3),
            ReLU(),
            Dropout(dropout),
            *[Conv1D(hidden_dim // 4, hidden_dim // 4, kernel_size=3), ReLU(), Dropout(dropout)] * (num_layers - 1)
        ])
        self.fc = Linear(hidden_dim // 4, 480)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 15]
            weather_future: [B, 120, 15]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # FFT and soft masking
        freq = self.fft(power_past)  # [B, 480] complex
        masked_freq = self.soft_mask(freq)  # [B, 480]

        # CNN processing
        x = masked_freq.unsqueeze(1)  # [B, 1, 480]
        x = self.cnn_layers(x)  # [B, hidden_dim//4, T']

        # Output projection
        return self.fc(x.mean(dim=-1))  # [B, 480]
