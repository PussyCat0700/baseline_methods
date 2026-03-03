"""
FDD-CNN - Frequency Domain Decomposition + CNN (Ultrashort Solar)

Uses FFT and CNN for frequency-domain ultrashort solar forecasting.
"""

class FDD_CNN:
    def __init__(self, n_bands=3, cnn_channels=[32,64,128]):
        self.fft = FFT()
        self.band_cnns = [
            Sequential([
                Conv1D(1, cnn_channels[0], kernel_size=3),
                ReLU(),
                Conv1D(cnn_channels[0], cnn_channels[1], kernel_size=3),
                ReLU(),
                Conv1D(cnn_channels[1], cnn_channels[2], kernel_size=3),
                ReLU()
            ])
            for _ in range(n_bands)
        ]
        self.fc = Linear(cnn_channels[2] * n_bands, 480)

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

        # Split into frequency bands
        bands = split_frequency_bands(freq, n_bands=3)  # List of [B, T_i]

        # CNN for each band
        features = []
        for cnn, band in zip(self.band_cnns, bands):
            x = band.unsqueeze(1)  # [B, 1, T_i]
            x = cnn(x)  # [B, 128, T']
            features.append(x.mean(dim=-1))  # [B, 128]

        # Concatenate and predict
        x = concat(features, dim=-1)  # [B, 128*3]
        return self.fc(x)  # [B, 480]
