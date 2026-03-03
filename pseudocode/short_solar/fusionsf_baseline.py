"""
Baseline Method: FusionSF (Fusion of Spatial Features)
Task: Short-term Solar Power Forecasting

Fuses spatial and temporal features for solar power forecasting.

Tested Configurations:
    Config 1: d_model=128, nhead=2, num_decoder_layers=3, look_back_steps=48
    Config 2: d_model=256, nhead=4, num_decoder_layers=6, look_back_steps=96  ✓ BEST
    Config 3: d_model=512, nhead=8, num_decoder_layers=12, look_back_steps=96

Best Configuration: Config 2
    - d_model: 256
    - nhead: 4
    - num_decoder_layers: 6
    - look_back_steps: 96
    - dim_feedforward: 1024
"""

class FusionSF:
    def __init__(self, d_model=256, nhead=4, num_decoder_layers=6,
                 look_back_steps=96, dim_feedforward=1024):
        self.encoder = GRU(12, d_model, num_layers=2)
        self.decoder = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward
        )
        self.fc = Linear(d_model, 480)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 12]
            weather_future: [B, 120, 12]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Encode past context
        _, h_n = self.encoder(weather_past)  # h_n: [2, B, d_model]
        memory = h_n[-1].unsqueeze(1)  # [B, 1, d_model]

        # Decode future
        future_tokens = weather_future  # [B, 120, 12]
        decoder_out = self.decoder(future_tokens, memory)  # [B, 120, d_model]

        # Output projection
        return self.fc(decoder_out.mean(dim=1))  # [B, 480]
