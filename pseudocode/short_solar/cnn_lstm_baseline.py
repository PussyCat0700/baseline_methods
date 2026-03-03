"""
Baseline Method: CNN-LSTM Hybrid
Task: Short-term Solar Power Forecasting

Combines CNN for feature extraction with LSTM for temporal modeling.

Tested Configurations:
    Config 1: d_model=128, nhead=4, cnn_layers_past=2, cnn_layers_future=2, look_back_steps=96
    Config 2: d_model=256, nhead=8, cnn_layers_past=3, cnn_layers_future=2, look_back_steps=192, ffn_dim=1024  ✓ BEST
    Config 3: d_model=128, nhead=4, cnn_layers_past=4, cnn_layers_future=3, look_back_steps=48, dropout=0.2

Best Configuration: Config 2
    - d_model: 256
    - nhead: 8
    - cnn_layers_past: 3
    - cnn_layers_future: 2
    - look_back_steps: 192
    - ffn_dim: 1024
"""

class CNN_LSTM_Solar:
    def __init__(self, d_model=256, nhead=8, cnn_layers_past=3, cnn_layers_future=2,
                 look_back_steps=192, ffn_dim=1024):
        self.cnn_past = Sequential([
            Conv1D(12, 32, kernel_size=3) for _ in range(cnn_layers_past)
        ])
        self.cnn_future = Sequential([
            Conv1D(12, 32, kernel_size=3) for _ in range(cnn_layers_future)
        ])
        self.attention = MultiHeadAttention(nhead, d_model)
        self.ffn = Sequential([
            Linear(d_model, ffn_dim),
            ReLU(),
            Linear(ffn_dim, 480)
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
        # CNN feature extraction
        past_feat = self.cnn_past(weather_past.transpose(1, 2))  # [B, C, T]
        future_feat = self.cnn_future(weather_future.transpose(1, 2))  # [B, C, T]

        # Cross-attention
        past_feat = past_feat.transpose(1, 2)  # [B, T, C]
        future_feat = future_feat.transpose(1, 2)  # [B, T, C]
        attn_out = self.attention(query=future_feat, key=past_feat, value=past_feat)  # [B, T, C]

        # Output projection
        return self.ffn(attn_out.mean(dim=1))  # [B, 480]
