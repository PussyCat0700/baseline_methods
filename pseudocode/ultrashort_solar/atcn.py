"""
ATCN - Attention-based Temporal Convolutional Network (Ultrashort Solar)

Uses dilated CNN with attention for ultrashort solar forecasting.
"""

class ATCN:
    def __init__(self, channels=[32,64,128], dilation_rates=[1,2,4]):
        self.dilated_convs = [
            DilatedConv1D(12, channels[i], kernel_size=3, dilation=dilation_rates[i])
            for i in range(len(channels))
        ]
        self.attention = SelfAttention(channels[-1])
        self.fc = Linear(channels[-1], 480)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 12]
            weather_future: [B, 120, 12]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Dilated convolutions
        x = weather_future.transpose(1, 2)  # [B, 12, 120]
        for conv in self.dilated_convs:
            x = F.relu(conv(x))  # [B, channels[i], T]

        x = x.transpose(1, 2)  # [B, T, channels[-1]]

        # Self-attention
        x = self.attention(x)  # [B, T, channels[-1]]

        # Output projection
        return self.fc(x.mean(dim=1))  # [B, 480]
