"""
GREEK - Graph-based Recurrent Network (Short-term Wind)

Uses graph convolution and GRU for spatial-temporal modeling.
"""

class GREEK:
    def __init__(self, hidden_dim=256, num_layers=2):
        self.graph_conv = GraphConvolution(input_dim, hidden_dim)
        self.gru = GRU(hidden_dim, num_layers)
        self.fc = Linear(hidden_dim, 480)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 15]
            weather_future: [B, 120, 15]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Concatenate inputs
        x = concat([
            weather_future,
            power_past.unsqueeze(-1)
        ], dim=1)  # [B, 600, 15/1]

        # Graph convolution
        x = self.graph_conv(x, adjacency_matrix)  # [B, 600, hidden_dim]

        # GRU encoding
        _, h_n = self.gru(x)  # h_n: [num_layers, B, hidden_dim]

        # Output projection
        return self.fc(h_n[-1])  # [B, 480]
