"""
LSTM-GCN-MLP - LSTM-GCN-MLP Hybrid (Ultrashort Solar)

Combines LSTM, Graph Convolution, and MLP for ultrashort solar forecasting.
"""

class LSTM_GCN_MLP:
    def __init__(self, lstm_hidden=256, gcn_hidden=256, mlp_hidden=512):
        self.lstm = LSTM(1, lstm_hidden, num_layers=2)
        self.gcn = GraphConvolution(lstm_hidden, gcn_hidden)
        self.mlp = Sequential([
            Linear(gcn_hidden, mlp_hidden),
            ReLU(),
            Dropout(0.3),
            Linear(mlp_hidden, mlp_hidden),
            ReLU(),
            Dropout(0.3),
            Linear(mlp_hidden, 480)
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
        # LSTM encoding
        _, (h_n, _) = self.lstm(power_past.unsqueeze(-1))  # h_n: [2, B, lstm_hidden]

        # Graph convolution
        x = self.gcn(h_n[-1], adjacency_matrix)  # [B, gcn_hidden]

        # MLP prediction
        return self.mlp(x)  # [B, 480]
