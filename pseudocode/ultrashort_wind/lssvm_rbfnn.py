"""
Baseline Method: LSSVM-RBFNN (Least Squares SVM + RBF Neural Network)
Task: Ultrashort Wind Power Forecasting

Combines LSSVM and RBFNN for ultrashort-term wind forecasting.

Tested Configurations:
    Config 1: embed_dim=64, encoder_hidden_dim=128, encoder_num_layers=2, num_prototypes=64, gamma=0.8
    Config 2: embed_dim=128, encoder_hidden_dim=256, encoder_num_layers=2, num_prototypes=128, gamma=0.5
    Config 3: embed_dim=128, encoder_hidden_dim=384, encoder_num_layers=3, num_prototypes=256, gamma=0.3  ✓ BEST

Best Configuration: Config 3
    - embed_dim: 128
    - encoder_hidden_dim: 384
    - encoder_num_layers: 3
    - num_prototypes: 256
    - gamma: 0.3
"""

class LSSVM_RBFNN:
    def __init__(self, embed_dim=128, encoder_hidden_dim=384, encoder_num_layers=3,
                 num_prototypes=256, gamma=0.3):
        self.embed = Linear(15, embed_dim)
        self.encoder = LSTM(embed_dim, encoder_hidden_dim, num_layers=encoder_num_layers)
        self.lssvm = LSSVM(gamma=gamma)
        self.rbfnn = RBFNetwork(n_centers=num_prototypes)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 15]
            weather_future: [B, 120, 15]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Embed and encode weather
        weather_emb = self.embed(weather_future)  # [B, 120, embed_dim]
        _, (h_n, _) = self.encoder(weather_emb)  # h_n: [encoder_num_layers, B, encoder_hidden_dim]
        features = h_n[-1]  # [B, encoder_hidden_dim]

        # LSSVM prediction
        svm_out = self.lssvm.predict(features)  # [B, 480]

        # RBFNN prediction
        rbf_out = self.rbfnn(features)  # [B, 480]

        # Ensemble
        return (svm_out + rbf_out) / 2  # [B, 480]
