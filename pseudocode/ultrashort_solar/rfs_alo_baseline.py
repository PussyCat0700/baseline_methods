"""
Baseline Method: RFs-ALO (Random Forests with Ant Lion Optimizer)
Task: Ultrashort Solar Power Forecasting

Random Forest optimized with Ant Lion Optimizer for ultrashort solar forecasting.

Tested Configurations:
    Config 1: num_trees=500, num_leaves=10, embed_dim=128, power_ctx_dim=128, encoder_hidden_dim=256,
              cnn_c_hidden=48, cnn_fc_hidden=256, cutoff_frac=0.12, dropout=0.1
    Config 2: num_trees=1000, num_leaves=5, embed_dim=64, power_ctx_dim=64, encoder_hidden_dim=128,
              cnn_c_hidden=32, cnn_fc_hidden=128, cutoff_frac=0.10, dropout=0.15  ✓ BEST
    Config 3: num_trees=300, num_leaves=20, embed_dim=96, power_ctx_dim=96, encoder_hidden_dim=192,
              cnn_c_hidden=40, cnn_fc_hidden=192, cutoff_frac=0.14, dropout=0.12

Best Configuration: Config 2
    - num_trees: 1000
    - num_leaves: 5
    - embed_dim: 64
    - power_ctx_dim: 64
    - encoder_hidden_dim: 128
    - cnn_c_hidden: 32
    - cnn_fc_hidden: 128
    - cutoff_frac: 0.10
    - dropout: 0.15
"""

class RFs_ALO:
    def __init__(self, num_trees=1000, num_leaves=5, embed_dim=64, power_ctx_dim=64,
                 encoder_hidden_dim=128, cnn_c_hidden=32, cnn_fc_hidden=128,
                 cutoff_frac=0.10, dropout=0.15):
        self.embed = Linear(12, embed_dim)
        self.power_encoder = LSTM(power_ctx_dim, encoder_hidden_dim)
        self.cnn = Sequential([
            Conv1D(1, cnn_c_hidden, kernel_size=3),
            ReLU(),
            Dropout(dropout),
            Linear(cnn_c_hidden, cnn_fc_hidden)
        ])
        self.rf = SoftRandomForest(
            n_estimators=num_trees,
            num_leaves=num_leaves
        )
        self.alo = AntLionOptimizer(n_ants=50, max_iter=100)

    def forward(self, weather_past, weather_future, power_past):
        """
        Input:
            weather_past: [B, 120, 12]
            weather_future: [B, 120, 12]
            power_past: [B, 480]
        Output:
            power_future: [B, 480]
        """
        # Embed weather features
        weather_feat = self.embed(flatten(weather_future))  # [B, embed_dim]

        # Encode power context
        power_feat = self.power_encoder(power_past.unsqueeze(-1))[-1]  # [B, encoder_hidden_dim]

        # CNN features
        cnn_feat = self.cnn(power_past.unsqueeze(1))  # [B, cnn_fc_hidden]

        # Concatenate features
        features = concat([weather_feat, power_feat, cnn_feat])  # [B, total_dim]

        # Random Forest prediction (optimized by ALO)
        return self.rf.predict(features)  # [B, 480]
