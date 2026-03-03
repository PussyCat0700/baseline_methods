# Baseline Methods - Complete Configuration Reference

All tested configurations for 20 baseline methods. Best configurations are marked with ✓.

## Short-term Wind

### 1. FFNN (ExampleMLP) - Best: Config 3
```python
Config 1: hidden_dim=512, num_layers=12
Config 2: hidden_dim=256, num_layers=8
Config 3: hidden_dim=128, num_layers=4  ✓
```

### 2. GEFCom12 (GEFCom12GDBoostModel) - Best: Config 3
```python
Config 1: n_estimators=5, num_leaves=16, embed_dim=64, power_ctx_dim=32, encoder_hidden_dim=128
Config 2: n_estimators=10, num_leaves=32, embed_dim=128, power_ctx_dim=64, encoder_hidden_dim=256
Config 3: n_estimators=20, num_leaves=64, embed_dim=256, power_ctx_dim=128, encoder_hidden_dim=512  ✓
```

### 3. HEFTCom24 (HEFTCom24LGBMModelv2) - Best: Config 2
```python
Config 1: n_estimators=100, num_leaves=200, embed_dim=128, power_ctx_dim=64, encoder_hidden_dim=256
Config 2: n_estimators=200, num_leaves=400, embed_dim=128, power_ctx_dim=64, encoder_hidden_dim=256  ✓
Config 3: n_estimators=400, num_leaves=700, embed_dim=64, power_ctx_dim=32, encoder_hidden_dim=128
```

### 4. CNN-LSTM (CNNLSTMModel) - Best: Config 3
```python
Config 1: hidden_size=64, num_layers=1
Config 2: hidden_size=128, num_layers=2
Config 3: hidden_size=256, num_layers=4  ✓
```

### 5. GREEK (GREEKModel) - Best: Config 3
```python
Config 1: n_estimators=50, max_features=32, embed_dim=64, power_ctx_dim=32, encoder_hidden_dim=128
Config 2: n_estimators=100, max_features=64, embed_dim=128, power_ctx_dim=64, encoder_hidden_dim=256
Config 3: n_estimators=150, max_features=128, embed_dim=256, power_ctx_dim=128, encoder_hidden_dim=512  ✓
```

## Short-term Solar

### 6. Solar-MLP (SolarMLPModel) - Best: Config 2
```python
Config 1: hidden_dim=512, num_hidden_layers=3, dropout=0.3, time_encoding_dim=32
Config 2: hidden_dim=384, num_hidden_layers=2, dropout=0.2, time_encoding_dim=24  ✓
Config 3: hidden_dim=256, num_hidden_layers=2, dropout=0.2, time_encoding_dim=16
```

### 7. CrossViViT (CrossViViTModelv2) - Best: Config 3
```python
Config 1: dim=384, depth=16, heads=12, dim_head=64, mlp_ratio=4, dropout=0.4, use_glu=True,
          depth_cross=4, decoder_dim=128, decoder_depth=4, decoder_heads=6, decoder_dim_head=128,
          use_weather_past=False
Config 2: dim=256, depth=12, heads=8, dim_head=64, mlp_ratio=4, dropout=0.3, use_glu=True,
          depth_cross=3, decoder_dim=96, decoder_depth=3, decoder_heads=4, decoder_dim_head=96,
          use_weather_past=False
Config 3: dim=128, depth=8, heads=6, dim_head=64, mlp_ratio=4, dropout=0.2, use_glu=True,
          depth_cross=2, decoder_dim=64, decoder_depth=2, decoder_heads=3, decoder_dim_head=64,
          use_weather_past=False  ✓
```

### 8. PVTransNet (PVTransNetE) - Best: Config 3
```python
Config 1: emb_dim=128, num_layers=1, num_heads=2, head_dim=64
Config 2: emb_dim=256, num_layers=4, num_heads=4, head_dim=128
Config 3: emb_dim=512, num_layers=12, num_heads=8, head_dim=128  ✓
```

### 9. CNN-LSTM (PVAttnCNNModel) - Best: Config 2
```python
Config 1: d_model=128, nhead=4, cnn_layers_past=2, cnn_layers_future=2, look_back_steps=96
Config 2: d_model=256, nhead=8, cnn_layers_past=3, cnn_layers_future=2, look_back_steps=192, ffn_dim=1024  ✓
Config 3: d_model=128, nhead=4, cnn_layers_past=4, cnn_layers_future=3, look_back_steps=48, dropout=0.2
```

### 10. FusionSF (PVCrossAttentionUltraShortModelv2) - Best: Config 2
```python
Config 1: d_model=128, nhead=2, num_decoder_layers=3, look_back_steps=48
Config 2: d_model=256, nhead=4, num_decoder_layers=6, look_back_steps=96  ✓
Config 3: d_model=512, nhead=8, num_decoder_layers=12, look_back_steps=96
```

## Ultrashort Wind

### 11. TFT (Wang2024StaticTFTModelv2) - Best: Config 1
```python
Config 1: d_model=112, lstm_layers=9, num_heads=7, dropout=0.1  ✓
Config 2: d_model=56, lstm_layers=7, num_heads=7, dropout=0.1
Config 3: d_model=49, lstm_layers=5, num_heads=7, dropout=0.1
```

### 12. HBOLA (WindLSTMEncoderDecoderResidualv2) - Best: Config 2
```python
Config 1: num_layers=5, hidden_size=64, dropout=0.1
Config 2: num_layers=10, hidden_size=96, dropout=0.1  ✓
Config 3: num_layers=20, hidden_size=128, dropout=0.15
```

### 13. LSSVM-RBFNN (LSSVMNNModel) - Best: Config 3
```python
Config 1: embed_dim=64, encoder_hidden_dim=128, encoder_num_layers=2, num_prototypes=64, gamma=0.8
Config 2: embed_dim=128, encoder_hidden_dim=256, encoder_num_layers=2, num_prototypes=128, gamma=0.5
Config 3: embed_dim=128, encoder_hidden_dim=384, encoder_num_layers=3, num_prototypes=256, gamma=0.3  ✓
```

### 14. DMOM (MultiScaleFDDCNNResidualPowerModelv2) - Best: Config 3
```python
Config 1: c_hidden=32, alpha=0.3, fc_hidden_dim=128, dropout=0.05
Config 2: c_hidden=48, alpha=0.5, fc_hidden_dim=256, dropout=0.10
Config 3: c_hidden=64, alpha=0.7, fc_hidden_dim=384, dropout=0.15  ✓
```

### 15. SC-VAR (SoftMaskFDDCNNResidualPowerModelv2) - Best: Config 3
```python
Config 1: hidden_dim=192, num_layers=2, dropout=0.15
Config 2: hidden_dim=320, num_layers=3, dropout=0.20
Config 3: hidden_dim=512, num_layers=4, dropout=0.25  ✓
```

## Ultrashort Solar

### 16. WPD-LSTM (MultiScaleFDDCNNResidualPowerModel) - Best: Config 2
```python
Config 1: c_hidden=32, alpha=0.3, fc_hidden_dim=128, dropout=0.05
Config 2: c_hidden=48, alpha=0.5, fc_hidden_dim=256, dropout=0.10  ✓
Config 3: c_hidden=64, alpha=0.7, fc_hidden_dim=384, dropout=0.15
```

### 17. ATCN (TriBandFDDCNNResidualPowerModel) - Best: Config 1
```python
Config 1: cutoff1_frac=0.06, cutoff2_frac=0.18, conv1_channels=16, conv2_channels=32,
          fc_hidden_dim=128, near_horizon_len=16  ✓
Config 2: cutoff1_frac=0.08, cutoff2_frac=0.20, conv1_channels=32, conv2_channels=64,
          fc_hidden_dim=256, near_horizon_len=16
Config 3: cutoff1_frac=0.10, cutoff2_frac=0.24, conv1_channels=32, conv2_channels=96,
          fc_hidden_dim=256, near_horizon_len=8, dropout=0.2
```

### 18. LSTM-GCN-MLP (SoftMaskFDDCNNResidualPowerModelv3) - Best: Config 3
```python
Config 1: hidden_dim=64, num_layers=2, dropout=0.1, conv1_channels=16, conv2_channels=32,
          cutoff_frac=0.10, soft_temp=1.5
Config 2: hidden_dim=96, num_layers=3, dropout=0.1, conv1_channels=24, conv2_channels=48,
          cutoff_frac=0.12, soft_temp=2.0
Config 3: hidden_dim=128, num_layers=3, dropout=0.15, conv1_channels=32, conv2_channels=64,
          cutoff_frac=0.15, soft_temp=2.5  ✓
```

### 19. FDD-CNN (Yan2021FDDCNNModel) - Best: Config 3
```python
Config 1: cutoff_frac=0.08, conv1_channels=16, conv2_channels=32, fc_hidden_dim=128
Config 2: cutoff_frac=0.12, conv1_channels=32, conv2_channels=64, fc_hidden_dim=256
Config 3: cutoff_frac=0.16, conv1_channels=32, conv2_channels=96, fc_hidden_dim=256  ✓
```

### 20. RFs-ALO (Ibrahim2020RFALOModelv2) - Best: Config 2
```python
Config 1: num_trees=500, num_leaves=10, embed_dim=128, power_ctx_dim=128, encoder_hidden_dim=256,
          cnn_c_hidden=48, cnn_fc_hidden=256, cutoff_frac=0.12, dropout=0.1
Config 2: num_trees=1000, num_leaves=5, embed_dim=64, power_ctx_dim=64, encoder_hidden_dim=128,
          cnn_c_hidden=32, cnn_fc_hidden=128, cutoff_frac=0.10, dropout=0.15  ✓
Config 3: num_trees=300, num_leaves=20, embed_dim=96, power_ctx_dim=96, encoder_hidden_dim=192,
          cnn_c_hidden=40, cnn_fc_hidden=192, cutoff_frac=0.14, dropout=0.12
```

## Notes

- All configurations extracted from original repository (verified against scheduler.py)
- All configurations tested on 60-station subset during hyperparameter search
- Best configurations selected based on validation performance and documented in scheduler.py
- Original model class names in parentheses for reference

## Model Name Mapping (from scheduler.py)

**Short-term Wind:**
- HEFTCom24LGBMModelv2 → HEFTCom24
- GEFCom12GDBoostModel → GEFCom12
- ExampleMLP → FFNN
- GREEKModel → GREEK
- CNNLSTMModel → CNN-LSTM

**Short-term Solar:**
- SolarMLPModel → Solar-MLP
- CrossViViTModelv2 → CrossViViT
- PVTransNetE → PVTransNet
- PVAttnCNNModel → CNN-LSTM
- PVCrossAttentionUltraShortModelv2 → FusionSF

**Ultrashort Wind:**
- WindLSTMEncoderDecoderResidualv2 → HBOLA
- Wang2024StaticTFTModelv2 → TFT
- LSSVMNNModel → LSSVM-RBFNN
- MultiScaleFDDCNNResidualPowerModelv2 → DMOM
- SoftMaskFDDCNNResidualPowerModelv2 → SC-VAR

**Ultrashort Solar:**
- MultiScaleFDDCNNResidualPowerModel → WPD-LSTM
- TriBandFDDDilatedCNNResidualModel → ATCN
- SoftMaskFDDCNNResidualPowerModelv3 → LSTM-GCN-MLP
- Yan2021FDDCNNModel → FDD-CNN
- Ibrahim2020RFALOModelv2 → RFs-ALO
