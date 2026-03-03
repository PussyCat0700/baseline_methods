"""
Pseudocode: Baseline Methods Summary

This file documents 20+ baseline power forecasting methods used for comparison.
Each baseline follows the standard interface defined in baseline_interface.py.

All baselines receive the same inputs and produce the same outputs:
    Inputs: weather_past [B,120,12/15], weather_future [B,120,12/15], power_past [B,480]
    Output: power_future [B,480]
"""

# ===================================================================
# 1. Feed-Forward Neural Network (FFNN)
# ===================================================================

class FFNN_Baseline:
    """
    Simple multi-layer perceptron baseline.

    Architecture:
        Input → Flatten → MLP(3 layers) → Output

    Hyperparameters:
        - hidden_dim: 256, 512
        - num_layers: 2, 3
        - activation: ReLU

    Reference: Standard neural network baseline
    """
    pass


# ===================================================================
# 2. LSTM Baseline
# ===================================================================

class LSTM_Baseline:
    """
    Long Short-Term Memory network for time series.

    Architecture:
        Weather → LSTM → Features
        Power → LSTM → Features
        Concatenate → MLP → Output

    Hyperparameters:
        - hidden_dim: 128, 256
        - num_layers: 2, 3
        - bidirectional: True/False

    Reference: Hochreiter & Schmidhuber, 1997
    """
    pass


# ===================================================================
# 3. CNN-LSTM Hybrid
# ===================================================================

class CNN_LSTM_Baseline:
    """
    Combines CNN for feature extraction with LSTM for temporal modeling.

    Architecture:
        Weather → CNN(1D) → LSTM → Features
        Power → LSTM → Features
        Concatenate → MLP → Output

    Hyperparameters:
        - cnn_channels: [32, 64, 128]
        - lstm_hidden: 128, 256
        - kernel_size: 3, 5

    Reference: Common hybrid architecture
    """
    pass


# ===================================================================
# 4. Temporal Fusion Transformer (TFT)
# ===================================================================

class TFT_Baseline:
    """
    Temporal Fusion Transformer with attention mechanisms.

    Architecture:
        Variable Selection → LSTM Encoder → Multi-Head Attention → Output

    Key Features:
        - Variable selection network
        - Temporal self-attention
        - Gating mechanisms

    Hyperparameters:
        - hidden_dim: 128, 256
        - num_heads: 4, 8
        - num_layers: 2, 3

    Reference: Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting", 2021
    """
    pass


# ===================================================================
# 5. Attention-Based Model (ACTN)
# ===================================================================

class ACTN_Baseline:
    """
    Attention-based model for power forecasting.

    Architecture:
        Input → Embedding → Multi-Head Attention → MLP → Output

    Key Features:
        - Self-attention on temporal dimension
        - Cross-attention between weather and power

    Hyperparameters:
        - hidden_dim: 128, 256
        - num_heads: 4, 8
        - num_layers: 2, 4

    Reference: Attention mechanism for time series
    """
    pass


# ===================================================================
# 6. CrossViViT
# ===================================================================

class CrossViViT_Baseline:
    """
    Cross-attention Vision Transformer for solar forecasting.

    Architecture:
        Weather → Patch Embedding → Transformer → Cross-Attention → Output

    Key Features:
        - Patch-based processing
        - Cross-attention between modalities
        - Designed for solar forecasting

    Hyperparameters:
        - patch_size: 4, 8
        - hidden_dim: 128, 256
        - num_layers: 4, 6

    Reference: Adapted from Vision Transformer
    """
    pass


# ===================================================================
# 7. PVTransNet
# ===================================================================

class PVTransNet_Baseline:
    """
    Transformer network specifically for PV (solar) power forecasting.

    Architecture:
        Weather + Power → Transformer Encoder → Decoder → Output

    Key Features:
        - Solar-specific feature engineering
        - Temporal positional encoding
        - Multi-scale attention

    Hyperparameters:
        - hidden_dim: 128, 256
        - num_heads: 4, 8
        - num_layers: 3, 6

    Reference: Transformer for solar forecasting
    """
    pass


# ===================================================================
# 8. GEFCom 2012 Winner (Gradient Boosting)
# ===================================================================

class GEFCom12_GDBoost:
    """
    Gradient Boosting Decision Trees (winner of GEFCom 2012).

    Architecture:
        Feature Engineering → Gradient Boosting → Output

    Key Features:
        - Hand-crafted features (lag, rolling statistics)
        - Ensemble of decision trees
        - Non-parametric

    Hyperparameters:
        - n_estimators: 100, 200, 500
        - max_depth: 5, 7, 10
        - learning_rate: 0.01, 0.05, 0.1

    Reference: GEFCom 2012 competition winner
    """
    pass


# ===================================================================
# 9. HEFTCom 2024 (LightGBM)
# ===================================================================

class HEFTCom24_LGBM:
    """
    LightGBM model (HEFTCom 2024 approach).

    Architecture:
        Feature Engineering → LightGBM → Output

    Key Features:
        - Efficient gradient boosting
        - Histogram-based learning
        - Fast training

    Hyperparameters:
        - n_estimators: 100, 200, 500
        - max_depth: 5, 7, 10
        - learning_rate: 0.01, 0.05, 0.1
        - num_leaves: 31, 63, 127

    Reference: HEFTCom 2024 competition
    """
    pass


# ===================================================================
# 10. DMOM (Decomposition-based Multi-Objective Model)
# ===================================================================

class DMOM_Baseline:
    """
    Multi-scale decomposition with CNN.

    Architecture:
        Power → Wavelet Decomposition → Multi-Scale CNN → Reconstruction → Output

    Key Features:
        - Wavelet packet decomposition
        - Multi-scale feature extraction
        - Residual connections

    Hyperparameters:
        - wavelet_type: 'db4', 'sym4'
        - decomposition_level: 3, 4, 5
        - cnn_channels: [32, 64, 128]

    Reference: Multi-scale decomposition approach
    """
    pass


# ===================================================================
# 11. SC-VAR (Spatially Correlated Vector Autoregression)
# ===================================================================

class SC_VAR_Baseline:
    """
    Vector autoregression with spatial correlation.

    Architecture:
        Power + Weather → VAR Model → Spatial Correlation → Output

    Key Features:
        - Vector autoregression
        - Spatial correlation modeling
        - Linear model

    Hyperparameters:
        - lag_order: 24, 48, 96
        - spatial_neighbors: 5, 10, 20

    Reference: VAR with spatial correlation
    """
    pass


# ===================================================================
# 12. HBOLA (Hybrid LSTM with Online Learning)
# ===================================================================

class HBOLA_Baseline:
    """
    Hybrid LSTM with online adaptation.

    Architecture:
        LSTM Encoder-Decoder + Online Learning

    Key Features:
        - Encoder-decoder LSTM
        - Online learning for adaptation
        - Residual connections

    Hyperparameters:
        - hidden_dim: 128, 256
        - num_layers: 2, 3
        - online_lr: 0.001, 0.0001

    Reference: LSTM with online adaptation
    """
    pass


# ===================================================================
# 13. FusionSF (Fusion of Spatial Features)
# ===================================================================

class FusionSF_Baseline:
    """
    Fusion of spatial and temporal features for solar forecasting.

    Architecture:
        Spatial CNN + Temporal LSTM + Cross-Attention → Output

    Key Features:
        - Spatial feature extraction
        - Temporal modeling
        - Multi-modal fusion

    Hyperparameters:
        - spatial_channels: [32, 64]
        - temporal_hidden: 128, 256
        - fusion_method: 'concat', 'attention'

    Reference: Spatial-temporal fusion
    """
    pass


# ===================================================================
# 14. LSTM-GCN-MLP
# ===================================================================

class LSTM_GCN_MLP_Baseline:
    """
    Combines LSTM, Graph Convolutional Network, and MLP.

    Architecture:
        LSTM → GCN (spatial) → MLP → Output

    Key Features:
        - Temporal modeling with LSTM
        - Spatial modeling with GCN
        - Final prediction with MLP

    Hyperparameters:
        - lstm_hidden: 128, 256
        - gcn_hidden: 64, 128
        - mlp_hidden: 256, 512

    Reference: Hybrid LSTM-GCN architecture
    """
    pass


# ===================================================================
# 15. LSSVM + RBFNN
# ===================================================================

class LSSVM_RBFNN_Baseline:
    """
    Least Squares Support Vector Machine with Radial Basis Function Neural Network.

    Architecture:
        Feature Engineering → LSSVM → RBFNN → Output

    Key Features:
        - LSSVM for regression
        - RBFNN for non-linear mapping
        - Kernel-based learning

    Hyperparameters:
        - C: 1, 10, 100 (regularization)
        - gamma: 0.1, 1, 10 (kernel parameter)
        - rbf_centers: 50, 100, 200

    Reference: Hybrid SVM-NN approach
    """
    pass


# ===================================================================
# 16. WPD-LSTM (Wavelet Packet Decomposition + LSTM)
# ===================================================================

class WPD_LSTM_Baseline:
    """
    Wavelet packet decomposition with LSTM.

    Architecture:
        Power → WPD → Multiple LSTM (one per frequency band) → Reconstruction → Output

    Key Features:
        - Wavelet packet decomposition
        - Separate LSTM for each frequency band
        - Multi-resolution analysis

    Hyperparameters:
        - wavelet_type: 'db4', 'sym4'
        - decomposition_level: 3, 4
        - lstm_hidden: 64, 128

    Reference: Wavelet-LSTM hybrid
    """
    pass


# ===================================================================
# 17. RFs-ALO (Random Forests with Ant Lion Optimizer)
# ===================================================================

class RFs_ALO_Baseline:
    """
    Random Forests optimized with Ant Lion Optimizer.

    Architecture:
        Feature Engineering → Random Forests (hyperparameters optimized by ALO) → Output

    Key Features:
        - Random forest ensemble
        - Ant Lion Optimizer for hyperparameter tuning
        - Non-parametric

    Hyperparameters:
        - n_estimators: 100, 200, 500
        - max_depth: 10, 20, 30
        - min_samples_split: 2, 5, 10

    Reference: RF with metaheuristic optimization
    """
    pass


# ===================================================================
# 18. FDD-CNN (Frequency Domain Decomposition + CNN)
# ===================================================================

class FDD_CNN_Baseline:
    """
    Frequency domain decomposition with CNN.

    Architecture:
        Power → FFT → Multi-Scale CNN → IFFT → Output

    Key Features:
        - Frequency domain processing
        - Multi-scale CNN
        - Inverse transform for reconstruction

    Hyperparameters:
        - cnn_channels: [32, 64, 128]
        - kernel_sizes: [3, 5, 7]
        - num_scales: 2, 3, 4

    Reference: Frequency domain CNN
    """
    pass


# ===================================================================
# 19. GREEK (Graph-based Recurrent Network)
# ===================================================================

class GREEK_Baseline:
    """
    Graph-based recurrent network for power forecasting.

    Architecture:
        Graph Construction → GRU on Graph → Aggregation → Output

    Key Features:
        - Graph neural network
        - Recurrent processing on graph
        - Spatial-temporal modeling

    Hyperparameters:
        - gru_hidden: 128, 256
        - num_layers: 2, 3
        - graph_k: 5, 10 (k-nearest neighbors)

    Reference: Graph-based RNN
    """
    pass


# ===================================================================
# 20. Solar-Specific MLP
# ===================================================================

class Solar_MLP_Baseline:
    """
    MLP specifically designed for solar forecasting.

    Architecture:
        Solar Features → MLP → Output

    Key Features:
        - Solar-specific feature engineering
        - Time-of-day encoding
        - Clear-sky model integration

    Hyperparameters:
        - hidden_dim: 256, 512
        - num_layers: 3, 4
        - dropout: 0.1, 0.2

    Reference: Solar-specific neural network
    """
    pass


# ===================================================================
# Summary Table
# ===================================================================

"""
Baseline Methods Summary:

| Method | Type | Best For | Key Feature |
|--------|------|----------|-------------|
| FFNN | Neural Network | Simple baseline | Fully connected |
| LSTM | RNN | Temporal patterns | Long-term memory |
| CNN-LSTM | Hybrid | Feature + temporal | Convolution + RNN |
| TFT | Transformer | Multi-horizon | Variable selection |
| ACTN | Attention | Long sequences | Self-attention |
| CrossViViT | Transformer | Solar | Cross-attention |
| PVTransNet | Transformer | Solar | PV-specific |
| GEFCom12 | Boosting | Wind | Competition winner |
| HEFTCom24 | Boosting | Wind | Fast training |
| DMOM | Decomposition | Multi-scale | Wavelet |
| SC-VAR | Statistical | Linear | Spatial correlation |
| HBOLA | RNN | Online learning | Adaptation |
| FusionSF | Hybrid | Solar | Spatial fusion |
| LSTM-GCN-MLP | Graph | Spatial-temporal | GCN |
| LSSVM-RBFNN | Kernel | Non-linear | SVM + NN |
| WPD-LSTM | Decomposition | Multi-resolution | Wavelet + LSTM |
| RFs-ALO | Ensemble | Optimization | Metaheuristic |
| FDD-CNN | Frequency | Periodic patterns | FFT + CNN |
| GREEK | Graph | Spatial | Graph RNN |
| Solar-MLP | Neural Network | Solar | Solar-specific |

Hyperparameter Configurations:
    Each baseline provides 3 configurations for hyperparameter search.
    Configuration selection based on validation performance.

Training:
    - Optimizer: AdamW
    - Learning rate: 1e-4
    - Batch size: 32
    - Epochs: 10-20
    - Early stopping: patience=5

Evaluation:
    - Metrics: RMSE, MAE, R², MAPE
    - Test period: Last 20% of data
    - Temporal split (no random split)
"""
