# Baseline Methods for Power Forecasting

**Pseudo-Code Documentation for 20+ Baseline Comparison Methods**

This repository contains pseudo-code documentation for baseline power forecasting methods used for comparison with the main IResFM model. All baselines follow a unified interface and are trained on the same data.

## Purpose

This repository provides:
- **Pseudo-code documentation** for 20+ baseline methods
- **Unified interface** for fair comparison
- **Open-source dataset** with 185 renewable energy stations
- **Implementation guidance** for researchers

**Note**: This repository contains pseudo-code for documentation purposes, not executable code.

## Repository Structure

```
.
├── README.md                        # This file
├── pseudocode/                      # Pseudo-code documentation
│   ├── baseline_interface.py        # Common interface for all baselines
│   └── baselines_summary.py         # Summary of 20+ baseline methods
└── data/                            # Open-source dataset (185 stations)
    ├── README.md                    # Data documentation
    ├── info.csv                     # Station metadata
    ├── active_power_norm/           # Power data (15-min resolution)
    ├── weather_finetune/            # Training weather (12h forecasts)
    └── weather_infer/               # Testing weather (131h forecasts)
```

## Baseline Methods

### Method Categories

1. **Neural Networks**: FFNN, LSTM, CNN-LSTM
2. **Transformers**: TFT, ACTN, CrossViViT, PVTransNet
3. **Boosting**: GEFCom12 (GDBoost), HEFTCom24 (LightGBM)
4. **Decomposition**: DMOM, WPD-LSTM, FDD-CNN
5. **Statistical**: SC-VAR
6. **Hybrid**: HBOLA, FusionSF, LSTM-GCN-MLP, LSSVM-RBFNN
7. **Ensemble**: RFs-ALO
8. **Graph-based**: GREEK
9. **Task-specific**: Solar-MLP

### Baseline Summary Table

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

## Unified Interface

All baseline methods follow the same interface:

### Input Specification

```python
input_weather_past: [B, 120, 12/15]    # 5 days hourly weather history
input_weather_future: [B, 120, 12/15]  # 5 days hourly weather forecast
input_power_past: [B, 480]             # 5 days power history (15-min)
```

- Weather channels: 12 for solar, 15 for wind
- B: batch size

### Output Specification

```python
output_power: [B, 480]  # 5 days power forecast (15-min)
```

### Training Output

```python
{
    "loss": scalar,      # MSE loss
    "output": [B, 480]   # Predictions
}
```

## Weather Variables

### Wind (15 channels)
1-2. u10, v10: 10m wind components
3-4. u100, v100: 100m wind components
5-6. u200, v200: 200m wind components
7. t2m: 2m temperature
8. sp: Surface pressure
9. tcc: Total cloud cover
10. tp: Total precipitation
11-15. z1000, q1000, t1000, u1000, v1000: 1000 hPa variables

### Solar (12 channels)
1-2. u10, v10: 10m wind components
3. t2m: 2m temperature
4. sp: Surface pressure
5. ssr: Surface solar radiation
6. tcc: Total cloud cover
7. tp: Total precipitation
8-12. z1000, q1000, t1000, u1000, v1000: 1000 hPa variables

## Training Configuration

### Hyperparameters

Each baseline provides 3 configurations for hyperparameter search:
- Config 1: Larger model
- Config 2: Medium model
- Config 3: Smaller model

### Training Settings

- **Optimizer**: AdamW
- **Learning rate**: 1e-4
- **Batch size**: 32
- **Epochs**: 10-20
- **Early stopping**: patience=5
- **Loss function**: MSE

### Data Split

- **Training**: First 80% of time period
- **Testing**: Last 20% of time period
- **Temporal split** (no random split to avoid data leakage)

## Evaluation Metrics

- **RMSE**: Root Mean Square Error (MW)
- **MAE**: Mean Absolute Error (MW)
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error (%)

## Dataset

Same dataset as main method repository:
- **Stations**: 185 (wind and solar)
- **Time Period**: ~1 year per station
- **Power Resolution**: 15-minute intervals
- **Weather Resolution**: Hourly

See `data/README.md` for detailed documentation.

## Implementation Guide

### 1. Implement Base Class

```python
class YourBaseline(BaseSFTModel):
    def __init__(self, args):
        super().__init__(args)
        # Initialize your model layers

    def _forward_impl(self, weather_past, weather_future, power_past):
        # Implement your forward logic
        # Return: [B, 480]
        pass
```

### 2. Define Hyperparameters

```python
class YourBaselineArguments(BaseSFTModelArguments):
    def __init__(self, hidden_dim=256, task_type="wind"):
        super().__init__(task_type)
        self.hidden_dim = hidden_dim

    @classmethod
    def get_test_configs(cls):
        return [
            cls(hidden_dim=512),  # Config 1
            cls(hidden_dim=256),  # Config 2
            cls(hidden_dim=128),  # Config 3
        ]
```

### 3. Train and Evaluate

```python
# Training
model = YourBaseline(args)
trained_model = train_baseline_model(model, train_dataset, valid_dataset, config)

# Evaluation
results = evaluate_baseline_model(trained_model, test_dataset, station_id)
```

## Comparison with Main Method

The main IResFM model differs from baselines in:

1. **Architecture**: Swin Transformer vs. simpler architectures
2. **Training**: Two-stage (pretrain + finetune) vs. single-stage
3. **Input**: Spatial weather (8×8 patch) vs. averaged weather
4. **Parameters**: ~88M vs. typically < 10M for baselines

## Citation

If you use these baseline methods in your research, please cite:

```
[Citation information will be added upon publication]
```

## License

[License information will be added]

## Acknowledgments

- Baseline methods adapted from published literature
- Dataset same as main method repository
- Unified interface for fair comparison

## Related Work

- GEFCom 2012: Global Energy Forecasting Competition
- HEFTCom 2024: Hierarchical Energy Forecasting Competition
- Various published methods from renewable energy forecasting literature
