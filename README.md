# Baseline Methods for Power Forecasting

Pseudocode documentation for 20 baseline methods used for comparison with IResFM.

## Overview

This repository contains simplified pseudocode for baseline power forecasting methods. All baselines follow a unified interface and are trained on the same open-source dataset (185 stations).

## Repository Structure

```
.
├── README.md
├── pseudocode/
│   ├── short_wind/          # Short-term wind forecasting (5 methods)
│   ├── short_solar/         # Short-term solar forecasting (5 methods)
│   ├── ultrashort_wind/     # Ultrashort wind forecasting (5 methods)
│   └── ultrashort_solar/    # Ultrashort solar forecasting (5 methods)
└── data/                    # Open-source dataset (symlink)
```

## Baseline Methods by Task

### Short-term Wind (5 methods)
- **FFNN** - Feed-Forward Neural Network
- **GEFCom12** - GEFCom 2012 Winner (Gradient Boosting)
- **HEFTCom24** - HEFTCom 2024 Winner (LightGBM)
- **CNN-LSTM** - CNN-LSTM Hybrid
- **GREEK** - Graph-based Recurrent Network

### Short-term Solar (5 methods)
- **Solar-MLP** - Solar-specific MLP
- **CrossViViT** - Cross-attention Vision Transformer
- **PVTransNet** - PV Transformer Network
- **CNN-LSTM** - CNN-LSTM Hybrid
- **FusionSF** - Fusion of Spatial Features

### Ultrashort Wind (5 methods)
- **TFT** - Temporal Fusion Transformer
- **HBOLA** - Hybrid LSTM with Online Learning
- **LSSVM-RBFNN** - Least Squares SVM + RBF Neural Network
- **DMOM** - Decomposition-based Multi-Objective Model
- **SC-VAR** - Spatially Correlated Vector Autoregression

### Ultrashort Solar (5 methods)
- **WPD-LSTM** - Wavelet Packet Decomposition + LSTM
- **ATCN** - Attention-based Temporal Convolutional Network
- **LSTM-GCN-MLP** - LSTM-GCN-MLP Hybrid
- **FDD-CNN** - Frequency Domain Decomposition + CNN
- **RFs-ALO** - Random Forests with Ant Lion Optimizer

## Unified Interface

All baselines use the same input/output format:

**Input**:
- `weather_past`: [B, 120, 12/15] - 5 days hourly weather history
- `weather_future`: [B, 120, 12/15] - 5 days hourly weather forecast
- `power_past`: [B, 480] - 5 days power history (15-min intervals)

**Output**:
- `power_future`: [B, 480] - 5 days power forecast (15-min intervals)

Weather channels: 12 for solar, 15 for wind

## Dataset

Open-source dataset with 185 power stations:
- Wind: 611 stations
- Solar: 546 stations
- Data: Power CSVs + Weather NPY arrays
- Location: `data/` (symlinked from `/home/laohe/finetune_data_process/open`)

## Key Differences from IResFM

- **Training**: Baselines train from scratch, IResFM uses pretrained weights
- **Data**: Baselines need full station history, IResFM transfers knowledge
- **Architecture**: Baselines use various architectures, IResFM uses Swin Transformer

## Hyperparameter Configurations

Each baseline method was tested with 3 different hyperparameter configurations:
- **Config 1**: Typically larger/more complex model
- **Config 2**: Medium-sized model
- **Config 3**: Smaller/simpler model

The best performing configuration for each method is documented in `CONFIGS.md`.

**Configuration Selection Process**:
1. Tested all 3 configs on 60-station subset
2. Selected best config based on validation performance
3. Used best config for full training on all stations

See `CONFIGS.md` for complete configuration details.

## Notes

- This is **pseudocode** for documentation, not executable code
- All baselines use the same data and evaluation protocol
- Each baseline tested with 3 hyperparameter configurations (see CONFIGS.md)
- Original implementation: `/home/laohe/dev_yfliu/upstream_backup/ne_baseline/`
