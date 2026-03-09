# Baseline Methods Repository - Instructions for Claude

## Repository Purpose

This repository contains pseudocode documentation for 20 baseline power forecasting methods used for comparison with IResFM. For Nature Energy reviewers to understand the baseline implementations.

## Current Status

### Completed
- ✅ Pseudocode files created in `pseudocode/` directory (20 methods, 4 task categories)
- ✅ Main README.md created
- ✅ .gitignore created
- ✅ Data directory structure created (symlink)
- ✅ Initial git commit

## Repository Structure

```
.
├── README.md                        # Main documentation
├── CLAUDE.md                        # This file
├── .gitignore                       # Git ignore rules
├── pseudocode/                      # Pseudocode documentation
│   ├── short_wind/                  # 5 short-term wind methods
│   ├── short_solar/                 # 5 short-term solar methods
│   ├── ultrashort_wind/             # 5 ultrashort wind methods
│   └── ultrashort_solar/            # 5 ultrashort solar methods
└── data/                            # Open-source dataset (symlink)
    ├── README.md                    # Data documentation
    ├── info.csv                     # Station metadata
    ├── active_power_norm/           # Power CSVs
    ├── weather_finetune/            # Training weather
    └── weather_infer/               # Testing weather
```

## Baseline Methods (20 total)

### Short-term Wind (5)
1. FFNN - Feed-Forward Neural Network
2. GEFCom12 - GEFCom 2012 Winner
3. HEFTCom24 - HEFTCom 2024 Winner
4. CNN-LSTM - CNN-LSTM Hybrid
5. GREEK - Graph-based Recurrent Network

### Short-term Solar (5)
6. Solar-MLP - Solar-specific MLP
7. CrossViViT - Cross-attention Vision Transformer
8. PVTransNet - PV Transformer Network
9. CNN-LSTM - CNN-LSTM Hybrid
10. FusionSF - Fusion of Spatial Features

### Ultrashort Wind (5)
11. TFT - Temporal Fusion Transformer
12. HBOLA - Hybrid LSTM with Online Learning
13. LSSVM-RBFNN - Least Squares SVM + RBF Neural Network
14. DMOM - Decomposition-based Multi-Objective Model
15. SC-VAR - Spatially Correlated Vector Autoregression

### Ultrashort Solar (5)
16. WPD-LSTM - Wavelet Packet Decomposition + LSTM
17. ATCN - Attention-based Temporal Convolutional Network
18. LSTM-GCN-MLP - LSTM-GCN-MLP Hybrid
19. NARX-GA - Frequency Domain Decomposition + CNN
20. RFs-ALO - Random Forests with Ant Lion Optimizer

## Important Notes

1. **This is pseudocode only** - Not executable, for documentation purposes
2. **For reviewers** - To understand baseline implementations
3. **Unified interface** - All baselines follow same input/output specification
4. **Data is included** - 185 stations dataset included

## Data Source

Data directory: `data/`

## Conda Environment

Use conda environment: `zhp`

```bash
conda activate zhp
```
