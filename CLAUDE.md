# Baseline Methods Repository - Instructions for Claude

## Repository Purpose

This repository contains pseudo-code documentation for 20+ baseline power forecasting methods used for comparison with IResFM. For Nature Energy reviewers to understand the baseline implementations.

## Current Status

### Completed
- ✅ Pseudo-code files created in `pseudocode/` directory:
  - `baseline_interface.py` - Common interface for all baselines
  - `baselines_summary.py` - Summary of 20+ baseline methods
- ✅ Main README.md created
- ✅ .gitignore created
- ✅ Data directory structure created

### Pending
- ⬜ Initial git commit
- ⬜ Verify data integrity
- ⬜ Optional: Create individual pseudo-code files for each baseline (currently summarized in baselines_summary.py)
- ⬜ Optional: Add LICENSE file

## Next Steps

1. **Verify data copying is complete**:
   ```bash
   ls -la data/
   # Should see: info.csv, active_power_norm/, weather_finetune/, weather_infer/, README.md
   ```

2. **Make initial git commit**:
   ```bash
   cd /home/laohe/dev_yfliu/pseudocode/baseline_methods
   git add .
   git commit -m "Initial commit: Baseline methods pseudo-code documentation

   - Add pseudo-code for 20+ baseline methods
   - Include unified interface for fair comparison
   - Include open-source dataset with 185 stations
   - Document baseline architectures and hyperparameters"
   ```

3. **Optional: Set up remote repository**:
   ```bash
   git remote add origin <repository-url>
   git push -u origin master
   ```

## Repository Structure

```
.
├── README.md                        # Main documentation
├── CLAUDE.md                        # This file
├── .gitignore                       # Git ignore rules
├── pseudocode/                      # Pseudo-code documentation
│   ├── baseline_interface.py        # Common interface
│   └── baselines_summary.py         # 20+ baseline methods
└── data/                            # Open-source dataset
    ├── README.md                    # Data documentation
    ├── info.csv                     # Station metadata
    ├── active_power_norm/           # Power CSVs
    ├── weather_finetune/            # Training weather
    └── weather_infer/               # Testing weather
```

## Baseline Methods Included

1. FFNN - Feed-Forward Neural Network
2. LSTM - Long Short-Term Memory
3. CNN-LSTM - CNN-LSTM Hybrid
4. TFT - Temporal Fusion Transformer
5. ACTN - Attention-based Model
6. CrossViViT - Cross-attention Vision Transformer
7. PVTransNet - PV Transformer Network
8. GEFCom12 - GEFCom 2012 Winner (GDBoost)
9. HEFTCom24 - HEFTCom 2024 (LightGBM)
10. DMOM - Decomposition-based Multi-Objective Model
11. SC-VAR - Spatially Correlated VAR
12. HBOLA - Hybrid LSTM with Online Learning
13. FusionSF - Fusion of Spatial Features
14. LSTM-GCN-MLP - LSTM-GCN-MLP Hybrid
15. LSSVM-RBFNN - LSSVM + RBFNN
16. WPD-LSTM - Wavelet Packet Decomposition + LSTM
17. RFs-ALO - Random Forests with Ant Lion Optimizer
18. FDD-CNN - Frequency Domain Decomposition + CNN
19. GREEK - Graph-based Recurrent Network
20. Solar-MLP - Solar-specific MLP

## Important Notes

1. **This is pseudo-code only** - Not executable, for documentation purposes
2. **For Nature Energy reviewers** - To understand baseline implementations
3. **Unified interface** - All baselines follow same input/output specification
4. **Data is included** - Same 185 stations as main method repository
5. **Original repository** - Located at `/home/laohe/dev_yfliu/upstream_backup/ne_baseline/` (reference only)

## Data Source

Data dir `data` linked from: `/home/laohe/finetune_data_process/open`

## Conda Environment

Use conda environment: `zhp`

```bash
conda activate zhp
```

## Related Repository

Main method repository: `/home/laohe/dev_yfliu/pseudocode/iresfm_main_method/`

## Optional Enhancements

If more detail is needed for reviewers, consider:

1. **Create individual baseline files**: Split `baselines_summary.py` into separate files for each baseline method in `pseudocode/baselines/` directory

2. **Add training pseudo-code**: Create `baseline_training.py` and `baseline_evaluation.py` with detailed workflows

3. **Add comparison tables**: Document performance comparisons between baselines and main method

4. **Add visualization examples**: Show example predictions and error analysis
