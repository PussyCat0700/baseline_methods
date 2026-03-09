# Baseline Methods for Power Forecasting

Pseudocode documentation for 20 baseline methods used for comparison with IResFM.

## Overview

This repository contains simplified pseudocode for baseline power forecasting methods. All baselines follow a unified interface and are trained on the same dataset.

## Repository Structure

```
.
├── README.md
├── pseudocode/
│   ├── short_wind/          # Short-term wind forecasting (5 methods)
│   ├── short_solar/         # Short-term solar forecasting (5 methods)
│   ├── ultrashort_wind/     # Ultrashort wind forecasting (5 methods)
│   └── ultrashort_solar/    # Ultrashort solar forecasting (5 methods)
└── data/                    # Open-source dataset path
```

## Baseline Methods by Task

### Short-term Wind (5 methods)
- **FFNN** - AWNN-Assisted Wind Power Forecasting Using Feed-Forward Neural Network <sup>1</sup>
- **GEFCom12** - A feature engineering approach to wind power forecasting <sup>2</sup>
- **HEFTCom24** - A Hybrid Strategy for Aggregated Probabilistic Forecasting and Energy Trading in HEFTCom2024 <sup>3</sup>
- **CNN-LSTM** - A hybrid deep learning-based neural network for 24-h ahead wind power forecasting <sup>4</sup>
- **GREEK** - Short-term Renewable Energy Forecasting in Greece using Prophet Decomposition and Tree-based Ensembles <sup>5</sup>

### Short-term Solar (5 methods)
- **Solar-MLP** - Selection of most relevant input parameters using WEKA for artificial neural network based solar radiation prediction models <sup>6</sup>
- **CrossViViT** - Improving day-ahead Solar Irradiance Time Series Forecasting by Leveraging Spatio Temporal Context <sup>7</sup>
- **PVTransNet** - Multi-step photovoltaic power forecasting using transformer and recurrent neural networks <sup>8</sup>
- **CNN-LSTM** - CNN-LSTM: An efficient hybrid deep learning architecture for predicting short-term photovoltaic power production <sup>9</sup>
- **FusionSF** - FusionSF: Fuse Heterogeneous Modalities in a Vector Quantized Framework for Robust Solar Power Forecasting <sup>10</sup>

### Ultrashort Wind (5 methods)
- **TFT** - Very short-term wind power forecasting considering static data: An improved transformer model <sup>11</sup>
- **HBOLA** - Hedge backpropagation based online LSTM architecture for ultra-short-term wind power forecasting <sup>12</sup>
- **LSSVM-RBFNN** - Hybrid Forecasting Model for Very-short Term Wind Power Forecasting Based on Grey Relational Analysis and Wind Speed Distribution Features <sup>13</sup>
- **DMOM** - Ultra-short-term Wind Power Forecasting Based on the Strategy of “Dynamic Matching and Online Modeling” <sup>14</sup>
- **SC-VAR** - Correlation-Constrained and Sparsity-Controlled Vector Autoregressive Model for Spatio Temporal Wind Power Forecasting <sup>15</sup>

### Ultrashort Solar (5 methods)
- **WPD-LSTM** - A hybrid deep learning model for short-term PV power forecasting <sup>16</sup>
- **ATCN** - Ultra-Short-Term Spatiotemporal Forecasting of Renewable Resources: An Attention Temporal Convolutional Network-Based Approach <sup>17</sup>
- **LSTM-GCN-MLP** - Ultra-Short-Term Forecasting of Large Distributed Solar PV Fleets Using Sparse Smart Inverter Data <sup>18</sup>
- **NARX-GA** - Frequency-domain decomposition and deep learning based solar PV power ultra-short-term forecasting model <sup>19</sup>
- **RFs-ALO** - An Optimized Offline Random Forests-Based Model for Ultra-short-term Prediction of PV Characteristics <sup>20</sup>

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

Open-source dataset with real power stations:
- Data: Power CSVs + Weather NPY arrays
- Location: `data/` directory

## Key Differences from IResFM

- **Training**: Baselines train from scratch, IResFM uses pretrained weights
- **Data**: Baselines need full station history, IResFM transfers knowledge

## Hyperparameter Configurations

Each baseline method was tested with 3 different hyperparameter configurations.
The best performing configuration for each method is documented in `CONFIGS.md`.

**Configuration Selection Process**:
1. Tested all 3 configs on 60-station subset
2. Selected best config based on validation performance
3. Used best config for full training on all stations

See `CONFIGS.md` for complete configuration details.

# References
 <sup>1</sup> Bhaskar, Kanna, and Sri Niwas Singh. "AWNN-assisted wind power forecasting using feed-forward neural network." IEEE transactions on sustainable energy 3.2 (2012): 306-315.

 <sup>2</sup>Silva, Lucas. "A feature engineering approach to wind power forecasting: GEFCom 2012." International Journal of Forecasting 30.2 (2014): 395-401.

 <sup>3</sup> Pu, Chuanqing, et al. "A hybrid strategy for aggregated probabilistic forecasting and energy trading in HEFTCom2024." arXiv e-prints (2025): arXiv-2505.

 <sup>4</sup> Hong, Ying-Yi, and Christian Lian Paulo P. Rioflorido. "A hybrid deep learning-based neural network for 24-h ahead wind power forecasting." Applied Energy 250 (2019): 530-539.

 <sup>5</sup> Vartholomaios, Argyrios, et al. "Short-term renewable energy forecasting in greece using prophet decomposition and tree-based ensembles." International conference on database and expert systems applications. Cham: Springer International Publishing, 2021.

 <sup>6</sup> Yadav, Amit Kumar, Hasmat Malik, and S. S. Chandel. "Selection of most relevant input parameters using WEKA for artificial neural network based solar radiation prediction models." Renewable and Sustainable Energy Reviews 31 (2014): 509-519.

 <sup>7</sup> Boussif, Oussama, et al. "Improving* day-ahead* solar irradiance time series forecasting by leveraging spatio-temporal context." Advances in Neural Information Processing Systems 36 (2023): 2342-2367.

 <sup>8</sup> Kim, Jimin, et al. "Multi-step photovoltaic power forecasting using transformer and recurrent neural networks." Renewable and Sustainable Energy Reviews 200 (2024): 114479.

 <sup>9</sup> Agga, Ali, et al. "CNN-LSTM: An efficient hybrid deep learning architecture for predicting short-term photovoltaic power production." Electric Power Systems Research 208 (2022): 107908.

 <sup>10</sup> Ma, Ziqing, et al. "Fusionsf: Fuse heterogeneous modalities in a vector quantized framework for robust solar power forecasting." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024.

 <sup>11</sup> Wang, Sen, et al. "Very short-term wind power forecasting considering static data: An improved transformer model." Energy 312 (2024): 133577.

 <sup>12</sup> Pan, Chunyang, et al. "Hedge backpropagation based online LSTM architecture for ultra-short-term wind power forecasting." IEEE Transactions on Power Systems 39.2 (2023): 4179-4192.

 <sup>13</sup> Shi, Jie, et al. "Hybrid forecasting model for very-short term wind power forecasting based on grey relational analysis and wind speed distribution features." IEEE Transactions on Smart Grid 5.1 (2013): 521-526.

 <sup>14</sup> Li, Yuhao, et al. "Ultra-short-term wind power forecasting based on the strategy of “dynamic matching and online modeling”." IEEE Transactions on Sustainable Energy 16.1 (2024): 107-123.

 <sup>15</sup> Zhao, Yongning, et al. "Correlation-constrained and sparsity-controlled vector autoregressive model for spatio-temporal wind power forecasting." IEEE Transactions on Power Systems 33.5 (2018): 5029-5040.

 <sup>16</sup> Li, Pengtao, et al. "A hybrid deep learning model for short-term PV power forecasting." Applied Energy 259 (2020): 114216.

 <sup>17</sup> Liang, Junkai, and Wenyuan Tang. "Ultra-short-term spatiotemporal forecasting of renewable resources: An attention temporal convolutional network-based approach." IEEE Transactions on Smart Grid 13.5 (2022): 3798-3812.

 <sup>18</sup> Yue, Han, et al. "Ultra-short-term forecasting of large distributed solar PV fleets using sparse smart inverter data." IEEE transactions on sustainable energy 15.3 (2024): 1968-1980.

 <sup>19</sup> Yan, Jichuan, et al. "Frequency-domain decomposition and deep learning based solar PV power ultra-short-term forecasting model." IEEE Transactions on Industry Applications 57.4 (2021): 3282-3295.

 <sup>20</sup> Ibrahim, Ibrahim Anwar, M. J. Hossain, and Benjamin C. Duck. "An optimized offline random forests-based model for ultra-short-term prediction of PV characteristics." IEEE Transactions on Industrial Informatics 16.1 (2019): 202-214.