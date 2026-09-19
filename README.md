# Comparison methods for IresFM

This repository provides the 20 comparison methods, their parameter settings,
Persistence formulations and evaluation metrics used in the IresFM study.

The executable package uses the same input and output interface for every
method: 120 hourly weather fields before and after the forecast issue time,
480 quarter-hourly historical power values and 480 quarter-hourly forecast
values. Wind and solar methods receive the technology-specific weather
variables listed in the paper.

## Resources

| Resource | Repository location |
|---|---|
| Comparison-method implementations | `comparison_methods/models.py` |
| Persistence formulations | `comparison_methods/persistence.py` |
| Machine-readable parameter settings | `comparison_methods/configs.py` |
| Human-readable parameter settings | `CONFIGS.md` |
| Evaluation metrics | `comparison_methods/metrics.py` |

## Repository layout

```text
.
├── README.md
├── CONFIGS.md                         # candidate and selected configurations
├── comparison_methods/
│   ├── configs.py                     # machine-readable configurations
│   ├── metrics.py                     # site-first MAE, RMSE, R2 and Corr
│   ├── models.py                      # executable model implementations
│   ├── persistence.py                 # two manuscript Persistence formulations
│   └── run.py                         # synthetic-input smoke tests
└── tests/
    └── test_comparison_methods.py
```

## Parameter settings

`CONFIGS.md` and `comparison_methods/configs.py` record the three candidate
settings and the selected setting for each method, following Supplementary
Table 4. Calling `build_model(method)` uses that method's selected setting;
`build_model(method, config_index)` selects a specific candidate (1, 2 or 3).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verify all method implementations

The smoke test constructs every method at each of its three documented
settings and checks a complete forward and backward pass with synthetic inputs:

```bash
python -m comparison_methods.run smoke-test
python -m pytest -q
```

## Persistence formulations

The manuscript uses latest-value Persistence for wind and lagged-observation
Persistence for solar. For wind, the latest power value measured at the
forecast issue time is repeated over the forecast horizon. For solar, the
observation lags are 24 h for ultra-short-term, 48 h for short-term and 120 h
for medium-term forecasting, consistent with the forecast issue times and
available observations.

## Method inventory

| Task | Methods |
|---|---|
| Short/medium wind | FFNN, GEFCom12, HEFTCom24, CNN-RBFNN, HPE |
| Short/medium solar | ANN, CrossViViT, PVTransNet, CNN-LSTM, FusionSF |
| Ultra-short wind | TFT, HBOLA, LSSVM-RBFNN, DMOM, SC-VAR |
| Ultra-short solar | WPD-LSTM, ATCN, LSTM-GCN-MLP, NARX-GA, RFs-ALO |

The literature sources, search settings and implementation notes are given in
`CONFIGS.md`.
