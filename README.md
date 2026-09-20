# Renewable power forecasting comparison methods

Twenty method-specific implementations share a power/weather interface and an experimental protocol. Each file in `methods/` contains a short paper reference, the retained mechanism, task adaptations, training/prediction pseudocode, and the implementation.

## Installation

Use Python 3.12 in a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

```text
README.md          Shared interface and experiment protocol
requirements.txt   Runtime dependencies
configs.yaml       Common settings, fixed parameters and three candidates per method
run.py             Tuning, training, prediction and evaluation
common.py          Data handling, scaling, scoring and shared fitting functions
methods/           One documented Python file per method; explicit registration in __init__.py
```

## Data interface

Each site supplies `train.npz`, `validation.npz` and `test.npz`. Paths are provided through an external CSV manifest. Arrays must be finite and ordered by issue time.

| Array | Shape | Description |
|---|---|---|
| `Wp` | `[N,120,F]` | Historical hourly weather |
| `Wf` | `[N,120,F]` | Hourly forecast weather available at issue time |
| `P` | `[N,480]` | Historical power at 15-min resolution, ending at issue time |
| `Y` | `[N,480]` | Target power, beginning 15 minutes after issue time |
| `issue_times` | `[N]` | NumPy datetime64 issue times in local Beijing time |
| `target_times` | `[N,480]` | Optional; otherwise derived from issue time + 15,...,7200 minutes |

`F=15` for wind and `F=12` for solar. Weather channels retain the same order across splits. Each future hourly value covers four consecutive 15-min targets; each past hourly value aligns with four consecutive historical power points. Convert other weather timestamp conventions to this contract when preparing the arrays. `Y` is not required for prediction.

Use timezone-naive datetime arrays representing Beijing time; convert UTC timestamps before export. Date-sensitive methods use sample timestamps as metadata. Neither forecast targets nor observations from other sites are prediction inputs.

The manifest has these columns:

```csv
site_id,technology,train,validation,test
wind_001,wind,wind_001/train.npz,wind_001/validation.npz,wind_001/test.npz
solar_001,solar,solar_001/train.npz,solar_001/validation.npz,solar_001/test.npz
```

Relative file paths resolve against `--data-root`. The manifest example illustrates column syntax. The tuning manifest must contain the fixed 60 site IDs: 30 wind and 30 solar; the program does not resample them.

## Experimental protocol

All methods output 480 values. Evaluation uses the following windows from Supplementary Table 7:

| Task | Issue schedule | Output steps, one-based and inclusive | Python slice |
|---|---|---|---|
| Ultra-short | Every four hours | 1–16 | `0:16` |
| Short | Daily at 08:00 | 64–159, complete day D+1 | `63:159` |
| Medium | Daily at 08:00 | 352–447, complete day D+4 | `351:447` |

Training, validation and test target periods must not overlap. Power and weather scaling are fitted on each site's training data only; duplicated power timestamps are counted once when fitting the power scaler. Predictions are returned in the input power units. MAE and RMSE use these same units; R2 and Corr are dimensionless. The configuration applies no additional clipping or nighttime correction.

### Hyperparameter selection

Using the fixed 60-site pool (30 wind and 30 solar), evaluate the three configurations in `configs.yaml` on the 30 sites matching each method's energy type. Select the lowest site-averaged standardized validation MSE, with equal short- and medium-term weighting for SM. Save the selection to `selected.json` and freeze the hyperparameters for per-site training across the full experiment.

### Commands

```bash
python run.py tune --method cnn_lstm --config configs.yaml \
  --data-root /path/to/data --sites /path/to/fixed60.csv \
  --save-dir /path/to/runs/cnn_lstm/tuning

python run.py train --method cnn_lstm --config configs.yaml \
  --data-root /path/to/data --sites /path/to/all_sites.csv \
  --selection /path/to/runs/cnn_lstm/tuning/selected.json \
  --save-dir /path/to/runs/cnn_lstm/sites

python run.py predict \
  --checkpoint /path/to/runs/cnn_lstm/sites/solar_001/seed_42/model.pkl \
  --input /path/to/new_samples.npz --save-dir /path/to/predictions

python run.py evaluate --predictions /path/to/predictions/predictions.npz \
  --targets /path/to/targets.npz --protocol /path/to/predictions/protocol.json \
  --save-dir /path/to/evaluation
```

`evaluate` applies the saved issue schedule to both files and requires matching issue times and target shapes within that schedule; it then scores the saved task windows. `train` writes per-site metrics and site-first aggregate metrics, including the valid-site count for metrics undefined on constant series. Model files, score records and predictions are saved only to the supplied paths. Load model pickle files only from trusted sources.

### Python interface

```python
from common import load_config
from methods import create_method

settings = load_config("configs.yaml")
spec = settings["methods"]["cnn_lstm"]
protocol = dict(settings["protocol"], task=spec["task"])
model = create_method("cnn_lstm", spec["candidates"][0], spec["fixed"], protocol)
model.fit(train, validation)
power = model.predict(Wp, Wf, P, time_index=issue_times)
model.save("/path/to/model.pkl")
```

The Python example shows a single candidate fit. Use `tune` and the resulting frozen selection for the full experiment. Keep Supplementary Table 4 synchronized with the candidate fields in `configs.yaml`; selected entries follow the tuning records.
